# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, Union, List

import torch

import transformers

from ferret.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from ferret.train.ferret_trainer import FERRETTrainer

from ferret import conversation as conversation_lib
from ferret.model import *
from ferret.mm_utils import tokenizer_image_token

from PIL import Image
import torch.nn as nn
import random
import math
import pdb
from pycocotools import mask as mask_util
import numpy as np
import re

DEFAULT_REGION_FEA_TOKEN = "<region_fea>"
VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    add_region_feature: bool = False
    region_geo_sampler: bool = False
    sampler_pooler_mode: str = field(default='mean')    # Support 'mean' or 'max'
    no_coor: bool = False
    save_vision_tower: bool = field(default=False)


@dataclass
class DataArguments:
    # data_path: str = field(default=None,
    #                        metadata={"help": "Path to the training data."})
    data_path: List[str] = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_multiple: List[float] = field(default=None,
                           metadata={"help": "Data mutliplier for each dataset when mixed. None means direct concat."})    
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: List[str] = field(default=None)
    image_aspect_ratio: str = 'square_nocrop'  # Original Default:'square'
    resized_image_h: int = 336  #  224
    resized_image_w: int = 336  #  224
    point_input_sample: str = 'segment_mask|uniform'  # 'segment_mask|uniform', 'segment_mask|center', 'segment_mask|gaussian', 'gaussian', 'center'
    refer_previous_point: bool = True
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   save_vision_tower: bool):
    """Collects the state dict and dump to disk."""
    output_dir = os.path.join(output_dir, 'final-checkpoint')
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        os.makedirs(output_dir, exist_ok=True)

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    if save_vision_tower:
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                vision_tower_folder = os.path.join(parent_folder, "vision_tower")
            else:
                vision_tower_folder = os.path.join(output_dir, "vision_tower")
            trainer.model.model.get_vision_tower().vision_tower.save_pretrained(vision_tower_folder)
            print(f'Save vision tower ckpt to {vision_tower_folder}')


    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # print('sources["value"]:', [j["value"] for j in sources[0]])
    # raise NotImplementedError()
    # pdb.set_trace()
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # pdb.set_trace()
    # print('conversations:', conversations)
    # print('input_ids', input_ids)
    # print('target', target)
    # raise NotImplementedError()

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    # add_region_feature: bool=False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    # pdb.set_trace()
        # if add_region_feature:
            # target[target==tokenizer.convert_tokens_to_ids([DEFAULT_REGION_FEA_TOKEN])[0]] = IGNORE_INDEX


    return dict(input_ids=input_ids, labels=targets)

def extend_list(original_list, multiplier):
    # Calculate how many elements to replicate and how many to select randomly
    replicate_elements = math.floor(multiplier)
    random_elements = multiplier - replicate_elements

    # Replicate the list
    replicated_list = original_list * replicate_elements

    # Calculate how many elements to randomly select
    select_elements = math.ceil(len(original_list) * random_elements)

    # Randomly select elements and append to the replicated list
    for _ in range(select_elements):
        random_element = random.choice(original_list)
        replicated_list.append(random_element)

    return replicated_list


def extract_coors(s):
    # Regex pattern to match brackets content
    brackets_pattern = r'\[(.*?)\]'

    # Regex pattern to match values
    values_pattern = r'=\s*([^,\]]+)'

    # Find all bracketed strings
    brackets = re.findall(brackets_pattern, s)

    # Define a list to hold the list of values
    values_list = []

    # Extract values from each bracketed string
    for bracket in brackets:
        # Find all matches in the string
        matches = re.findall(values_pattern, bracket)
        # Convert matches to integers and add to values_list
        values_list.append([int(match) for match in matches])

    return values_list

def regulate_box(box, img_w, img_h):
    return [max(0, min(box[0], img_w-1)), max(0, min(box[1], img_h-1)), max(0, min(box[2], img_w-1)), max(0, min(box[3], img_h-1))]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def load_vg_object(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'vg_object'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_vg_yesno_object(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'vg_yesno_object'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_vg_attribute(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'vg_attribute'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_vg_relation(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'vg_relation'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_vg_region(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'vg_region'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_git_instruction(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'git_instruction'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas
        
    def load_llava(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'llava_instruct'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_grounded_llava_boxes(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i["dataset"] = "grounded_llava_boxes"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i["image"])
        return datas

    def load_refexp(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i["dataset"] = "refexp"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_flickr(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i["dataset"] = "flickr"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_objects365(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i["dataset"] = "objects365"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_cc3m(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        for data_i in datas:
            data_i['dataset'] = 'cc3m_595k'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model_args: DataArguments,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        data_multiple = data_args.data_multiple
        if not isinstance(data_path, list):
            data_path = [data_path]

        image_folders = data_args.image_folder
        if not isinstance(data_args.image_folder, list):
            image_folders = [data_args.image_folder]


        list_data_dict = []
        for data_path_i, image_folder_i in zip(data_path, image_folders):
            if 'vg_object' in data_path_i:
                logging.warning(f"Loading vg_object data")
                list_data_dict.append(self.load_vg_object(data_path_i, image_folder_i))
            elif 'vg_yesno_object' in data_path_i:
                logging.warning(f"Loading vg_yesno_object data")
                list_data_dict.append(self.load_vg_yesno_object(data_path_i, image_folder_i))
            elif 'vg_attribute' in data_path_i:
                logging.warning(f"Loading vg_attribute data")
                list_data_dict.append(self.load_vg_attribute(data_path_i, image_folder_i))
            elif 'vg_relation' in data_path_i:
                logging.warning(f"Loading vg_relation data")
                list_data_dict.append(self.load_vg_relation(data_path_i, image_folder_i))
            elif 'vg_region' in data_path_i:
                logging.warning(f"Loading vg_region data")
                list_data_dict.append(self.load_vg_region(data_path_i, image_folder_i))
            # elif 'grounded_llava_objects' in data_path_i:
            #     logging.warning(f"Loading grounded_llava_objects data")
            #     list_data_dict.append(self.load_grounded_llava_objects(data_path_i, image_folder_i))
            elif 'git_instruction' in data_path_i:
                logging.warning(f"Loading git_instruction data")
                if data_multiple is None:
                    logging.warning(f"Multiplying git_instruction by 3 times to make it around 100k")
                    list_data_dict.append(self.load_git_instruction(data_path_i, image_folder_i) * 3)
                else: 
                    list_data_dict.append(self.load_git_instruction(data_path_i, image_folder_i))
            elif 'llava_instruct' in data_path_i:
                logging.warning(f"Loading llava_instruct data")
                list_data_dict.append(self.load_llava(data_path_i, image_folder_i))
            elif 'grounded_llava_boxes' in data_path_i:
                logging.warning(f"Loading grounded_llava_boxes data")
                list_data_dict.append(self.load_grounded_llava_boxes(data_path_i, image_folder_i))
            elif 'refexp' in data_path_i:
                logging.warning(f"Loading refexp data")
                list_data_dict.append(self.load_refexp(data_path_i, image_folder_i))
            elif 'flickr' in data_path_i:
                logging.warning(f"Loading flickr data")
                list_data_dict.append(self.load_flickr(data_path_i, image_folder_i))
            elif 'objects365' in data_path_i:
                logging.warning(f"Loading o365 data")
                list_data_dict.append(self.load_objects365(data_path_i, image_folder_i))
            elif 'cc3m_595k' in data_path_i:
                logging.warning(f"Loading cc3m_595k data")
                list_data_dict.append(self.load_cc3m(data_path_i, image_folder_i))
            else:
                raise ValueError(f'{data_path_i} Not Supported.')
        
        if data_multiple is None:
            # Concat all data directly and Shuffle.
            list_data_dict = [item for dataset_i in list_data_dict for item in dataset_i]
            random.shuffle(list_data_dict)
        else:
            new_list_data_dict = []
            for data_scaler_i, dataset_i in zip(data_multiple, list_data_dict):
                dataset_name_i = dataset_i[0]['dataset']
                logging.warning(f"Multiplying {dataset_name_i} by {data_scaler_i} times")
                new_dataset_i = extend_list(dataset_i, data_scaler_i)
                new_list_data_dict.extend(new_dataset_i)
            list_data_dict = new_list_data_dict
            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.model_args = model_args
        self.point_input_sample = self.data_args.point_input_sample
        self.add_region_feature = self.model_args.add_region_feature
        self.no_coor = self.model_args.no_coor
        self.refer_previous_point = self.data_args.refer_previous_point

    def __len__(self):
        return len(self.list_data_dict)
    
    def get_obj_center(self, box, ratio_w, ratio_h, std_dev_weight=0.15):
        box_center_w = ratio_w * (box[0]+box[2])/2.0
        box_center_h = ratio_h * (box[1]+box[3])/2.0
        
        box_min_w = ratio_w * box[0]
        box_max_w = ratio_w * box[2]

        box_min_h = ratio_h * box[1]
        box_max_h = ratio_h * box[3]

        # Set std of gaussian sampling, 68%/95% is sampled within +- 1/2 std_dev.
        gau_std_w = (box_max_w - box_min_w)*std_dev_weight
        gau_std_h = (box_max_h - box_min_h)*std_dev_weight

        def sample_float_within_range(mean, std_dev, min_val, max_val):
            while True:
                x = random.gauss(mean[0], std_dev[0])
                y = random.gauss(mean[1], std_dev[1])
                if min_val[0] <= x <= max_val[0] and min_val[1] <= y <= max_val[1]:
                    return x, y
        
        jit_x, jit_y = sample_float_within_range(mean=[box_center_w, box_center_h], std_dev=[gau_std_w, gau_std_h], 
                                                 min_val=[box_min_w, box_min_h], max_val=[box_max_w, box_max_h])

        return jit_x, jit_y

    def sample_point_in_segment(self, mask, ratio_w, ratio_h, box=None, sampling='uniform'):
        mask['counts'] = mask['counts'].encode('ascii')
        bin_mask = mask_util.decode(mask)
        # Get the indices of True elements in the mask
        # Note here the size of bin_mask is [h, w].
        indices = np.transpose(np.nonzero(bin_mask))
        if sampling == 'center' or sampling == 'gaussian':
            if sampling == 'center':
                box_anchor_w = int((box[0]+box[2])/2.0)
                box_anchor_h = int((box[1]+box[3])/2.0)
            elif sampling == 'gaussian':
                # Sample a point based on centered gaussian distribution. ratio_w and ratio_h is set to 1 to keep original wh.
                box_anchor_w, box_anchor_h = self.get_obj_center(box, 1, 1, std_dev_weight=0.15)
            # get 1000 random items from the indices
            sampled_list = random.sample(list(range(len(indices))), min(1000, len(indices)))
            min_dis = 1e6
            min_point = None
            for sampled_i in sampled_list:
                point_i = indices[sampled_i]
                dis_i = (point_i[0] - box_anchor_h)**2 + (point_i[1] - box_anchor_w)**2
                if dis_i <= min_dis or min_point is None:
                    min_dis = dis_i
                    min_point = point_i
            point = min_point
        elif sampling == 'uniform':
            # Randomly select an index
            random_index = np.random.choice(len(indices))
            # Get the randomly selected point
            point = indices[random_index]
        else:
            raise NotImplementedError(f'Not support {sampling}.')
        # Note here point is in original image size and its order is [h, w].
        cor_x = point[1] * ratio_w
        cor_y = point[0] * ratio_h
        return cor_x, cor_y


    def get_bbox_coor(self, box, ratio_w, ratio_h):
        return box[0] * ratio_w, box[1] * ratio_h, box[2] * ratio_w, box[3] * ratio_h


    def generate_mask_for_feature(self, coor, box, mask, raw_w, raw_h):
        # Build SAM mask
        if mask is not None:
            mask['counts'] = mask['counts'].encode('ascii')
            sam_mask = mask_util.decode(mask)
            # Note [h, w] -> [w, h].
            sam_mask = np.transpose(sam_mask)
        else:
            sam_mask = None
        # Build box mask
        box_mask = np.zeros((raw_w, raw_h))
        box_mask[box[0]:box[2]+1, box[1]:box[3]+1] = 1
        
        coor_mask = np.zeros((raw_w, raw_h))
        # Assume it samples a point.
        if len(coor) == 2:
            # Define window size
            span = 5
            # Make sure the window does not exceed array bounds
            x_min = max(0, coor[0] - span)
            x_max = min(raw_w, coor[0] + span + 1)
            y_min = max(0, coor[1] - span)
            y_max = min(raw_h, coor[1] + span + 1)
            coor_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
            # SAM mask might be out of bounding box, so don't use sam_mask * box_mask. 
            coor_mask = coor_mask * sam_mask if sam_mask is not None else coor_mask * box_mask
            assert (coor_mask==1).any(), f"coor: {coor}, box: {box}, raw_w: {raw_w}, raw_h: {raw_h}"
        elif len(coor) == 4:
            coor_mask = box_mask * sam_mask if sam_mask is not None else box_mask
            if (coor_mask==0).all():
                print('Find one sample sam mask and box has no overlap, use box mask only')
                coor_mask = box_mask
            assert (coor_mask==1).any(), f"coor: {coor}, box: {box}, raw_w: {raw_w}, raw_h: {raw_h}"
        else:
            raise NotImplementedError('Coordinates must be 2d or 4d.')
        coor_mask = torch.from_numpy(coor_mask)
        assert len(coor_mask.nonzero()) != 0

        return coor_mask


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = deepcopy(self.list_data_dict[i])
        # sources = self.list_data_dict[i]
        cache_region_masks = []
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = Image.open(image_file).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == 'square_nocrop':
                resized_image_h = self.data_args.resized_image_h
                resized_image_w = self.data_args.resized_image_w
                image = processor.preprocess(image, return_tensors='pt', do_resize=True, do_center_crop=False, size=[resized_image_h, resized_image_w])['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            # Process Locations/Coordinations.
            if 'location_instruction' in sources[0]:
                assert sources[0]['dataset'] in ['vg_object', 'vg_yesno_object', 'vg_attribute', 'vg_relation', 'vg_region', \
                                                 'git_instruction', 'grounded_llava_boxes', 'refexp', 'flickr',\
                                                    'objects365']
                ratio_w = VOCAB_IMAGE_W * 1.0 / sources[0]['image_w']
                ratio_h = VOCAB_IMAGE_H * 1.0 / sources[0]['image_h']
                conversation = deepcopy(sources[0]['conversations'])
                assert len(sources) == 1
                for box_list_idx, box_list_i in enumerate(sources[0]['box_x1y1x2y2']):
                    # For human input, always build a cache to save sampled point in this round of human input.
                    if box_list_idx % 2 == 0 and self.refer_previous_point:
                        point_box_cache = {}

                    # Replace location placeholders with points or boxes.
                    if len(box_list_i) == 0:
                        # No location mentioned in this round of conversation.
                        continue

                    if box_list_idx % 2 == 0:
                        # Randomly choose point or box as coordination format in human round.
                        location_format = random.choice(['point', 'box'])
                    else:
                        # Always output box in model reponse.
                        location_format = 'box'
                    cur_conv = conversation[box_list_idx]['value']
                    # Iteratively replace <bbox_location> in current conv with real box/point coordinate.
                    for box_idx, box_i in enumerate(box_list_i):
                        box_i = regulate_box(box_i, sources[0]['image_w'], sources[0]['image_h'])
                        if location_format == 'box':
                            # If this box is mentioned in last human input, use the same coordinates as human mentioned.
                            if 'point_box_cache' in locals() and tuple(box_i) in point_box_cache:
                                raw_coor_i = point_box_cache[tuple(box_i)]
                                coor_i = f'[{int(raw_coor_i[0])}, {int(raw_coor_i[1])}]'
                            else:
                                raw_coor_i = self.get_bbox_coor(box=box_i, ratio_w=ratio_w, ratio_h=ratio_h)
                                coor_i = f'[{int(raw_coor_i[0])}, {int(raw_coor_i[1])}, {int(raw_coor_i[2])}, {int(raw_coor_i[3])}]'
                        elif location_format == 'point':
                            # Assert it's human input.
                            assert box_list_idx % 2 == 0
                            # If this box is mentioned previously in this round of human input, use the same coordinates as previously mentioned.
                            if 'point_box_cache' in locals() and tuple(box_i) in point_box_cache:
                                raw_coor_i = point_box_cache[tuple(box_i)]
                            else:
                                if 'segment_mask' in self.point_input_sample:
                                    if 'masks' in sources[0]:
                                        cur_mask = deepcopy(sources[0]['masks'][box_list_idx][box_idx])
                                        assert cur_mask['size'][0] == sources[0]['image_h']
                                        assert cur_mask['size'][1] == sources[0]['image_w']
                                        if 'uniform' in self.point_input_sample.split('|')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h)
                                        elif 'center' in self.point_input_sample.split('|')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h, box=box_i, sampling='center')
                                        elif 'gaussian' in self.point_input_sample.split('|')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h, box=box_i, sampling='gaussian')
                                    else:
                                        # Not all data have/need segment masks.
                                        obj_center_x, obj_center_y = self.get_obj_center(box=box_i, ratio_w=ratio_w, ratio_h=ratio_h, std_dev_weight=0.15)
                                elif self.point_input_sample == 'gaussian':
                                    obj_center_x, obj_center_y = self.get_obj_center(box=box_i, ratio_w=ratio_w, ratio_h=ratio_h, std_dev_weight=0.15)
                                elif self.point_input_sample == 'center':
                                    obj_center_x = ratio_w * (box_i[0]+box_i[2])/2.0
                                    obj_center_y = ratio_h * (box_i[1]+box_i[3])/2.0
                                else:
                                    raise NotImplementedError(f'Not support {self.point_input_sample} in data sampling')
                                raw_coor_i = [obj_center_x, obj_center_y]
                                if 'point_box_cache' in locals() and self.refer_previous_point:
                                    point_box_cache[tuple(box_i)] = raw_coor_i
                            coor_i = f'[{int(raw_coor_i[0])}, {int(raw_coor_i[1])}]'
                        assert f'<bbox_location{box_idx}>' in cur_conv
                        if self.add_region_feature and box_list_idx % 2 == 0:
                            if self.no_coor:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', f'{DEFAULT_REGION_FEA_TOKEN}')
                            else:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', coor_i + f' {DEFAULT_REGION_FEA_TOKEN}')
                            cur_box = box_i
                            cur_mask = deepcopy(sources[0]['masks'][box_list_idx][box_idx]) if 'masks' in sources[0] else None
                            ori_size_raw_coor_i = [raw_coor_i[0]/ratio_w, raw_coor_i[1]/ratio_h, raw_coor_i[2]/ratio_w, raw_coor_i[3]/ratio_h] if len(raw_coor_i) == 4 \
                                else [raw_coor_i[0]/ratio_w, raw_coor_i[1]/ratio_h]
                            cur_region_mask = self.generate_mask_for_feature(ori_size_raw_coor_i, cur_box, cur_mask, raw_w=sources[0]['image_w'], raw_h=sources[0]['image_h'])
                            cache_region_masks.append(cur_region_mask)
                            # print('cur_conv:', cur_conv)
                            # print('cur_region_mask:', cur_region_mask.nonzero())
                            # raise NotImplementedError()
                            # pdb.set_trace()
                        else:
                            if self.no_coor:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', '')
                            else:
                                cur_conv = cur_conv.replace(f'<bbox_location{box_idx}>', coor_i)
                    # Assign this round of conv back.
                    conversation[box_list_idx]['value'] = cur_conv
                sources[0]['conversations'] = conversation
                # print(conversation)
                # exit(0)

            # cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)   # FIXME: 14 is hardcoded patch size
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            # add_region_feature=self.add_region_feature
            )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if self.add_region_feature:
            data_dict['region_masks'] = cache_region_masks
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'region_masks' in instances[0]:
            region_masks = [instance['region_masks'] for instance in instances]
            batch['region_masks'] = region_masks

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                model_args=model_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.no_coor:
        assert model_args.add_region_feature
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        model = FERRETLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing and not training_args.fsdp:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["ferret_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
            add_region_feature=model_args.add_region_feature,
            region_geo_sampler=model_args.region_geo_sampler,
            sampler_pooler_mode=model_args.sampler_pooler_mode
        )
        
        vision_tower = model.get_vision_tower()

        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16

        vision_tower.to(dtype=dtype, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        training_args.save_vision_tower = model_args.save_vision_tower

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if model_args.add_region_feature:
            if model_args.region_geo_sampler:
                for p in model.get_model().region_geo_sampler.parameters():
                    p.requires_grad = True
            else:
                for p in model.get_model().region_fea_adapter.parameters():
                    p.requires_grad = True

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer, add_region_feature=model_args.add_region_feature)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # pdb.set_trace()
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    params_has_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'Params being frozen: {params_no_grad}.')
    print(f'Params being updated: {params_has_grad}.')
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              model_args=model_args)
    trainer = FERRETTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    # pdb.set_trace()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if not training_args.fsdp:
        model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,
                                       save_vision_tower=model_args.save_vision_tower)


if __name__ == "__main__":
    train()
