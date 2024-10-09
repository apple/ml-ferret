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
import copy
import json
import logging
import os
import re
import pathlib
import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
from pycocotools import mask as mask_util
from functools import partial
from copy import deepcopy

import tokenizers
import torch
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset

from ferretui import conversation as conversation_lib
from ferretui.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                IMAGE_TOKEN_INDEX, DEFAULT_REGION_FEA_TOKEN,
                                VOCAB_IMAGE_H, VOCAB_IMAGE_W, GROUNDING_TEMPLATES)
from ferretui.mm_utils import process_anyres_image, tokenizer_image_token
from ferretui.model import *
from ferretui.train.ferret_trainer import FerretTrainer

from PIL import Image

import torch.distributed as dist

# local_rank = None
local_rank = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
global_rank = int(os.getenv('RANK', '0')) 
world_size = int(os.environ["WORLD_SIZE"])

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(
    tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    add_region_feature: bool = False
    region_geo_sampler: bool = False
    sampler_pooler_mode: str = field(default='mean')    # Support 'mean' or 'max'
    no_coor: bool = False


@dataclass
class DataArguments:
    data_path: List[str] = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_multiple: List[float] = field(default=None,
                           metadata={"help": "Data mutliplier for each dataset when mixed. None means direct concat."}) 
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: List[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    resized_image_h: int = 336  #  224
    resized_image_w: int = 336  #  224
    point_input_sample: str = 'segment_mask-uniform'  # 'segment_mask-uniform', 'segment_mask-center', 'segment_mask-gaussian', 'gaussian', 'center'
    refer_previous_point: bool = False
    use_shard_datasets: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
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
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
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
    lora_qv_proj_only: bool = False
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    use_safetensors: Optional[bool] = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
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
        to_return = {k: t for k,
                     t in named_params if "lora_" in k or "bias" in k}
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
    to_return = {k: maybe_zero_3(v, ignore_status=True)
                 for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, qv_proj_only=False):
    if qv_proj_only:
        rank0_print('Only add LoRA to QV proj')
        return ['q_proj', 'v_proj']
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(
                    parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

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
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token)

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
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

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


def preprocess_llama3(
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
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i > 0:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

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
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
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
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

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


def preprocess_gemma(
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
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask targets
    sep = "<start_of_turn>" + conv.sep + conv.roles[1] + "\n"
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
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1 # exclude <bos>
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1 # exclude <bos>

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


def preprocess_phi3(
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
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

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


def preprocess_mpt(
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
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(
                rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

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
        conversation = source[0]['value'] + source[1]['value'] + \
            conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(
            source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
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
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("gemma"):
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_phi3(sources, tokenizer, has_image=has_image)

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
        input_ids = [tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

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

    def shard_data(self, datas, data_path, ori_counts):
        no_shard_per_worker = int(math.ceil(len(datas) / self.world_size))
        datas = datas[no_shard_per_worker*self.global_rank: no_shard_per_worker*(self.global_rank+1)]
        print(f"Shard {data_path}: ori: {ori_counts}, now: {len(datas)}")
        return datas

    def load_pretrain(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "pretrain"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_llava_mixed(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'llava_v1_5_mixed'
            # may contain sharegpt data
            if "image" in data_i:
                data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas

    def load_git_instruction(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'git_instruction'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas
    
    def load_vg_element(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'vg_element'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
            data_i['location_instruction'] = True
        return datas

    def load_llava_grounded(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "llava_grounded"
            data_i["image"] = os.path.join(image_folder, data_i["image"])
            data_i["location_instruction"] = True
        return datas
    
    def load_flickr(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "flickr_entities"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas
    
    def load_refexp(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "refexp"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas

    def load_obj365(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "objects365"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas
    
    def load_sharegpt4v(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'sharegpt4v'
            if "image" in data_i:
                data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_lvisinstruct4v(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i['dataset'] = 'lvisinstruct4v'
            if "image" in data_i:
                data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas
    
    # plain multimodal data without bbox from LLaVA v1.5
    def load_vqa(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "vqa"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas 

    def load_swit(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "vqa"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_sharegpt(self, data_path):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "sharegpt"
        return datas 
    
    def load_screen2words(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = 'screen2words'
            data_i['image'] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_widgetcaptions(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "widgetcaptions"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_taperception(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "taperception"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_widget_listing(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "widget_listing"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_ocr(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "ocr"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_find_text(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "find_text"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_icon_recognition(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "icon_recognition"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_find_icons(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "find_icons"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_widget_classification(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "widget_classification"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_find_widget(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "find_widget"
            data_i["location_instruction"] = True
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_detailed_description(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "detailed_description"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_conversation_perception(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "conversation_perception"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas
    
    def load_conversation_interaction(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "conversation_interaction"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
            data_i["location_instruction"] = True
        return datas

    def load_function(self, data_path, image_folder):
        datas = json.load(open(data_path, "r"))
        ori_counts = len(datas)
        if self.data_args.use_shard_datasets:
            datas = self.shard_data(datas, data_path, ori_counts)
        for data_i in datas:
            data_i["dataset"] = "function"
            data_i["image"] = os.path.join(image_folder, data_i['image'])
        return datas

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: DataArguments,
                 ):
        super(LazySupervisedDataset, self).__init__()
        data_multiple = data_args.data_multiple
        if not isinstance(data_path, list):
            data_path = [data_path]

        image_folders = data_args.image_folder
        if not isinstance(data_args.image_folder, list):
            image_folders = [data_args.image_folder]

        self.data_args = data_args

        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.global_rank = int(os.getenv('RANK', '0')) 

        rank0_print(f"world size: {self.world_size}")
        print(f"global rank: {self.global_rank}")

        list_data_dict = []
        for data_path_i, image_folder_i in zip(data_path, image_folders):
            if 'blip_laion_cc_sbu' in data_path_i:
                rank0_print(f"Loading Pretrain Blip_Laion_CC_SBU data")
                list_data_dict.append(self.load_pretrain(data_path_i, image_folder_i))
            elif 'minigemini' in data_path_i:
                rank0_print(f"Loading Pretrain ALLaVA data")
                list_data_dict.append(self.load_pretrain(data_path_i, image_folder_i))
            elif 'llava_v1_5_mix' in data_path_i:
                rank0_print(f"Loading LLaVA v1.5 mixed data")
                list_data_dict.append(self.load_llava_mixed(data_path_i, image_folder_i))
            elif 'svit_v1_mix' in data_path_i:
                rank0_print(f"Loading SVIT v1 mixed data")
                list_data_dict.append(self.load_llava_mixed(data_path_i, image_folder_i))
            elif 'git_instruction' in data_path_i:
                rank0_print(f"Loading GIT instruct data")
                if data_multiple is None:
                    rank0_print(f"Multiplying GIT instruct by 3 times to make it around 100k")
                    list_data_dict.append(self.load_git_instruction(data_path_i, image_folder_i) * 3)
                else:
                    list_data_dict.append(self.load_git_instruction(data_path_i, image_folder_i))
            elif 'vg_objects' in data_path_i:
                rank0_print(f"Loading VG objects data")
                list_data_dict.append(self.load_vg_element(data_path_i, image_folder_i))
            elif 'vg_relations' in data_path_i:
                rank0_print(f"Loading VG relations data")
                list_data_dict.append(self.load_vg_element(data_path_i, image_folder_i))
            elif 'vg_regions' in data_path_i:
                rank0_print(f"Loading VG regions data")
                list_data_dict.append(self.load_vg_element(data_path_i, image_folder_i))
            elif 'grounded_llava_box' in data_path_i:
                rank0_print(f"Loading LLaVA grounded data")
                list_data_dict.append(self.load_llava_grounded(data_path_i, image_folder_i))
            elif 'flickr' in data_path_i:
                rank0_print(f"Loading Flickr30k entities data")
                list_data_dict.append(self.load_flickr(data_path_i, image_folder_i))
            elif 'refexp' in data_path_i:
                rank0_print(f"Loading Ref expression data")
                list_data_dict.append(self.load_refexp(data_path_i, image_folder_i))
            elif 'objects365' in data_path_i:
                rank0_print(f"Loading O365 data")
                list_data_dict.append(self.load_obj365(data_path_i, image_folder_i))
            elif 'sharegpt4v' in data_path_i.lower():
                rank0_print(f"Loading sharegpt4v data")
                list_data_dict.append(self.load_sharegpt4v(data_path_i, image_folder_i))
            elif 'lvis-instruct4v' in data_path_i.lower():
                rank0_print(f"Loading lvisinstruct4v data")
                list_data_dict.append(self.load_lvisinstruct4v(data_path_i, image_folder_i))

            elif 'okvqa' in data_path_i:
                rank0_print(f"Loading COCO OKVQA data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'vqav2' in data_path_i:
                rank0_print(f"Loading COCO VQAv2 data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'ocr_vqa' in data_path_i:
                rank0_print(f"Loading OCRVQA data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'textvqa_textcaps' in data_path_i:
                rank0_print(f"Loading TextVQA TextCaps data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'gqa_vqa' in data_path_i:
                rank0_print(f"Loading GQA data")
                list_data_dict.append(self.load_vqa(data_path_i, image_folder_i))
            elif 'svit_v1' in data_path_i:
                rank0_print(f"Loading SWIT complex data")
                list_data_dict.append(self.load_swit(data_path_i, image_folder_i))
            elif 'sharegpt' in data_path_i or 'wizardlm' in data_path_i:
                rank0_print(f"Loading ShareGPT/WizardLM text only data")
                list_data_dict.append(self.load_sharegpt(data_path_i))
            elif 'screen2words' in data_path_i:
                logging.warning(f"Loading screen2words data")
                list_data_dict.append(self.load_screen2words(data_path_i, image_folder_i))
            elif 'widgetcaptions' in data_path_i:
                logging.warning(f"Loading widgetcaptions data")
                list_data_dict.append(self.load_widgetcaptions(data_path_i, image_folder_i))
            elif 'taperception' in data_path_i:
                logging.warning(f"Loading taperception data")
                list_data_dict.append(self.load_taperception(data_path_i, image_folder_i))
            elif 'widget_listing' in data_path_i:
                logging.warning(f"Loading widget_listing data")
                list_data_dict.append(self.load_widget_listing(data_path_i, image_folder_i))
            elif 'ocr' in data_path_i:
                logging.warning(f"Loading ocr data")
                list_data_dict.append(self.load_ocr(data_path_i, image_folder_i))
            elif 'find_text' in data_path_i:
                logging.warning(f"Loading find_text data")
                list_data_dict.append(self.load_find_text(data_path_i, image_folder_i))
            elif 'icon_recognition' in data_path_i:
                logging.warning(f"Loading icon_recognition data")
                list_data_dict.append(self.load_icon_recognition(data_path_i, image_folder_i))
            elif 'find_icons' in data_path_i:
                logging.warning(f"Loading find_icons data")
                list_data_dict.append(self.load_find_icons(data_path_i, image_folder_i))
            elif 'widget_classification' in data_path_i:
                logging.warning(f"Loading widget_classification data")
                list_data_dict.append(self.load_widget_classification(data_path_i, image_folder_i))
            elif 'find_widget' in data_path_i:
                logging.warning(f"Loading find_widget data")
                list_data_dict.append(self.load_find_widget(data_path_i, image_folder_i))
            elif 'detailed_description' in data_path_i:
                logging.warning(f"Loading detailed_description data")
                list_data_dict.append(self.load_detailed_description(data_path_i, image_folder_i))
            elif 'conversation_perception' in data_path_i:
                logging.warning(f"Loading conversation_perception data")
                list_data_dict.append(self.load_conversation_perception(data_path_i, image_folder_i))
            elif 'conversation_interaction' in data_path_i:
                logging.warning(f"Loading conversation_interaction data")
                list_data_dict.append(self.load_conversation_interaction(data_path_i, image_folder_i))
            elif 'function' in data_path_i:
                logging.warning(f"Loading function data")
                list_data_dict.append(self.load_function(data_path_i, image_folder_i))
            else:
                rank0_print("Loading {} not supported".format(data_path_i))
        
        if data_multiple is None:
            # Concat all data directly and shuffle.
            list_data_dict = [item for dataset_i in list_data_dict for item in dataset_i]
            random.shuffle(list_data_dict)
        else:
            new_list_data_dict = []
            for data_scaler_i, dataset_i in zip(data_multiple, list_data_dict):
                dataset_name_i = dataset_i[0]['dataset']
                rank0_print(f"Multiplying {dataset_name_i} by {data_scaler_i} times")
                new_dataset_i = extend_list(dataset_i, data_scaler_i)
                new_list_data_dict.extend(new_dataset_i)
            list_data_dict = new_list_data_dict
            random.shuffle(list_data_dict)

        print(f"R{self.global_rank} number of samples: {len(list_data_dict)}")
        rank0_print(f"The total training set contains {len(list_data_dict)} samples.")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.model_args = model_args
        self.point_input_sample = self.data_args.point_input_sample
        self.add_region_feature = self.model_args.add_region_feature
        self.no_coor = self.model_args.no_coor
        self.refer_previous_point = self.data_args.refer_previous_point

        if self.data_args.use_shard_datasets:
            self.sync_iter_counts()

    def __len__(self):
        return len(self.list_data_dict)

    def sync_iter_counts(self):
        # Sync the total sample counts on each worker
        # Calculate the number of samples for this worker
        num_samples = len(self)
        rank0_print(f"sync iter counts num_samples: {num_samples}")
        # Gather the number of samples from all workers
        min_num_samples_tensor = torch.tensor(num_samples, dtype=torch.int64).cuda()
        dist.all_reduce(min_num_samples_tensor, op=dist.ReduceOp.MIN)
        min_num_samples = min_num_samples_tensor.item()
        print(f"min_num_sample: {min_num_samples}")
        # Create a subset of the dataset based on min_num_samples
        # indices = list(range(num_samples))
        # np.random.shuffle(indices)
        # subset_indices = indices[:min_num_samples]
        self.list_data_dict = self.list_data_dict[:min_num_samples]
        print(f"[R{self.global_rank}] sync_iter_counts at {os.path.basename(__file__)}: ori dataset counts: {num_samples}, now mmx list dataset len: {len(self.list_data_dict)}.")

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
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
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
                rank0_print('Find one sample sam mask and box has no overlap, use box mask only')
                coor_mask = box_mask
            assert (coor_mask==1).any(), f"coor: {coor}, box: {box}, raw_w: {raw_w}, raw_h: {raw_h}"
        else:
            raise NotImplementedError('Coordinates must be 2d or 4d.')
        coor_mask = torch.from_numpy(coor_mask)
        assert len(coor_mask.nonzero()) != 0

        return coor_mask

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    @staticmethod
    def format_unicode_filenames(filename):
        # replace unicode characters
        # i.e. wikiart/images/arnold-b#U00e3#U00b6cklin_soldiers-amount-towards-a-mountain-fortress.jpg
        #   -> wikiart/images/arnold-bã¶cklin_soldiers-amount-towards-a-mountain-fortress.jpg
            return re.subn(r"(#U[0-9a-f]{4})", lambda cp: chr(int(cp.groups()[0][2:], 16)), filename)[0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = deepcopy(self.list_data_dict[i])
        cache_region_masks = []
        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if not os.path.isfile(image_file):
                image_file = self.format_unicode_filenames(image_file)
                possible_exts = ['.gif', '.jpg', '.jpeg', '.png']
                for ext_ in possible_exts:
                    filename_ = os.path.splitext(image_file)[0]
                    if os.path.isfile(filename_ + ext_):
                        image_file = filename_ + ext_
                        break
            image = Image.open(image_file).convert('RGB')
            image_size = image.size
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255)
                                      for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')[ #(640, 480)
                    'pixel_values'][0]
            elif self.data_args.image_aspect_ratio == 'square_nocrop':
                resized_image_h = self.data_args.resized_image_h
                resized_image_w = self.data_args.resized_image_w
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt', do_resize=True, do_center_crop=False, \
                                             size=[resized_image_h, resized_image_w])['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == 'anyres':
                resized_image_h = self.data_args.resized_image_h
                resized_image_w = self.data_args.resized_image_w
                image_size = image.size
                image_process_func = partial(processor.preprocess, return_tensors='pt', do_resize=True, do_center_crop=False, \
                                             size=[resized_image_h, resized_image_w])
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, \
                                             image_process_func=image_process_func) # torch.Size([5, 3, 336, 336])
                # image_size = image.size
                # image = process_anyres_image(       # torch.Size([5, 3, 336, 336])
                #     image, processor, self.data_args.image_grid_pinpoints)
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
            
            # Process Locations/Coordinations.
            if 'location_instruction' in sources[0]:
                assert sources[0]['dataset'] in ['git_instruction', 'vg_element', 'llava_grounded',\
                                                 'refexp', 'flickr_entities', 'objects365', 'widgetcaptions', 'taperception', \
                                                 'widget_listing', 'ocr', 'find_text', 'icon_recognition', 'find_icons', \
                                                 'widget_classification',  'find_widget', 'conversation_interaction']
                ratio_w = VOCAB_IMAGE_W * 1.0 / sources[0]['image_w']
                ratio_h = VOCAB_IMAGE_H * 1.0 / sources[0]['image_h']
                conversation = deepcopy(sources[0]['conversations'])
                assert len(sources) == 1

                # add GROUNDING_PROMPT to grounding dataset at its first human round
                if sources[0]['dataset'] in ['llava_grounded', 'refexp', 'flickr_entities', 'objects365']:
                    # conversation[0]['value'] = conversation[0]['value'] + random.choice(GROUNDING_TEMPLATES)
                    conversation[0]['value'] = conversation[0]['value'] + '\nProvide the bounding boxes of the mentioned objects.'

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
                                        if 'uniform' in self.point_input_sample.split('-')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h)
                                        elif 'center' in self.point_input_sample.split('-')[1]:
                                            obj_center_x, obj_center_y = self.sample_point_in_segment(mask=cur_mask, ratio_w=ratio_w, ratio_h=ratio_h, box=box_i, sampling='center')
                                        elif 'gaussian' in self.point_input_sample.split('-')[1]:
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
                        assert f'<bbox_location{box_idx}>' in cur_conv, f"String '<bbox_location{box_idx}>' not found in {cur_conv}"
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
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
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
            image_sizes = [instance['image_size'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes
        
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


def unfreeze_vit(vision_tower):
    for _, p in vision_tower.named_parameters():
        p.requires_grad = True


def format_bytes(size):
    billion = 10**9
    million = 10**6

    if size >= billion:
        return f"{size / billion:.2f}B"
    elif size >= million:
        return f"{size / million:.2f}M"
    else:
        return f"{size} bytes"


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.resized_image_w = data_args.resized_image_w
    model_args.resized_image_h = data_args.resized_image_h

    if model_args.no_coor:
        assert model_args.add_region_feature

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))

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
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    
    model_max_length_args = {}
    if 'llava-v1.6-8b' not in model_args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True)
        if config.max_position_embeddings < training_args.model_max_length:
            rank0_print(
                f'Set the max_position_embeddings from {config.max_position_embeddings} to {training_args.model_max_length}')
            model_max_length_args.update(
                {'max_position_embeddings': training_args.model_max_length})
            
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = FerretMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,

                **bnb_model_from_pretrained_args
            )
        elif "gemma" in model_args.model_name_or_path:
            model = FerretGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                use_safetensors=training_args.use_safetensors,
                **bnb_model_from_pretrained_args
            )
        elif "phi3" in model_args.model_name_or_path:
            model = LlavaPhiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = FerretLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                use_safetensors=training_args.use_safetensors,
                **bnb_model_from_pretrained_args,
                **model_max_length_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
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
            target_modules=find_all_linear_names(model, training_args.lora_qv_proj_only),
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

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
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
    elif "gemma" in model_args.version:
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["gemma"]
    else:
        if tokenizer.pad_token is None:
            rank0_print("Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
            add_region_feature=model_args.add_region_feature,
            region_geo_sampler=model_args.region_geo_sampler,
            sampler_pooler_mode=model_args.sampler_pooler_mode,
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids]
        
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        if training_args.unfreeze_mm_vision_tower:
            lr_of_vit = training_args.mm_vision_tower_lr if training_args.mm_vision_tower_lr is not None else training_args.learning_rate
            lr_of_mlp = training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate
            assert lr_of_vit > 0.0 and lr_of_mlp > 0.0
            training_args.mm_projector_lr = lr_of_mlp
            unfreeze_vit(vision_tower)
            rank0_print(
                f'Tune the entire model! The LR of ViT is {lr_of_vit}. The LR of MLP is {lr_of_mlp}. The LR of LLM is {training_args.learning_rate}')

        if model_args.add_region_feature:
            if model_args.region_geo_sampler:
                for p in model.get_model().region_geo_sampler.parameters():
                    p.requires_grad = True
            else:
                for p in model.get_model().region_fea_adapter.parameters():
                    p.requires_grad = True

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in model.get_model().parameters() if p.requires_grad)

        rank0_print(f"Total parameters: {format_bytes(total_params)}")
        rank0_print(f"Trainable parameters: {format_bytes(trainable_params)}")

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.mm_patch_merge_type = model_args.mm_patch_merge_type
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer, add_region_feature=model_args.add_region_feature)
        model.config.pad_token_id = tokenizer.pad_token_id

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

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              model_args=model_args)
    trainer = FerretTrainer(model=model,
                            tokenizer=tokenizer,
                            args=training_args,
                            **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

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
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

def init_distributed_mode():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank
    )
    print(f"dist.is_initialized(): {dist.is_initialized()}")

if __name__ == "__main__":
    init_distributed_mode()
    train()
