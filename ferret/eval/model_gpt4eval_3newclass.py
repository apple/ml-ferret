"""
Usage:

Example:
CUDA_VISIBLE_DEVICES=0 python -m ferret.eval.model_gpt4eval_3newclass \
    --model-path checkpoints/ferret_13b \
    --data_path ferret/eval/ferret_gpt4_data/refer_desc/question.jsonl \
    --answers-file gpt4_result/refer_desc/ferret_ft_clipL336_vicunaV1-3-13b_3Ep  --add_region_feature

"""

import argparse
import torch
import os
import json
from tqdm import tqdm

# Added
from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ferret.model.builder import load_pretrained_model
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ferret.conversation import conv_templates, SeparatorStyle
from ferret.utils import disable_torch_init
from PIL import Image
import math
import pdb
import numpy as np
from copy import deepcopy
from functools import partial
import re

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def generate_mask_for_feature(coor, raw_w, raw_h, mask=None):
    if mask is not None:
        assert mask.shape[0] == raw_w and mask.shape[1] == raw_h
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
        assert (coor_mask==1).any(), f"coor: {coor}, raw_w: {raw_w}, raw_h: {raw_h}"
    elif len(coor) == 4:
        # Box input or Sketch input.
        coor_mask[coor[0]:coor[2]+1, coor[1]:coor[3]+1] = 1
        if mask is not None:
            coor_mask = coor_mask * mask
    coor_mask = torch.from_numpy(coor_mask)
    try:
        assert len(coor_mask.nonzero()) != 0
    except:
        pdb.set_trace()
    return coor_mask
    
class GPTEval_Data():
    def __init__(self, data_path, image_path, args) -> None:
        datas = [json.loads(q) for q in open(os.path.expanduser(args.data_path), "r")]

        for idx, i in enumerate(datas):
            i['image_path'] = os.path.join(image_path, i['image'].split('/')[-1])
            img_raw_w, img_raw_h = Image.open(i['image_path']).size
            pattern = r'\[.*?\]'
            matches = re.findall(pattern, i['text'])
            question = i['text']
            masks = []
            for match in matches:
                coor_cur = match.replace('[', '')
                coor_cur = coor_cur.replace(']', '')
                coor_cur = coor_cur.split(',')
                coor_cur = [float(i.strip()) for i in coor_cur]
                raw_box_coor = [int(coor_cur[0] * img_raw_w), int(coor_cur[1] * img_raw_h), int(coor_cur[2] * img_raw_w), int(coor_cur[3] * img_raw_h)]
                converted_box_coor = [int(coor_cur[0] * VOCAB_IMAGE_W), int(coor_cur[1] * VOCAB_IMAGE_H), int(coor_cur[2] * VOCAB_IMAGE_W), int(coor_cur[3] * VOCAB_IMAGE_H)]
                if args.add_region_feature:
                    question = question.replace(match, f'[{converted_box_coor[0]}, {converted_box_coor[1]}, {converted_box_coor[2]}, {converted_box_coor[3]}] {DEFAULT_REGION_FEA_TOKEN}')
                    generated_mask = generate_mask_for_feature(raw_box_coor, raw_w=img_raw_w, raw_h=img_raw_h, mask=None)
                    masks.append(generated_mask)
                else:
                    question = question.replace(match, f'[{converted_box_coor[0]}, {converted_box_coor[1]}, {converted_box_coor[2]}, {converted_box_coor[3]}]')

            # pdb.set_trace()
            if args.add_region_feature:
                i['region_masks'] = masks
            else:
                i['region_masks'] = None
            i['question'] = question
        # obj_list = [json.loads(line) for line in tqdm(open(data_path))]
        # question_prompt = "Is the object in <location> of the image a <obj1> or a <obj2>?"
        
        # for idx, i in enumerate(obj_list):
        #     i['image_path'] = os.path.join(image_path, i['image_path'].split('/')[-1])
        #     ratio_w = VOCAB_IMAGE_W * 1.0 / i['width']
        #     ratio_h = VOCAB_IMAGE_H * 1.0 / i['height']

        #     point_x_textvocab = int(i['sample_point'][0]*ratio_w)
        #     point_y_textvocab = int(i['sample_point'][1]*ratio_h)

        #     box_x1 = int(i['bbox_norm'][0]*i['width'])
        #     box_y1 = int(i['bbox_norm'][1]*i['height'])
        #     box_x2 = int(i['bbox_norm'][2]*i['width'])
        #     box_y2 = int(i['bbox_norm'][3]*i['height'])

        #     box_x1_textvocab = int(i['bbox_norm'][0]*VOCAB_IMAGE_W)
        #     box_y1_textvocab = int(i['bbox_norm'][1]*VOCAB_IMAGE_H)
        #     box_x2_textvocab = int(i['bbox_norm'][2]*VOCAB_IMAGE_W)
        #     box_y2_textvocab = int(i['bbox_norm'][3]*VOCAB_IMAGE_H)

        #     if args.region_format == 'point':
        #         region_coordinate_raw = [i['sample_point'][0], i['sample_point'][1]]
        #         i['question'] = question_prompt.replace('<location>', '[{}, {}]'.format(point_x_textvocab, point_y_textvocab))
        #         segment_mask = None
        #     elif args.region_format == 'box' or args.region_format == 'segment':
        #         region_coordinate_raw = [box_x1, box_y1, box_x2, box_y2]
        #         i['question'] = question_prompt.replace('<location>', '[{}, {}, {}, {}]'.format(box_x1_textvocab, box_y1_textvocab, box_x2_textvocab, box_y2_textvocab))
        #         if args.region_format == 'segment':
        #             segment_mask = np.array(i['segment_mask'])
        #         else:
        #             segment_mask = None
        #     else:
        #         raise NotImplementedError(f'{args.region_format} is not supported.')

        #     if args.add_region_feature:
        #         i['question'] = i['question'].replace('of the image', f'{DEFAULT_REGION_FEA_TOKEN} of the image')
        #         generated_mask = generate_mask_for_feature(region_coordinate_raw, raw_w=i['width'], raw_h=i['height'], mask=segment_mask)
        #         i['region_masks'] = [generated_mask]
        #     else:
        #         i['region_masks'] = None

        #     if idx % 2 == 0:
        #         i['question'] = i['question'].replace('<obj1>', i['name'])
        #         i['question'] = i['question'].replace('<obj2>', i['neg_class'])
        #     else:
        #         i['question'] = i['question'].replace('<obj2>', i['name'])
        #         i['question'] = i['question'].replace('<obj1>', i['neg_class'])

        self.datas = datas
        self._ids = range(len(self.datas))
        # pdb.set_trace()
    
    @property
    def ids(self):
        return deepcopy(self._ids)
    
    def fetch_data(self, id):
        ann = self.datas[id]
        return ann


def eval_model(args):
    # Data
    dataset = GPTEval_Data(data_path=args.data_path, image_path=args.image_path, args=args)
    data_ids = dataset.ids

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # chunk_data_ids = get_chunk(data_ids, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(answers_file, exist_ok=True)
    # answers_file = os.path.join(answers_file, f'{args.chunk_idx}_of_{args.num_chunks}.jsonl')
    answers_file = os.path.join(answers_file, f'ferret_answer.jsonl')
    ans_file = open(answers_file, "w")

    for i, id in enumerate(tqdm(data_ids)):
        ann = dataset.fetch_data(id)
        image_path = ann['image_path']

        qs = ann['question']
        cur_prompt = qs
        # pdb.set_trace()
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt', do_resize=True, 
                                                  do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        region_masks = ann['region_masks']
        if region_masks is not None:
            region_masks = [[region_mask_i.cuda().half() for region_mask_i in region_masks]]
        else:
            region_masks = None

        # pdb.set_trace()
        with torch.inference_mode():
            model.orig_forward = model.forward
            model.forward = partial(
                model.orig_forward,
                region_masks=region_masks
            )
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            model.forward = model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # pdb.set_trace()
        ans_file.write(json.dumps({"question_id":ann['question_id'],     # +1 offset     
                                   "image_path":image_path,
                                   "prompt": cur_prompt,
                                   "text": outputs
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="dataset/cocoval2017")
    parser.add_argument("--data_path", type=str, default="dataset/lvis/lvis_v1_minival_inserted_image_name.json")
    parser.add_argument("--answers-file", type=str, default="lvis_result/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="ferret_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--region_format", type=str, default="box", choices=["point", "box", "segment"])
    args = parser.parse_args()

    eval_model(args)
