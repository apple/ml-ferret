"""
Usage:
--data_path: path of pope annotation. 
--image_path:  path of coco2014 val images. 
--answers-file: path of output result.

Example:
CUDA_VISIBLE_DEVICES=0 python -m ferret.eval.model_pope \
    --model-path checkpoints/ferret_13b/checkpoint-final \
    --image_path data/refcoco/val2014 \
    --data_path data/pope/coco_pope_adversarial.json \
    --answers-file pope/coco_pope_adversarial \
    --add_region_feature \
    --chunk-idx 0 \
    --num-chunks 8 

"""

import argparse
from typing import Any, Tuple
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
import re
import math
import torchvision
import numpy as np
from copy import deepcopy

# Added for visualization
from PIL import Image, ImageDraw, ImageFont

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


def plot_pope(img, boxes, text):
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.load_default()
    draw.rectangle([boxes[0], boxes[1], boxes[2], boxes[3]], outline="blue")
    draw.text((boxes[0], boxes[1]-5), f'{text}', font=fnt, fill="green")
    return img


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def find_bbox_template_v3(text, img_w, img_h):
    pattern = r'\[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, text)
    new_bboxes = []
    old_bboxes = []
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        new_box = resize_bbox([x1, y1, x2, y2], img_w, img_h)
        new_bboxes.append(new_box)
        old_bboxes.append([x1, y1, x2, y2])
    
    set_old_bboxes = sorted(set(map(tuple, old_bboxes)), key=list(map(tuple, old_bboxes)).index)
    list_old_bboxes = list(map(list, set_old_bboxes))

    set_bboxes = sorted(set(map(tuple, new_bboxes)), key=list(map(tuple, new_bboxes)).index)
    list_bboxes = list(map(list, set_bboxes))

    for i in range(len(list_bboxes)):
        x1, y1, x2, y2 = list_old_bboxes[i]
        obj_string = '[obj{}]'.format(i)
        text = text.replace('[{}, {}, {}, {}]'.format(x1, y1, x2, y2), obj_string)
    
    return text, list_bboxes


class PopeGrounding():
    def __init__(self, img_folder, ann_file):
        self.img_folder = img_folder
        self.ann_file = ann_file

        self.label_list = [json.loads(q) for q in open(self.ann_file, 'r')]
        self._ids = range(len(self.label_list))

    def __getitem__(self, idx):
        label = self.label_list[idx]
        filename = label["image"]
        image = Image.open(os.path.join(self.img_folder, filename)).convert('RGB')
        question = label["text"]

        return image, question
    
    @property
    def ids(self):
        return deepcopy(self._ids)

        
def eval_model_pope(args):
    # Data
    dataset = PopeGrounding(img_folder=args.image_path, ann_file=args.data_path)
    data_ids = dataset.ids

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    chunk_data_ids = get_chunk(data_ids, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(answers_file, exist_ok=True)
    answers_file = os.path.join(answers_file, f'{args.chunk_idx}_of_{args.num_chunks}.jsonl')
    ans_file = open(answers_file, "w")

    for i, id in enumerate(tqdm(chunk_data_ids)):
        img, question = dataset[id]
        qs = question
        img_w, img_h = img.size

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(img, return_tensors='pt', do_resize=True, 
                                                  do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
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
                stopping_criteria=[stopping_criteria],
                )
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # Plot Preds
        # text, bboxes = find_bbox_template_v3(outputs, img_w=img_w, img_h=img_h)
        # # print(text, bboxes)
        # img = plot_pope(img, bboxes[0], text)
        # img.save('pope/images/{}.png'.format(i))

        ans_file.write(json.dumps({"question": question,
                                   "answer": outputs.lower(),
                                   }) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="data/refcoco/val2014")
    parser.add_argument("--data_path", type=str, default="data/pope/coco_pope_popular.json")
    parser.add_argument("--answers-file", type=str, default="pope/coco_pope_popular")
    parser.add_argument("--conv-mode", type=str, default="ferret_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model_pope(args)
