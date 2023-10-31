"""
Usage:
- If eval on center point:
CUDA_VISIBLE_DEVICES=1 python -m ferret.eval.model_point_cls_single_image \
    --model-path checkpoints/ferret_13b/checkpoint-4500 \
    --img_path ferret/serve/examples/extreme_ironing.jpg \
    --answers-file lvis_result/single_img/ \
    --add_region_feature
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm

from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ferret.model.builder import load_pretrained_model
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ferret.conversation import conv_templates, SeparatorStyle
from ferret.utils import disable_torch_init

from PIL import Image
import random
import math
from copy import deepcopy
import pdb
import numpy as np
from functools import partial

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"




def generate_mask_for_feature(coor,raw_w, raw_h):
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
    else:
        raise NotImplementedError('Coordinates must be 2d.')
    coor_mask = torch.from_numpy(coor_mask)
    assert len(coor_mask.nonzero()) != 0
    return coor_mask


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    image_path_list = ['ferret/serve/examples/extreme_ironing.jpg']
    # image_path_list = ['ferret/serve/examples/2409138.jpg', 'ferret/serve/examples/extreme_ironing.jpg', 'ferret/serve/examples/2332136.jpg']
    image_path = args.img_path
    coor_list = []
    grid_w = 10
    grid_h = 10
    for i in range(grid_w):
        for j in range(grid_h):
            coor_i = VOCAB_IMAGE_W * (i + 1) / (grid_w+1)
            coor_j = VOCAB_IMAGE_H * (j + 1) / (grid_h+1)
            coor_list.append([int(coor_i), int(coor_j)])

    if args.add_region_feature: 
        question = f'What is the class of object <coor> {DEFAULT_REGION_FEA_TOKEN}?'
    else:
        question = 'What is the class of object <coor>?'

    for image_path in image_path_list:
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(answers_file, exist_ok=True)
        image_name = image_path.split('.')[0].split('/')[-1]
        answers_file = os.path.join(answers_file, f'{image_name}.jsonl')
        ans_file = open(answers_file, "w")

        for i, coor_i in enumerate(tqdm(coor_list)):
            qs = question.replace('<coor>', '[{}, {}]'.format(int(coor_i[0]), int(coor_i[1])))
            cur_prompt = qs

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # inputs = tokenizer([prompt])

            image = Image.open(image_path).convert('RGB')
            # image.save(os.path.join(save_image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt', do_resize=True, 
                                                    do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]
            # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            if args.add_region_feature:
                generated_mask = generate_mask_for_feature(coor_i, VOCAB_IMAGE_W, VOCAB_IMAGE_H)
                region_masks = [generated_mask]
                region_masks = [[region_mask_i.cuda().half() for region_mask_i in region_masks]]
            else:
                region_masks = None

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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
                    max_new_tokens=1024,
                    num_beams=1,
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
            img_w, img_h = image.size
            ans_file.write(json.dumps({"img_w": img_w,
                                    "img_h": img_h,
                                    "VOCAB_IMAGE_W": VOCAB_IMAGE_W,
                                    "VOCAB_IMAGE_H": VOCAB_IMAGE_H,
                                    "coor": coor_i,  
                                    "image_path":image_path,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    }) + "\n")
            ans_file.flush()
        ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--img_path", type=str, default='llava/serve/examples/extreme_ironing.jpg')
    parser.add_argument("--answers-file", type=str, default="lvis_result/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="ferret_v1")
    parser.add_argument("--image_w", type=int, default=336)
    parser.add_argument("--image_h", type=int, default=336)
    parser.add_argument("--answer_prompter", action="store_true")
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    args = parser.parse_args()

    eval_model(args)
