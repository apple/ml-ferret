import argparse
import torch
import os
import json
from tqdm import tqdm

from ferretui.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_REGION_FEA_TOKEN, VOCAB_IMAGE_W, VOCAB_IMAGE_H
from ferretui.conversation import conv_templates, SeparatorStyle
from ferretui.model.builder import load_pretrained_model
from ferretui.utils import disable_torch_init
from ferretui.mm_utils import tokenizer_image_token, process_images

from PIL import Image
import math
import pdb
import numpy as np
from copy import deepcopy
from functools import partial


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

def get_task_from_file(file):
    box_in_tasks = ['widgetcaptions', 'taperception', 'ocr', 'icon_recognition', 'widget_classification', 'example_0']
    # box_out_tasks = ['widget_listing', 'find_text', 'find_icons', 'find_widget', 'conversation_interaction']
    # no_box = ['screen2words', 'detailed_description', 'conversation_perception', 'gpt4']
    if any(task in file for task in box_in_tasks):
        return 'box_in'
    else:
        return 'no_box_in'
    # elif any(task in file for task in box_out_tasks):
    #     return 'box_out'
    # elif any(task in file for task in no_box):
    #     return 'no_box'
    
def get_bbox_coor(box, ratio_w, ratio_h):
    return box[0] * ratio_w, box[1] * ratio_h, box[2] * ratio_w, box[3] * ratio_h

def get_model_name_from_path(model_path):
    if 'gemma' in model_path:
        return 'ferret_gemma'
    elif 'llama' or 'vicuna' in model_path:
        return 'ferret_llama'
    else:
        raise ValueError(f"No model matched for {model_path}")

class UIData:
    def __init__(self, data_path, image_path, args) -> None:
        self.obj_list = json.load(open(data_path, 'r'))
        self.image_path = image_path
        self.args = args
        self._ids = range(len(self.obj_list))
        self.task = get_task_from_file(data_path)

    @property
    def ids(self):
        return deepcopy(self._ids)

    def __getitem__(self, idx):
        i = self.obj_list[idx]

        # image stuff
        image_path_i = os.path.join(self.image_path, i['image'].split('/')[-1])
        image = Image.open(image_path_i).convert('RGB')

        q_turn = i['conversations'][0]['value']
        if "<image>" in q_turn:
            prompt = q_turn.split('\n')[1]
        else:
            prompt = q_turn
        i['question'] = prompt
        i['region_masks'] = None

        if self.task == 'box_in':
            ratio_w = VOCAB_IMAGE_W * 1.0 / i['image_w']
            ratio_h = VOCAB_IMAGE_H * 1.0 / i['image_h']

            box = i['box_x1y1x2y2'][0][0]
            box_x1, box_y1, box_x2, box_y2 = box
            box_x1_textvocab, box_y1_textvocab, box_x2_textvocab, box_y2_textvocab = get_bbox_coor(box=box, ratio_h=ratio_h, ratio_w=ratio_w)

            if self.args.region_format == 'box':
                region_coordinate_raw = [box_x1, box_y1, box_x2, box_y2]
                if args.add_region_feature:
                    i['question'] = prompt.replace('<bbox_location0>', '[{}, {}, {}, {}] {}'.format(int(box_x1_textvocab), int(box_y1_textvocab), int(box_x2_textvocab), int(box_y2_textvocab), DEFAULT_REGION_FEA_TOKEN))
                    generated_mask = generate_mask_for_feature(region_coordinate_raw, raw_w=i['image_w'], raw_h=i['image_h'], mask=None)
                    i['region_masks'] = [generated_mask]
                else:
                    i['question'] = prompt.replace('<bbox_location0>', '[{}, {}, {}, {}]'.format(int(box_x1_textvocab), int(box_y1_textvocab), int(box_x2_textvocab), int(box_y2_textvocab)))
            else:
                raise NotImplementedError(f'{self.args.region_format} is not supported.')

        return image, i, image.size

def eval_model(args):
    # Data
    dataset = UIData(data_path=args.data_path, image_path=args.image_path, args=args)
    data_ids = dataset.ids

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = \
        load_pretrained_model(model_path, args.model_base, model_name, use_safetensors=True)

    chunk_data_ids = get_chunk(data_ids, args.num_chunks, args.chunk_idx)
    answers_folder = os.path.expanduser(args.answers_file)
    os.makedirs(answers_folder, exist_ok=True)
    answers_file = os.path.join(answers_folder, f'{args.chunk_idx}_of_{args.num_chunks}.jsonl')
    ans_file = open(answers_file, "w")

    for i, id in enumerate(tqdm(chunk_data_ids)):
        img, ann, image_size = dataset[id]
        image_path = ann['image']
        qs = ann["question"]
        cur_prompt = qs

        if "<image>" in qs:
            qs = qs.split('\n')[1]

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        if model.config.image_aspect_ratio == "square_nocrop":
            image_tensor = image_processor.preprocess(img, return_tensors='pt', do_resize=True, 
                                                  do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]
        elif model.config.image_aspect_ratio == "anyres":
            image_process_func = partial(image_processor.preprocess, return_tensors='pt', do_resize=True, do_center_crop=False, size=[args.image_h, args.image_w])
            image_tensor = process_images([img], image_processor, model.config, image_process_func=image_process_func)[0]
        else:
            image_tensor = process_images([img], image_processor, model.config)[0]

        images = image_tensor.unsqueeze(0).to(args.data_type).cuda()

        region_masks = ann['region_masks']
        
        if region_masks is not None:
            region_masks = [[region_mask_i.cuda().half() for region_mask_i in region_masks]]
        else:
            region_masks = None

        with torch.inference_mode():
            model.orig_forward = model.forward
            model.forward = partial(
                model.orig_forward,
                region_masks=region_masks
            )
            output_ids = model.generate(
                input_ids,
                images=images,
                region_masks=region_masks,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            model.forward = model.orig_forward

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if 'label' in ann:
            label = ann['label']
        elif len(ann['conversations']) > 1:
            label = ann['conversations'][1]['value']
        else: 
            label = None

        ans_file.write(json.dumps({"id":ann['id'],     # +1 offset   
                                    "image_path":image_path,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "label": label,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--vision_model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="")
    parser.add_argument("--conv_mode", type=str, default="ferret_gemma_instruct",
                        help="[ferret_gemma_instruct,ferret_llama_3,ferret_vicuna_v1]")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--region_format", type=str, default="point", choices=["point", "box", "segment", "free_shape"])
    parser.add_argument("--no_coor", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--data_type", type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    args = parser.parse_args()

    if args.data_type == 'fp16':
        args.data_type = torch.float16
    elif args.data_type == 'bf16':
        args.data_type = torch.bfloat16
    else:
        args.data_type = torch.float32
    
    eval_model(args)