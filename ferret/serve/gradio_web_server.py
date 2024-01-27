'''
Usage:

python -m ferret.serve.gradio_web_server --controller http://localhost:10000 --add_region_feature
'''
import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from ferret.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from ferret.constants import LOGDIR
from ferret.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib
# Added 
import re
from copy import deepcopy
from PIL import ImageDraw, ImageFont
from gradio import processing_utils
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
import pdb
from ferret.serve.gradio_css import code_highlight_css

DEFAULT_REGION_REFER_TOKEN = "[region]"
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FERRET Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}

VOCAB_IMAGE_W = 1000  # 224
VOCAB_IMAGE_H = 1000  # 224

def generate_mask_for_feature(coor, raw_w, raw_h, mask=None):
    if mask is not None:
        assert mask.shape[0] == raw_w and mask.shape[1] == raw_h
    coor_mask = torch.zeros((raw_w, raw_h))
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
        coor_mask = torch.zeros((raw_w, raw_h))
        coor_mask[coor[0]:coor[2]+1, coor[1]:coor[3]+1] = 1
        if mask is not None:
            coor_mask = coor_mask * mask
    # coor_mask = torch.from_numpy(coor_mask)
    # pdb.set_trace()
    assert len(coor_mask.nonzero()) != 0
    return coor_mask.tolist()


def draw_box(coor, region_mask, region_ph, img, input_mode):
    colors = ["red"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./ferret/serve/dejavu/DejaVuSans.ttf", size=18)
    if input_mode == 'Box':
        draw.rectangle([coor[0], coor[1], coor[2], coor[3]], outline=colors[0], width=4)
        draw.rectangle([coor[0], coor[3] - int(font.size * 1.2), coor[0] + int((len(region_ph) + 0.8) * font.size * 0.6), coor[3]], outline=colors[0], fill=colors[0], width=4)
        draw.text([coor[0] + int(font.size * 0.2), coor[3] - int(font.size*1.2)], region_ph, font=font, fill=(255,255,255))
    elif input_mode == 'Point':
        r = 8 
        leftUpPoint = (coor[0]-r, coor[1]-r)
        rightDownPoint = (coor[0]+r, coor[1]+r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, outline=colors[0], width=4)
        draw.rectangle([coor[0], coor[1], coor[0] + int((len(region_ph) + 0.8) * font.size * 0.6), coor[1] + int(font.size * 1.2)], outline=colors[0], fill=colors[0], width=4)
        draw.text([coor[0] + int(font.size * 0.2), coor[1]], region_ph, font=font, fill=(255,255,255))
    elif input_mode == 'Sketch':
        draw.rectangle([coor[0], coor[3] - int(font.size * 1.2), coor[0] + int((len(region_ph) + 0.8) * font.size * 0.6), coor[3]], outline=colors[0], fill=colors[0], width=4)
        draw.text([coor[0] + int(font.size * 0.2), coor[3] - int(font.size*1.2)], region_ph, font=font, fill=(255,255,255))
        # Use morphological operations to find the boundary
        mask = np.array(region_mask)
        dilated = binary_dilation(mask, structure=np.ones((3,3)))
        eroded = binary_erosion(mask, structure=np.ones((3,3)))
        boundary = dilated ^ eroded  # XOR operation to find the difference between dilated and eroded mask
        # Loop over the boundary and paint the corresponding pixels
        for i in range(boundary.shape[0]):
            for j in range(boundary.shape[1]):
                if boundary[i, j]:
                    # This is a pixel on the boundary, paint it red
                    draw.point((i, j), fill=colors[0])
    else:
        NotImplementedError(f'Input mode of {input_mode} is not Implemented.')
    return img


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()
    return (state,
            dropdown_update,
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (state, gr.Dropdown.update(
               choices=models,
               value=models[0] if len(models) > 0 else ""),
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5 + \
        (None, {'region_placeholder_tokens':[],'region_coordinates':[],'region_masks':[],'region_masks_in_prompts':[],'masks':[]}, [], None)


def resize_bbox(box, image_w=None, image_h=None, default_wh=VOCAB_IMAGE_W):
    ratio_w = image_w * 1.0 / default_wh
    ratio_h = image_h * 1.0 / default_wh

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def show_location(sketch_pad, chatbot):
    image = sketch_pad['image']
    img_w, img_h = image.size
    new_bboxes = []
    old_bboxes = []
    # chatbot[0] is image.
    text = chatbot[1:]
    for round_i in text:
        human_input = round_i[0]
        model_output = round_i[1]
        # TODO: Difference: vocab representation.
        # pattern = r'\[x\d*=(\d+(?:\.\d+)?), y\d*=(\d+(?:\.\d+)?), x\d*=(\d+(?:\.\d+)?), y\d*=(\d+(?:\.\d+)?)\]'
        pattern = r'\[(\d+(?:\.\d+)?), (\d+(?:\.\d+)?), (\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\]'
        matches = re.findall(pattern, model_output)
        for match in matches:
            x1, y1, x2, y2 = map(int, match)
            new_box = resize_bbox([x1, y1, x2, y2], img_w, img_h)
            new_bboxes.append(new_box)
            old_bboxes.append([x1, y1, x2, y2])
        
    set_old_bboxes = sorted(set(map(tuple, old_bboxes)), key=list(map(tuple, old_bboxes)).index)
    list_old_bboxes = list(map(list, set_old_bboxes))

    set_bboxes = sorted(set(map(tuple, new_bboxes)), key=list(map(tuple, new_bboxes)).index)
    list_bboxes = list(map(list, set_bboxes))

    output_image = deepcopy(image)
    draw = ImageDraw.Draw(output_image)
    font = ImageFont.truetype("./ferret/serve/dejavu/DejaVuSans.ttf", 28)
    for i in range(len(list_bboxes)):
        x1, y1, x2, y2 = list_old_bboxes[i]
        x1_new, y1_new, x2_new, y2_new = list_bboxes[i]
        obj_string = '[obj{}]'.format(i)
        for round_i in text:
            model_output = round_i[1]
            model_output = model_output.replace('[{}, {}, {}, {}]'.format(x1, y1, x2, y2), obj_string)
            round_i[1] = model_output
        draw.rectangle([(x1_new, y1_new), (x2_new, y2_new)], outline="red", width=3)
        draw.text((x1_new+2, y1_new+5), obj_string[1:-1], fill="red", font=font)

    return (output_image, [chatbot[0]] + text, disable_btn)


def add_text(state, text, image_process_mode, original_image, sketch_pad, request: gr.Request):
    image = sketch_pad['image']

    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if original_image is None:
        assert image is not None
        original_image = image.copy()
        print('No location, copy original image in add_text')

    if image is not None:
        if state.first_round:
            text = text[:1200]  # Hard cut-off for images
            if '<image>' not in text:
                # text = '<Image><image></Image>' + text
                text = text + '\n<image>'
            text = (text, original_image, image_process_mode)
            if len(state.get_images(return_pil=True)) > 0:
                new_state = default_conversation.copy()
                new_state.first_round = False
                state=new_state
                print('First round add image finsihed.')

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", original_image) + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def find_indices_in_order(str_list, STR):
    indices = []
    i = 0
    while i < len(STR):
        for element in str_list:
            if STR[i:i+len(element)] == element:
                indices.append(str_list.index(element))
                i += len(element) - 1
                break
        i += 1
    return indices


def format_region_prompt(prompt, refer_input_state):
    # Find regions in prompts and assign corresponding region masks
    refer_input_state['region_masks_in_prompts'] = []
    indices_region_placeholder_in_prompt = find_indices_in_order(refer_input_state['region_placeholder_tokens'], prompt)
    refer_input_state['region_masks_in_prompts'] = [refer_input_state['region_masks'][iii] for iii in indices_region_placeholder_in_prompt]

    # Find regions in prompts and replace with real coordinates and region feature token.
    for region_ph_index, region_ph_i in enumerate(refer_input_state['region_placeholder_tokens']):
        prompt = prompt.replace(region_ph_i, '{} {}'.format(refer_input_state['region_coordinates'][region_ph_index], DEFAULT_REGION_FEA_TOKEN))
    return prompt
    

def http_bot(state, model_selector, temperature, top_p, max_new_tokens, refer_input_state, request: gr.Request):
# def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        template_name = 'ferret_v1'
        # Below is LLaVA's original templates.
        # if "llava" in model_name.lower():
        #     if 'llama-2' in model_name.lower():
        #         template_name = "llava_llama_2"
        #     elif "v1" in model_name.lower():
        #         if 'mmtag' in model_name.lower():
        #             template_name = "v1_mmtag"
        #         elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
        #             template_name = "v1_mmtag"
        #         else:
        #             template_name = "llava_v1"
        #     elif "mpt" in model_name.lower():
        #         template_name = "mpt"
        #     else:
        #         if 'mmtag' in model_name.lower():
        #             template_name = "v0_mmtag"
        #         elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
        #             template_name = "v0_mmtag"
        #         else:
        #             template_name = "llava_v0"
        # elif "mpt" in model_name:
        #     template_name = "mpt_text"
        # elif "llama-2" in model_name:
        #     template_name = "llama_2"
        # else:
        #     template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state
        state.first_round = False

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Construct prompt
    prompt = state.get_prompt()
    if args.add_region_feature:
        prompt = format_region_prompt(prompt, refer_input_state)

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")
    if args.add_region_feature:
        pload['region_masks'] = refer_input_state['region_masks_in_prompts']
        logger.info(f"==== add region_masks_in_prompts to request ====\n")

    pload['images'] = state.get_images()
    print(f'Input Prompt: {prompt}')

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# 🦦 Ferret: Refer and Ground Anything Anywhere at Any Granularity
""")
# [[Project Page]](https://llava-vl.github.io) [[Paper]](https://arxiv.org/abs/2304.08485)

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only
""")


css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""

Instructions = '''
Instructions:
1. Select a 'Referring Input Type'
2. Draw on the image to refer to a region/point.
3. Copy the region id from 'Referring Input Type' to refer to a region in your chat.
'''

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)
    

def draw(input_mode, input, refer_input_state, refer_text_show, imagebox_refer):
    if type(input) == dict:
        image = deepcopy(input['image'])
        mask = deepcopy(input['mask'])
    else:
        mask = deepcopy(input)

    # W, H -> H, W, 3
    image_new = np.asarray(image)
    img_height = image_new.shape[0]
    img_width = image_new.shape[1]

    # W, H, 4 -> H, W
    mask_new = np.asarray(mask)[:,:,0].copy()
    mask_new = torch.from_numpy(mask_new)
    mask_new = (F.interpolate(mask_new.unsqueeze(0).unsqueeze(0), (img_height, img_width), mode='bilinear') > 0)
    mask_new = mask_new[0, 0].transpose(1, 0).long()

    if len(refer_input_state['masks']) == 0:
        last_mask = torch.zeros_like(mask_new)
    else:
        last_mask = refer_input_state['masks'][-1]

    diff_mask = mask_new - last_mask
    if mask_new.sum() == 0:
        return (refer_input_state, refer_text_show, image)
    elif torch.all(diff_mask == 0):
        return (refer_input_state, refer_input_state['refer_text_show'][-1], refer_input_state['imagebox_refer'][-1])
    else:
        refer_input_state['masks'].append(mask_new)

    if input_mode == 'Point':
        nonzero_points = diff_mask.nonzero()
        nonzero_points_avg_x = torch.median(nonzero_points[:, 0])
        nonzero_points_avg_y = torch.median(nonzero_points[:, 1])
        sampled_coor = [nonzero_points_avg_x, nonzero_points_avg_y]
        # pdb.set_trace()
        cur_region_masks = generate_mask_for_feature(sampled_coor, raw_w=img_width, raw_h=img_height)
    elif input_mode == 'Box' or input_mode == 'Sketch':
        # pdb.set_trace()
        x1x2 = diff_mask.max(1)[0].nonzero()[:, 0]
        y1y2 = diff_mask.max(0)[0].nonzero()[:, 0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()
        # pdb.set_trace()
        sampled_coor = [x1, y1, x2, y2]
        if input_mode == 'Box':
            cur_region_masks = generate_mask_for_feature(sampled_coor, raw_w=img_width, raw_h=img_height)
        else:
            cur_region_masks = generate_mask_for_feature(sampled_coor, raw_w=img_width, raw_h=img_height, mask=diff_mask)
    else:
        raise NotImplementedError(f'Input mode of {input_mode} is not Implemented.')

    # TODO(haoxuan): Hack img_size to be 224 here, need to make it a argument.
    if len(sampled_coor) == 2:
        point_x = int(VOCAB_IMAGE_W * sampled_coor[0] / img_width)
        point_y = int(VOCAB_IMAGE_H * sampled_coor[1] / img_height)
        cur_region_coordinates = f'[{int(point_x)}, {int(point_y)}]'
    elif len(sampled_coor) == 4:
        point_x1 = int(VOCAB_IMAGE_W * sampled_coor[0] / img_width)
        point_y1 = int(VOCAB_IMAGE_H * sampled_coor[1] / img_height)
        point_x2 = int(VOCAB_IMAGE_W * sampled_coor[2] / img_width)
        point_y2 = int(VOCAB_IMAGE_H * sampled_coor[3] / img_height)
        cur_region_coordinates = f'[{int(point_x1)}, {int(point_y1)}, {int(point_x2)}, {int(point_y2)}]'

    cur_region_id = len(refer_input_state['region_placeholder_tokens'])
    cur_region_token = DEFAULT_REGION_REFER_TOKEN.split(']')[0] + str(cur_region_id) + ']'
    refer_input_state['region_placeholder_tokens'].append(cur_region_token)
    refer_input_state['region_coordinates'].append(cur_region_coordinates)
    refer_input_state['region_masks'].append(cur_region_masks)
    assert len(refer_input_state['region_masks']) == len(refer_input_state['region_coordinates']) == len(refer_input_state['region_placeholder_tokens'])
    refer_text_show.append((cur_region_token, None))

    # Show Parsed Referring.
    imagebox_refer = draw_box(sampled_coor, cur_region_masks, \
                         cur_region_token, imagebox_refer, input_mode)

    refer_input_state['refer_text_show'].append(refer_text_show)
    refer_input_state['imagebox_refer'].append(imagebox_refer)

    return (refer_input_state, refer_text_show, imagebox_refer)

def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", visible=False, container=False)
    with gr.Blocks(title="FERRET", theme=gr.themes.Base(), css=css) as demo:
        state = gr.State()
        state.skip_next = False

        if not embed_mode:
            gr.Markdown(title_markdown)
            gr.Markdown(Instructions)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                original_image = gr.Image(type="pil", visible=False)
                image_process_mode = gr.Radio(
                    ["Raw+Processor", "Crop", "Resize", "Pad"],
                    value="Raw+Processor",
                    label="Preprocess for non-square image",
                    visible=False)

                # Added for any-format input.
                sketch_pad = gr.ImageMask(label="Image & Sketch", type="pil", elem_id="img2text")
                refer_input_mode = gr.Radio(
                    ["Point", "Box", "Sketch"],
                    value="Point",
                    label="Referring Input Type")
                refer_input_state = gr.State({'region_placeholder_tokens':[],
                                              'region_coordinates':[],
                                              'region_masks':[],
                                              'region_masks_in_prompts':[],
                                              'masks':[],
                                              'refer_text_show': [],
                                              'imagebox_refer': [],
                                              })
                refer_text_show = gr.HighlightedText(value=[], label="Referring Input Cache")

                imagebox_refer = gr.Image(type="pil", label="Parsed Referring Input")
                imagebox_output = gr.Image(type="pil", label='Output Vis')

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    # [f"{cur_dir}/examples/harry-potter-hogwarts.jpg", "What is in [region0]? And what do people use it for?"],
                    # [f"{cur_dir}/examples/ingredients.jpg", "What objects are in [region0] and [region1]?"],
                    # [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image? And tell me the coordinates of mentioned objects."],
                    [f"{cur_dir}/examples/ferret.jpg", "What's the relationship between object [region0] and object [region1]?"],
                    [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here? Tell me the coordinates in response."],
                    [f"{cur_dir}/examples/flickr_9472793441.jpg", "Describe the image in details."],
                    # [f"{cur_dir}/examples/coco_000000281759.jpg", "What are the locations of the woman wearing a blue dress, the woman in flowery top, the girl in purple dress, the girl wearing green shirt?"],
                    [f"{cur_dir}/examples/room_planning.jpg", "How to improve the design of the given room?"],
                    [f"{cur_dir}/examples/make_sandwitch.jpg", "How can I make a sandwich with available ingredients?"],
                    [f"{cur_dir}/examples/bathroom.jpg", "What is unusual about this image?"],
                    [f"{cur_dir}/examples/kitchen.png", "Is the object a man or a chicken? Explain the reason."],
                ], inputs=[sketch_pad, textbox])

                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=5):
                chatbot = gr.Chatbot(elem_id="chatbot", label="FERRET", visible=False).style(height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    # flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
                    location_btn = gr.Button(value="🪄 Show location", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, location_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, location_btn])
        downvote_btn.click(downvote_last_response,
            [state, model_selector], [textbox, upvote_btn, downvote_btn, location_btn])
        # flag_btn.click(flag_last_response,
        #     [state, model_selector], [textbox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, [state, image_process_mode],
            [state, chatbot, textbox] + btn_list).then(
            http_bot, [state, model_selector, temperature, top_p, max_output_tokens, refer_input_state],
            [state, chatbot] + btn_list)
        clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox_output, original_image] + btn_list + \
                        [sketch_pad, refer_input_state, refer_text_show, imagebox_refer])
        location_btn.click(show_location,
            [sketch_pad, chatbot], [imagebox_output, chatbot, location_btn])

        textbox.submit(add_text, [state, textbox, image_process_mode, original_image, sketch_pad], [state, chatbot, textbox, original_image] + btn_list
            ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens, refer_input_state],
                   [state, chatbot] + btn_list)

        submit_btn.click(add_text, [state, textbox, image_process_mode, original_image, sketch_pad], [state, chatbot, textbox, original_image] + btn_list
            ).then(http_bot, [state, model_selector, temperature, top_p, max_output_tokens, refer_input_state],
                   [state, chatbot] + btn_list)

        sketch_pad.edit(
            draw,
            inputs=[refer_input_mode, sketch_pad, refer_input_state, refer_text_show, imagebox_refer],
            outputs=[refer_input_state, refer_text_show, imagebox_refer],
            queue=True,
        )

        if args.model_list_mode == "once":
            demo.load(load_demo, [url_params], [state, model_selector,
                chatbot, textbox, submit_btn, button_row, parameter_row],
                _js=get_window_url_params)
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector,
                chatbot, textbox, submit_btn, button_row, parameter_row])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--add_region_feature", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10,
               api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share)
