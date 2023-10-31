"""
Usage:

python ferret/eval/eval_refexp.py \
    --prediction_file final_result/ferret_13b_checkpoint-final/refexp_result/finetune_refcocog_test \
    --annotation_file data/annotations/finetune_refcocog_test.json

"""
import os
import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.data
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from prettytable import PrettyTable

import re
import json

from misc.refcoco.box_ops import generalized_box_iou, box_iou

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def decode_bbox_from_caption(text, img_w, img_h, verbose=False):
    entities = []
    boxes = []
    
    start = 0
    in_brackets = False
    entity = ""
    box = ""
    
    for i, char in enumerate(text):
        if char == '[':
            in_brackets = True
            entity = text[start:i].strip()
            start = i + 1
        elif char == ']':
            in_brackets = False
            box = text[start:i].strip()
            start = i + 1
            
            # Convert box string to list of integers
            box_list = list(map(int, box.split(',')))
            resized_box_list = resize_bbox(box_list, img_w, img_h)
            entities.append(entity)
            boxes.append(resized_box_list)
            
            # Skip until the next entity (ignoring periods or other delimiters)
            while start < len(text) and text[start] not in ['.', ',', ';', '!', '?']:
                start += 1
            start += 1  # Skip the delimiter
        
    return entities, boxes


def are_phrases_similar(phrase1, phrase2):
    # Step 1: Convert to lower case
    phrase1 = phrase1.lower()
    phrase2 = phrase2.lower()
    
    # Step 2: Standardize spacing around punctuation
    phrase1 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase1).strip()
    phrase2 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase2).strip()
    
    # Step 3: Remove all punctuation
    phrase1 = re.sub(r'[^\w\s]', '', phrase1)
    phrase2 = re.sub(r'[^\w\s]', '', phrase2)
    
    # Step 4: Remove extra white spaces
    phrase1 = ' '.join(phrase1.split())
    phrase2 = ' '.join(phrase2.split())
    
    return phrase1 == phrase2


class RefExpEvaluatorFromJsonl(object):
    def __init__(self, refexp_gt_path, k=(1, -1), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        with open(refexp_gt_path, 'r') as f:
            self.refexp_gt = json.load(f)
        self.img_ids = [item['id'] for item in self.refexp_gt['images']]
        print(f"Load {len(self.img_ids)} images")
        print(f"Load {len(self.refexp_gt['annotations'])} annotations")
        self.k = k
        self.thresh_iou = thresh_iou

    def summarize(self,
                  prediction_file: str,
                  verbose: bool = False,):
        
        # get the predictions
        if os.path.isfile(prediction_file):
            predictions = [json.loads(line) for line in open(prediction_file)]
        elif os.path.isdir(prediction_file):
            predictions = [json.loads(line) for pred_file in os.listdir(prediction_file) for line in open(os.path.join(prediction_file, pred_file))]
        else:
            raise NotImplementedError('Not supported file format.')
        
        # sort the predictions based on 'image_id'
        predictions = sorted(predictions, key=lambda x: x['image_id'])

        predict_index = 0
        
        dataset2score = {
            "refcoco": {k: 0.0 for k in self.k},
            "refcoco+": {k: 0.0 for k in self.k},
            "refcocog": {k: 0.0 for k in self.k},
        }
        dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}
        for item_img, item_ann in tqdm(zip(self.refexp_gt['images'], self.refexp_gt['annotations'])):

            # quit when evaluating all predictions
            if predict_index == len(predictions):
                break
                
            if item_img['id'] != item_ann['image_id']:
                raise ValueError(f"Ann\n{item_ann} \nis not matched\n {item_img}")
            
            dataset_name = item_img['dataset_name']
            img_height = item_img['height']
            img_width = item_img['width']
            caption = item_img['caption']
            target_bbox = item_ann["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            target_bbox = torch.as_tensor(converted_bbox).view(-1, 4)

            prediction = predictions[predict_index]["text"]
            try:
                entities, boxes = decode_bbox_from_caption(prediction, img_width, img_height, verbose=verbose)
            except ValueError as e:
                entities, boxes = [], []

            predict_boxes = []
            for (entity, box) in zip(entities, boxes):
                if not are_phrases_similar(entity, caption):
                    if len(box) > 0:
                        predict_boxes.append(box)
                else:
                    predict_boxes.append(box)
            
            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase {caption}, \n{entities, boxes}")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]                                                                                                               

            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4).to(dtype=torch.float32)
            
            iou, _ = box_iou(predict_boxes, target_bbox)
            mean_iou, _ = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
            for k in self.k:
                if k == 'upper bound':
                    if max(iou) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                elif k == 'mean':
                    if max(mean_iou) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                else:
                    if max(iou[0, :k]) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0

            dataset2count[dataset_name] += 1.0
            predict_index += 1

        for key, value in dataset2score.items():
            for k in self.k:
                try:
                    value[k] /= dataset2count[key]
                except:
                    pass
                
        results = {}
        for key, value in dataset2score.items():
            results[key] = sorted([v for k, v in value.items()])
            print(f" Dataset: {key} - Precision @ 1, mean, all: {results[key]} \n")
        
        return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/json_annotations', help='annotation_file')
    
    args = parser.parse_args()
    
    evaluator = RefExpEvaluatorFromJsonl(
        refexp_gt_path=args.annotation_file, 
        k=(1, 'mean', 'upper bound'), 
        thresh_iou=0.5,
    )
    
    results = evaluator.summarize(args.prediction_file, verbose=False)
    
    with open(os.path.join(args.prediction_file, "metric.json"), "w") as f:
        json.dump(results, f, indent=2)