"""
Usage:

python ferret/eval/eval_flickr_entities.py \
    --prediction_file result_checkpoint-final/flickr_result/final_flickr_mergedGT_test \
    --annotation_file data/annotations/final_flickr_mergedGT_test.json \
    --flickr_entities_path data/flickr30k

"""


import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

import json
import os
import re

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
            box_list = [int(coord) for coord in box.split(',')]
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


def get_sentence_data(filename) -> List[Dict[str, Any]]:
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      filename - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(filename, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {"first_word_index": index, "phrase": phrase, "phrase_id": p_id, "phrase_type": p_type}
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(filename) -> Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]]:
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      filename - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
          height - int representing the height of the image
          width - int representing the width of the image
          depth - int representing the depth of the image
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info: Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]] = {}
    all_boxes: Dict[str, List[List[int]]] = {}
    all_noboxes: List[str] = []
    all_scenes: List[str] = []
    for size_element in size_container:
        assert size_element.text
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            assert box_id
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in all_boxes:
                    all_boxes[box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text)
                ymin = int(box_container[0].findall("ymin")[0].text)
                xmax = int(box_container[0].findall("xmax")[0].text)
                ymax = int(box_container[0].findall("ymax")[0].text)
                all_boxes[box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    all_noboxes.append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    all_scenes.append(box_id)
    anno_info["boxes"] = all_boxes
    anno_info["nobox"] = all_noboxes
    anno_info["scene"] = all_scenes

    return anno_info


#### END of import from flickr30k_entities
#### Bounding box utilities imported from torchvision and converted to numpy
def box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


#### End of import of box utilities

def _merge_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """
    Return the boxes corresponding to the smallest enclosing box containing all the provided boxes
    The boxes are expected in [x1, y1, x2, y2] format
    """
    if len(boxes) == 1:
        return boxes

    np_boxes = np.asarray(boxes)

    return [[np_boxes[:, 0].min(), np_boxes[:, 1].min(), np_boxes[:, 2].max(), np_boxes[:, 3].max()]]

class RecallTracker:
    """ Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat] for cat in self.total_byk_bycat[k]
            }
        return report


class Flickr30kEntitiesRecallEvaluator:
    def __init__(
        self,
        flickr_path: str,
        subset: str = "test",
        topk: Sequence[int] = (1, 5, 10, -1),
        iou_thresh: float = 0.5,
        merge_boxes: bool = False,
        verbose: bool = True,
    ):

        assert subset in ["train", "test", "val"], f"Wrong flickr subset {subset}"

        self.topk = topk
        self.iou_thresh = iou_thresh

        flickr_path = Path(flickr_path)

        # We load the image ids corresponding to the current subset
        with open(flickr_path / f"{subset}.txt") as file_d:
            self.img_ids = [line.strip() for line in file_d]

        if verbose:
            print(f"Flickr subset contains {len(self.img_ids)} images")

        # Read the box annotations for all the images
        self.imgid2boxes: Dict[str, Dict[str, List[List[int]]]] = {}

        if verbose:
            print("Loading annotations...")

        for img_id in self.img_ids:
            anno_info = get_annotations(flickr_path / "Annotations" / f"{img_id}.xml")["boxes"]
            if merge_boxes:
                merged = {}
                for phrase_id, boxes in anno_info.items():
                    merged[phrase_id] = _merge_boxes(boxes)
                anno_info = merged
            self.imgid2boxes[img_id] = anno_info

        # Read the sentences annotations
        self.imgid2sentences: Dict[str, List[List[Optional[Dict]]]] = {}

        if verbose:
            print("Loading annotations...")

        self.all_ids: List[str] = []
        tot_phrases = 0
        for img_id in self.img_ids:
            sentence_info = get_sentence_data(flickr_path / "Sentences" / f"{img_id}.txt")
            self.imgid2sentences[img_id] = [None for _ in range(len(sentence_info))]

            # Some phrases don't have boxes, we filter them.
            for sent_id, sentence in enumerate(sentence_info):
                phrases = [phrase for phrase in sentence["phrases"] if phrase["phrase_id"] in self.imgid2boxes[img_id]]
                if len(phrases) > 0:
                    self.imgid2sentences[img_id][sent_id] = phrases
                tot_phrases += len(phrases)

            self.all_ids += [
                f"{img_id}_{k}" for k in range(len(sentence_info)) if self.imgid2sentences[img_id][k] is not None
            ]

        if verbose:
            print(f"There are {tot_phrases} phrases in {len(self.all_ids)} sentences to evaluate")

    def evaluate(self, predictions: List[Dict]):
        evaluated_ids = set()

        recall_tracker = RecallTracker(self.topk)

        for pred in predictions:
            cur_id = f"{pred['image_id']}_{pred['sentence_id']}"
            if cur_id in evaluated_ids:
                print(
                    "Warning, multiple predictions found for sentence"
                    f"{pred['sentence_id']} in image {pred['image_id']}"
                )
                continue

            # Skip the sentences with no valid phrase
            if cur_id not in self.all_ids:
                if len(pred["boxes"]) != 0:
                    print(
                        f"Warning, in image {pred['image_id']} we were not expecting predictions "
                        f"for sentence {pred['sentence_id']}. Ignoring them."
                    )
                continue

            evaluated_ids.add(cur_id)

            pred_boxes = pred["boxes"]
            if str(pred["image_id"]) not in self.imgid2sentences:
                raise RuntimeError(f"Unknown image id {pred['image_id']}")
            if not 0 <= int(pred["sentence_id"]) < len(self.imgid2sentences[str(pred["image_id"])]):
                raise RuntimeError(f"Unknown sentence id {pred['sentence_id']}" f" in image {pred['image_id']}")
            target_sentence = self.imgid2sentences[str(pred["image_id"])][int(pred["sentence_id"])]

            phrases = self.imgid2sentences[str(pred["image_id"])][int(pred["sentence_id"])]
            if len(pred_boxes) != len(phrases):
                raise RuntimeError(
                    f"Error, got {len(pred_boxes)} predictions, expected {len(phrases)} "
                    f"for sentence {pred['sentence_id']} in image {pred['image_id']}"
                )

            for cur_boxes, phrase in zip(pred_boxes, phrases):
                target_boxes = self.imgid2boxes[str(pred["image_id"])][phrase["phrase_id"]]

                ious = box_iou(np.asarray(cur_boxes), np.asarray(target_boxes))
                for k in self.topk:
                    maxi = 0
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thresh:
                        recall_tracker.add_positive(k, "all")
                        for phrase_type in phrase["phrase_type"]:
                            recall_tracker.add_positive(k, phrase_type)
                    else:
                        recall_tracker.add_negative(k, "all")
                        for phrase_type in phrase["phrase_type"]:
                            recall_tracker.add_negative(k, phrase_type)

        if len(evaluated_ids) != len(self.all_ids):
            print("ERROR, the number of evaluated sentence doesn't match. Missing predictions:")
            un_processed = set(self.all_ids) - evaluated_ids
            for missing in un_processed:
                img_id, sent_id = missing.split("_")
                print(f"\t sentence {sent_id} in image {img_id}")
            raise RuntimeError("Missing predictions")

        return recall_tracker.report()


class Flickr30kEntitiesRecallEvaluatorFromJsonl(Flickr30kEntitiesRecallEvaluator):
    def evaluate(self, 
                 annotation_file: str,
                 prediction_file: str,
                 verbose: bool = False,
                ):
        recall_tracker = RecallTracker(self.topk)
        
        gt_json = json.load(open(annotation_file, 'r', encoding='utf-8'))

        # get the predictions
        if os.path.isfile(prediction_file):
            predictions = [json.loads(line) for line in open(prediction_file)]
        elif os.path.isdir(prediction_file):
            predictions = [json.loads(line) for pred_file in sorted(os.listdir(prediction_file)) for line in open(os.path.join(prediction_file, pred_file))]
        else:
            raise NotImplementedError('Not supported file format.')
        
        predict_index = 0
        
        valid_cnt = 0
        for item in tqdm(gt_json['images']):
            file_name = item["file_name"]
            caption = item["caption"]
            img_height = float(item['height'])
            img_width = float(item['width'])
            postive_item_pos = item['tokens_positive_eval']
            
            # to verify 
            phrases_from_self = self.imgid2sentences[str(item['original_img_id'])][int(item['sentence_id'])]
            for pos in postive_item_pos:
                # pdb.set_trace()
                if predict_index == len(predictions):
                    break
                
                pos_start, pos_end = pos[0]
                phrase = caption[pos_start:pos_end]
                phrase_from_self = [p for p in phrases_from_self if p['phrase'] == phrase]
                if len(phrase_from_self) == 0:
                    raise ValueError(f"Can't find the corresponding gt from two file {phrase} vs. {phrases_from_self}")
                else:
                    phrase_from_self = phrase_from_self[0]
                
                # get the prediction from text line
                try:
                    prediction = predictions[predict_index]["text"]
                except IndexError as e:
                    print("Raise Indexerror.")
                    print(f"prediction index / length: {predict_index} / {len(predictions)}")
                    import sys
                    sys.exit(0)
                try:
                    entities, boxes = decode_bbox_from_caption(prediction, img_width, img_height, verbose=verbose)
                    assert len(entities) == len(boxes)
                except ValueError as e:
                    entities, boxes = [], []

                predict_boxes = []

                for (entity, box) in zip(entities, boxes):
                    if not are_phrases_similar(entity, phrase): # get the matched noun phrase
                        # print(f"{entity} | {phrase}")
                        continue
                    else:
                        predict_boxes.append(box)

                if len(predict_boxes) == 0:
                    print(f"Can't find valid bbox for the given phrase ({phrase}) in caption ({caption}), \n{prediction}")
                    print(f"We set a 0-area box to calculate recall result")
                    predict_boxes = [[0., 0., 0., 0.]]
                    # exit(0)
                
                # evaluate
                target_boxes = self.imgid2boxes[str(item['original_img_id'])][phrase_from_self["phrase_id"]]
                ious = box_iou(np.asarray(predict_boxes), np.asarray(target_boxes))
                for k in self.topk:
                    maxi = 0
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thresh:
                        recall_tracker.add_positive(k, "all")
                        for phrase_type in phrase_from_self["phrase_type"]:
                            recall_tracker.add_positive(k, phrase_type)
                    else:
                        recall_tracker.add_negative(k, "all")
                        for phrase_type in phrase_from_self["phrase_type"]:
                            recall_tracker.add_negative(k, phrase_type)
                            
                # pdb.set_trace()
                valid_cnt += 1
            predict_index += 1
  
        print(f"Valid prediction {valid_cnt}/{len(predictions)}")     
        self.results = recall_tracker.report()
        return self.results
    
    def summarize(self):
        table = PrettyTable()
        all_cat = sorted(list(self.results.values())[0].keys())
        table.field_names = ["Recall@k"] + all_cat

        score = {}
        for k, v in self.results.items():
            cur_results = [v[cat] for cat in all_cat]
            header = "Upper_bound" if k == -1 else f"Recall@{k}"

            for cat in all_cat:
                score[f"{header}_{cat}"] = v[cat]
            table.add_row([header] + cur_results)

        print(table)
        return score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/final_flickr_mergedGT_test.json', help='annotation_file')
    parser.add_argument('--flickr_entities_path', default='/path/to/flickr30k_entities', help='flickr entities')
    
    args = parser.parse_args()

    if os.path.isfile(args.prediction_file):
        predictions = [json.loads(line) for line in open(args.prediction_file)]
    elif os.path.isdir(args.prediction_file):
        predictions = []
    
    if '_test.json' in args.annotation_file:
        subset = "test"
    else:
        subset = "val"
    
    evaluator = Flickr30kEntitiesRecallEvaluatorFromJsonl(
        flickr_path = args.flickr_entities_path,
        subset = subset,
        topk = (1, 5, 10, -1),
        iou_thresh = 0.5,
        merge_boxes = True,
        verbose = True,
    )
    
    evaluator.evaluate(args.annotation_file, args.prediction_file, verbose=False)
    score = evaluator.summarize()
    
    with open(os.path.join(args.prediction_file, "metric.json"), "w") as f:
        json.dump(score, f, indent=2)