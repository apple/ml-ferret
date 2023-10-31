"""
Usage:
- Eval Prediction:
python ferret/eval/eval_lvis.py --pred_file=[your generated result by running ferret/eval/model_lvis.py]

"""
import argparse
import json
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from tqdm import tqdm
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default='/Users/youhaoxuan/research_misc/lvis_result/llava_answer_debug.jsonl')
    return parser.parse_args()

def remove_not_phrases_v2(text):
    # Pattern covers the start of a phrase up to and including 'not' and any following characters until a comma or period
    pattern = r"\s+not[^,.]*[,.]"
    text = re.sub(pattern, "", text)
    pattern = r"\s+no[^,.]*[,.]"
    text = re.sub(pattern, "", text)
    return text 

if __name__ == "__main__":
    args = get_args()
    # Fix the random seed
    random.seed(42)
    if os.path.isfile(args.pred_file):
        predictions = [json.loads(line) for line in open(args.pred_file)]
    elif os.path.isdir(args.pred_file):
        predictions = [json.loads(line) for pred_file in os.listdir(args.pred_file) for line in open(os.path.join(args.pred_file, pred_file))]
    else:
        raise NotImplementedError('Not supported file format.')

    total_correct = 0
    for i in tqdm(predictions):
        # Process name and synonyms
        i['name'] = i['name'].replace('_', ' ').strip()
        new_synonyms = []
        for jj in i['synonyms']:
            if '(' in jj:
                assert ')' in jj
                split_list = jj.split('(')
                assert len(split_list) == 2
                new_synonyms.append(split_list[0].replace('_', ' ').strip())
                new_synonyms.append(split_list[1].replace('_', ' ').replace(')', '').strip())
            else:
                new_synonyms.append(jj.replace('_', ' ').strip())
        i['synonyms'] = new_synonyms

        # Match Result
        processed_text = remove_not_phrases_v2(i['text'])
        # pdb.set_trace()
        if i['name'] in processed_text or any(syn_i in processed_text for syn_i in i['synonyms']):
            total_correct += 1
        else:
            pass

    acc = total_correct / len(predictions)
    print(f'Acc:{acc*100:.3f}%')
    # pdb.set_trace()
