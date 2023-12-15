"""
Usage:

# 7B
To extract region_geo_sampler:
python misc/extract_geosampler_and_mm_projector.py \
    --keys_to_match=region_geo_sampler \
    --model-path=./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --output=./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/extracted_region_geo_sampler.bin

To extract mm_projector:
python misc/extract_geosampler_and_mm_projector.py \
    --keys_to_match=mm_projector \
    --model-path=./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --output=./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/extracted_mm_projector.bin

# 13B
To extract region_geo_sampler:
python misc/extract_geosampler_and_mm_projector.py \
    --keys_to_match=region_geo_sampler \
    --model-path=./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --output=./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/extracted_region_geo_sampler.bin

To extract mm_projector:
python misc/extract_geosampler_and_mm_projector.py \
    --keys_to_match=mm_projector \
    --model-path=./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --output=./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/extracted_mm_projector.bin
"""


import os
import argparse
import torch
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector or GeoSampler weights')
    parser.add_argument('--model-path', type=str, help='model folder')
    parser.add_argument('--output', type=str, help='output file')
    parser.add_argument('--keys_to_match', type=str, default="region_geo_sampler", choices=["mm_projector", "region_geo_sampler"], help='keys to be matched')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    keys_to_match = [args.keys_to_match]
    ckpt_to_key = defaultdict(list)
    print('----indexing keys_to_match...----')
    try:
        model_indices = json.load(open(os.path.join(args.model_path, 'pytorch_model.bin.index.json')))
        for k, v in model_indices['weight_map'].items():
            if any(key_match in k for key_match in keys_to_match):
                ckpt_to_key[v].append(k)
    except FileNotFoundError:
        # Smaller models or model checkpoints saved by DeepSpeed.
        v = 'pytorch_model.bin'
        for k in torch.load(os.path.join(args.model_path, v), map_location='cpu').keys():
            if any(key_match in k for key_match in keys_to_match):
                ckpt_to_key[v].append(k)

    loaded_weights = {}

    print('----loading weights...----')
    for ckpt_name, weight_keys in ckpt_to_key.items():
        ckpt = torch.load(os.path.join(args.model_path, ckpt_name), map_location='cpu')
        for k in weight_keys:
            loaded_weights[k] = ckpt[k]

    print('----saving weights...----')
    print(f'the keys of saved weights: {loaded_weights.keys()}')
    print(f'----saved to {args.output}----')
    torch.save(loaded_weights, args.output)
