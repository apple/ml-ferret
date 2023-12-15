"""
Usage:
# 7B
python3 -m ferret.model.make_delta \
    --base ./model/vicuna-7b-v1-3 \
    --target ./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --delta ./checkpoints/ferret_ft_clipL336_vicunaV1-3-7b_3Ep_dataV16_RSamplerV2/ferret-7b-delta

# 13B
python3 -m ferret.model.make_delta \
    --base ./model/vicuna-13b-v1-3 \
    --target ./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/checkpoint-final \
    --delta ./checkpoints/ferret_ft_clipL336_vicunaV1-3-13b_3Ep_dataV16_RSamplerV2/ferret-13b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ferret.model.utils import auto_upgrade

# all the parameters inside the geosampler and mm projector
exclude_name_lists = ['model.mm_projector.weight', 'model.mm_projector.bias', 
                    'model.region_geo_sampler.agg_projector_list.0.net.0.bias', 'model.region_geo_sampler.agg_projector_list.0.net.0.weight', 
                    'model.region_geo_sampler.agg_projector_list.0.norm.bias', 'model.region_geo_sampler.agg_projector_list.0.norm.weight', 
                    'model.region_geo_sampler.agg_projector_list.1.net.0.bias', 'model.region_geo_sampler.agg_projector_list.1.net.0.weight', 
                    'model.region_geo_sampler.agg_projector_list.1.norm.bias', 'model.region_geo_sampler.agg_projector_list.1.norm.weight', 
                    'model.region_geo_sampler.diff_projector_list.0.bias', 'model.region_geo_sampler.diff_projector_list.0.weight', 
                    'model.region_geo_sampler.diff_projector_list.1.bias', 'model.region_geo_sampler.diff_projector_list.1.weight', 
                    'model.region_geo_sampler.dim_projector.bias', 'model.region_geo_sampler.dim_projector.weight', 
                    'model.region_geo_sampler.flatten_projector.bias', 'model.region_geo_sampler.flatten_projector.weight'
                    ]


def make_delta(base_model_path, target_model_path, delta_path, hub_repo_id):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading target model")
    auto_upgrade(target_model_path)
    target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Calculating delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        if name not in base.state_dict():
            assert name in exclude_name_lists, f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            param.data -= base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] -= bparam

    print("Saving delta")
    if hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": hub_repo_id}
    else:
        kwargs = {}
    target.save_pretrained(delta_path, **kwargs)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.save_pretrained(delta_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, default=None)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path, args.hub_repo_id)
