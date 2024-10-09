#!/bin/bash
set -xe

model_path='/path_to_Ferret-UI_model'

# Example inference for referring tasks (ie bbox in input)
CUDA_VISIBLE_DEVICES=0 python -m ferretui.eval.model_UI \
    --model_path $model_path \
    --data_path ./playground/sample_data/eval_data_example_0_box_in.json \
    --image_path ./playground/images \
    --answers_file eval_output/data_box_in_eval.jsonl \
    --num_beam 1 \
    --max_new_tokens 32 \
    --region_format box \
    --add_region_feature \
    --conv_mode ferret_gemma_instruct # or ferret_llama_3 for Ferret-UI_llama8b model

# Example inference for non-referring tasks (ie no bbox in input)
CUDA_VISIBLE_DEVICES=0 python -m ferretui.eval.model_UI \
    --model_path $model_path \
    --data_path ./playground/sample_data/eval_data_example_1_no_box_in.json \
    --image_path ./playground/images \
    --answers_file eval_output/data_no_box_in_eval.jsonl \
    --num_beam 1 \
    --max_new_tokens 32 \
    --region_format box \
    --add_region_feature \
    --conv_mode ferret_gemma_instruct

