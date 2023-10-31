#!/usr/bin/env bash
set -xe

mkdir -p checkpoints

echo "Start Fine-Tuning"
# =================== Training ======================
data_path=(
            'dataset/git_instruction.json' 
            'dataset/vg_objects.json'  
            'dataset/vg_relations.json' 
            'dataset/vg_regions.json' 
            'dataset/grounded_llava_boxes_detail.json' 
            'dataset/grounded_llava_boxes_complex_reasoning.json' 
            'dataset/grounded_llava_boxes_conversation.json' 
            'dataset/refexp_all.json' 
            'dataset/flickr.json' 
            'dataset/objects365.json' 
            )
image_folder=(
            'dataset/coco2014/train2014' 
            'dataset/vg/images' 
            'dataset/vg/images' 
            'dataset/vg/images' 
            'dataset/coco2014/train2014' 
            'dataset/coco2014/train2014' 
            'dataset/coco2014/train2014' 
            'data/refcoco/train2014' 
            'data/flickr30k/flickr30k_images_split/train' 
            'data/objects365_v1/train' 
            )
data_multiple=(
            3 
            1 
            0.2 
            0.2 
            1 
            1 
            1 
            1 
            1 
            1 
            )

# convert array to string
data_path="${data_path[@]}"
image_folder="${image_folder[@]}"
data_multiple="${data_multiple[@]}"

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-13b-v1-3"
################## VICUNA ##################

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    ferret/train/train_mem.py \
    --lora_enable False \
    --model_name_or_path ./model/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path $data_path \
    --image_folder $image_folder \
    --data_multiple $data_multiple \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./model/llava-336px-pretrain-$MODEL_VERSION/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/ferret_13b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --point_input_sample 'segment_mask|center' \
    --add_region_feature True \
    --region_geo_sampler True \
    --sampler_pooler_mode 'max' \
    --add_region_feature True \
    --refer_previous_point False \
    --resized_image_h 336 \
    --resized_image_w 336 \
    --save_vision_tower True

