#!/bin/bash
 
CHECKPOINT_FILE='ferret_ft/final-checkpoint'

CUDA_VISIBLE_DEVICES=0 python -m ferret.eval.model_gpt4eval_3newclass --add_region_feature \
    --model-path checkpoints/${CHECKPOINT_FILE} \
    --data_path ferret/eval/ferret_gpt4_data/refer_desc/question.jsonl \
    --answers-file gpt4_result/${CHECKPOINT_FILE}/refer_desc &
CUDA_VISIBLE_DEVICES=1 python -m ferret.eval.model_gpt4eval_3newclass --add_region_feature \
    --model-path checkpoints/${CHECKPOINT_FILE} \
    --data_path ferret/eval/ferret_gpt4_data/ground_conv/question.jsonl \
    --answers-file gpt4_result/${CHECKPOINT_FILE}/ground_conv &
CUDA_VISIBLE_DEVICES=2 python -m ferret.eval.model_gpt4eval_3newclass --add_region_feature \
    --model-path checkpoints/${CHECKPOINT_FILE} \
    --data_path ferret/eval/ferret_gpt4_data/refer_reason/question.jsonl \
    --answers-file gpt4_result/${CHECKPOINT_FILE}/refer_reason &

wait
echo "Finish Inference."

OPENAI_API_KEY="xxx" python ferret/eval/eval_gpt_review_3newclass.py \
    --question ferret/eval/ferret_gpt4_data/refer_desc/question.jsonl \
    --context ferret/eval/ferret_gpt4_data/refer_desc/context.jsonl \
    --answer-list \
    ferret/eval/ferret_gpt4_data/refer_desc/answer.jsonl \
    gpt4_result/${CHECKPOINT_FILE}/refer_desc/ferret_answer.jsonl \
    --rule ferret/eval/ferret_gpt4_data/rule.json \
    --output gpt4_result/${CHECKPOINT_FILE}/review_refer_desc.jsonl &
OPENAI_API_KEY="xxx" python ferret/eval/eval_gpt_review_3newclass.py \
    --question ferret/eval/ferret_gpt4_data/ground_conv/question.jsonl \
    --context ferret/eval/ferret_gpt4_data/ground_conv/context.jsonl \
    --answer-list \
    ferret/eval/ferret_gpt4_data/ground_conv/answer.jsonl \
    gpt4_result/${CHECKPOINT_FILE}/ground_conv/ferret_answer.jsonl \
    --rule ferret/eval/ferret_gpt4_data/rule.json \
    --output gpt4_result/${CHECKPOINT_FILE}/review_ground_conv.jsonl &
OPENAI_API_KEY="xxx" python ferret/eval/eval_gpt_review_3newclass.py \
    --question ferret/eval/ferret_gpt4_data/refer_reason/question.jsonl \
    --context ferret/eval/ferret_gpt4_data/refer_reason/context.jsonl \
    --answer-list \
    ferret/eval/ferret_gpt4_data/refer_reason/answer.jsonl \
    gpt4_result/${CHECKPOINT_FILE}/refer_reason/ferret_answer.jsonl \
    --rule ferret/eval/ferret_gpt4_data/rule.json \
    --output gpt4_result/${CHECKPOINT_FILE}/review_refer_reason.jsonl &

wait
echo "Finish Review."

echo "Gather final score."
echo $CHECKPOINT_FILE
python ferret/eval/summarize_gpt_review.py  \
    --dir=gpt4_result/${CHECKPOINT_FILE}