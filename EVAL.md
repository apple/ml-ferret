# Evaluation
All evaluation scripts provided usage details/cases in the first several lines of codes. 

## Ferret-Bench
Please follow [gpt4_eval_script.sh](ferret/eval/gpt4_eval_script.sh) to run inference on Ferret-Bench data and use GPT-4 to rate. It's noted that `openai` package should be installed and user's OPENAI_KEY should be provided.

## LVIS-Referring Object Classification
Run `ferret/eval/model_lvis.py` following the usage in the file and then run `ferret/eval/eval_lvis.py`.

## RefCOCO/RefCOCO+/RefCOCOg
Run `ferret/eval/model_refcoco.py` following the usage in the file and then run `ferret/eval/eval_refexp.py`.

## Flickr
Run `ferret/eval/model_flickr.py` following the usage in the file and then run `ferret/eval/eval_flickr_entities.py`.

## POPE
Run `ferret/eval/model_pope.py` following the usage in the file and then run `ferret/eval/eval_pope.py`.