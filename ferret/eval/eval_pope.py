"""
Usage:

python ferret/eval/eval_pope.py \
    --prediction_file final_result/ferret_13b_checkpoint-final/pope_result/coco_pope_adversarial \
    --annotation_file data/pope/coco_pope_adversarial.json

python ferret/eval/eval_pope.py \
    --prediction_file final_result/ferret_13b_checkpoint-final/pope_result/coco_pope_popular \
    --annotation_file data/pope/coco_pope_popular.json

python ferret/eval/eval_pope.py \
    --prediction_file final_result/ferret_13b_checkpoint-final/pope_result/coco_pope_random \
    --annotation_file data/pope/coco_pope_random.json

"""
import os
import json

def evaluate_pope(prediction_file, annotation_file):
    # get the predictions
    if os.path.isfile(prediction_file):
        answers = [json.loads(line) for line in open(prediction_file)]
    elif os.path.isdir(prediction_file):
        answers = [json.loads(line) for pred_file in sorted(os.listdir(prediction_file)) for line in open(os.path.join(prediction_file, pred_file))]
    else:
        raise NotImplementedError('Not supported file format.')

    label_list = [json.loads(q)['label'] for q in open(annotation_file, 'r')]

    for answer in answers:
        text = answer['answer']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))

    score = {"Accuracy": acc, 
             "Precision": precision,
             "Recall": recall,
             "F1 score": f1,
             "Yes ratio": yes_ratio,
             }

    with open(os.path.join(args.prediction_file, 'metric.json'), "w") as f:
        json.dump(score, f, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/json_annotations', help='annotation_file')
    
    args = parser.parse_args()
    evaluate_pope(args.prediction_file, args.annotation_file)