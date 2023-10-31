import argparse
import json
import os

import openai
import time
import re
import pdb
from tqdm import tqdm

NUM_SECONDS_TO_SLEEP = 0.5
VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000

def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4-0314',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']

def postprocess_answer(answer, category):
    if category == 'refer_desc' or category == 'refer_reason':
        pattern = r'\[.*?\]'
        matches = re.findall(pattern, answer)
        for match in matches:
            answer = answer.replace(' '+match, '')
    elif category == 'ground_conv':
        pattern = r'\[.*?\]'
        matches = re.findall(pattern, answer)
        for match in matches:
            coor_cur = match.replace('[', '')
            coor_cur = coor_cur.replace(']', '')
            coor_cur = coor_cur.split(',')
            coor_cur = [float(i.strip()) for i in coor_cur]
            try:
                assert len(coor_cur) == 4 
            except:
                print('Found a exception when parsing coordinates')
                answer = answer.replace(match, '')
            converted_box_coor = [coor_cur[0]/VOCAB_IMAGE_W, coor_cur[1]/VOCAB_IMAGE_H, coor_cur[2]/VOCAB_IMAGE_W, coor_cur[3]/VOCAB_IMAGE_H]
            answer = answer.replace(match, f'[{converted_box_coor[0]:.3f}, {converted_box_coor[1]:.3f}, {converted_box_coor[2]:.3f}, {converted_box_coor[3]:.3f}]')

    return answer


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in tqdm(zip(f_q, f_ans1, f_ans2)):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]
        # cap_str = '\n'.join(inst['captions'])
        # box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

        category = json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        
        # Assume ans2 is the predicted one.
        processed_answer = postprocess_answer(ans2['text'], category)
        # pdb.set_trace()
        ans2['text'] = processed_answer
        # if category == 'refer_desc':
            
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\{inst["text"]}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        # content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
        #            f'[Question]\n{ques["text"]}\n\n'
        #            f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
        #            f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
        #            f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['question_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            cur_js['answer1'] = ans1["text"]
            cur_js['answer2'] = ans2["text"]
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
