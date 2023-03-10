from datasets import load_dataset
import csv
import json
import random
import tqdm

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


def create_convfinqa(output):

    #######
    # download and unzip https://github.com/czyssrs/ConvFinQA/blob/main/data.zip
    #######
    
    with open('data/train.json') as f:
        data = json.load(f)
    
    for item in tqdm.tqdm(data):
        
        ### qa step by step

        table = '| ' + ' | '.join(item['table_ori'][0]) + ' |\n'
        if len(item['table_ori'][0]) == len(item['table_ori'][-1]):
            table += '| ' + ' | '.join(['---------' for x in item['table_ori'][-1]]) + ' |\n'
        for x in item['table_ori'][1:]:
            table += '| ' + ' | '.join(x) + ' |\n'

        pre_texts = []
        for pre_text in item['pre_text']:
            pre_text = detokenizer.detokenize(pre_text.split(' '))
            pre_texts.append(pre_text)
        pre_text = ' '.join(pre_texts)

        post_texts = []
        for post_text in item['post_text']:
            post_text = detokenizer.detokenize(post_text.split(' '))
            post_texts.append(post_text)
        post_text = ' '.join(post_texts)

        q = item['annotation']['dialogue_break_ori'][0]
        a = item['annotation']['turn_program_ori'][0]

        prompt = f'<human>: {pre_text}\n{table}{post_text}\nPlease answer the following questions with expressions if necessary.\n{q}\n<bot>: {a}'

        for i in range(1, len(item['annotation']['exe_ans_list'])):
            q = item['annotation']['dialogue_break_ori'][i]
            a = item['annotation']['turn_program_ori'][i]
            prompt += f'\n<human>: {q}\n<bot>: {a}'
        
        output.write(json.dumps({'text': prompt}) + '\n')
        
        ### answer the final result
        
        q = item['annotation']['dialogue_break_ori'][0]
        a = item['annotation']['exe_ans_list'][0]

        prompt = f'<human>: {pre_text}\n{table}{post_text}\nPlease compute the result.\n{q}\n<bot>: {a}'

        for i in range(1, len(item['annotation']['exe_ans_list'])):
            q = item['annotation']['dialogue_break_ori'][i]
            a = item['annotation']['exe_ans_list'][i]
            prompt += f'\n<human>: {q}\n<bot>: {a}'
        
        output.write(json.dumps({'text': prompt}) + '\n')
        
        ### answer the expression in one-step.
        
        i = len(item['annotation']['turn_program_ori']) - 1
        q = item['annotation']['dialogue_break_ori'][i]
        a = item['annotation']['turn_program_ori'][i]
        
        prompt = f'<human>: {pre_text}\n{table}{post_text}\nPlease answer the following question with expressions if necessary.\n{q}\n<bot>: {a}'

        output.write(json.dumps({'text': prompt}) + '\n')
        