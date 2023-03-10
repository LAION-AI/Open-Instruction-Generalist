from datasets import load_dataset
import csv
import json
import random
import tqdm


def create_squad_v2(output):

    data = load_dataset('squad_v2')
    
    last = None
    prompt = ''
    
    for item in data['train']:
        c = item['context']
        q = item['question']
        if len(item['answers']['text']) > 0:
            a = item['answers']['text'][0]
            has_answer = True
        else:
            a = "Sorry, I cannot find a relevant answer in the given context."
            has_answer = False
            
        if c == last:
            prompt += f"\n<human>: {q}\n<bot>: {a}"
        else:
            if last is not None:
                output.write(json.dumps({'text': prompt}) + '\n')
                
            last = c
            prompt = f"<human>: {c}\n\n{q}\n<bot>: {a}"
            
            if not has_answer:
                output.write(json.dumps({'text': prompt}) + '\n')
                last = None
                continue
                
    output.write(json.dumps({'text': prompt}) + '\n')