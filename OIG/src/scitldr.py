from datasets import load_dataset
import csv
import json
import random
import tqdm
import os


def create_scitldr(output):
    
    os.system('git clone git@github.com:allenai/scitldr.git')
    
    data = load_dataset("json", data_files='scitldr/SciTLDR-Data/SciTLDR-A/train.jsonl')

    for item in tqdm.tqdm(data['train']):
        
        ##### instruction before
        
        text = ' '.join(item['source']).strip()
        summary = item['target'][0].strip()
        
        user_template = random.choice(user_templates)
        
        prompt = '<human>: ' + user_template
        prompt += '\n\n' + text
        
        prompt += '\n<bot>: ' + summary
        
        output.write(json.dumps({'text': prompt}) + '\n')
        
        ##### instruction after
        
        text = ' '.join(item['source']).strip()
        summary = item['target'][0].strip()
        
        user_template = random.choice(user_templates)
        
        prompt = '<human>: ' + text
        prompt += '\n\n' + user_template
        
        prompt += '\n<bot>: ' + summary
        
        output.write(json.dumps({'text': prompt}) + '\n')