from datasets import load_dataset
import csv
import json
import random
import tqdm

def create_p3(output):

    data = load_dataset('Muennighoff/P3')

    last = None
    prompt = ''
    
    for item in tqdm.tqdm(data['train']):

        chunks = item['inputs'].split('\n\n')

        c = '\n\n'.join(chunks[:-1])
        q = chunks[-1]

        a = item['targets']

        if c == last:
            prompt += f"\n<human>: {q}\n<bot>: {a}"
        else:
            if last is not None:
                
                output.write(json.dumps({'text': prompt}) + '\n')
                
            last = c
            prompt = f"<human>: {c}\n\n{q}\n<bot>: {a}"

    output.write(json.dumps({'text': prompt}) + '\n')