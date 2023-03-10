from datasets import load_dataset
import csv
import json
import random
import tqdm

def create_flan(output):

    data = load_dataset('Muennighoff/flan')

    for item in tqdm.tqdm(data['train']):
        q = item['inputs']
        a = item['targets']
        prompt = f"<human>: {q}\n<bot>: {a}"
        output.write(json.dumps({'text': prompt}) + '\n')
