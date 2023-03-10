from datasets import load_dataset
import csv
import json
import random
import tqdm

def create_ni(output):

    data = load_dataset('Muennighoff/natural-instructions')

    last_task_def = None
    last_question = None
    prompt = ''

    for item in tqdm.tqdm(data['train']):

        task_def = item['definition']
        question = item['inputs']

        if question == last_question:
            continue

        last_question = question

        answer = item['targets']

        # Do a cut every 20 examples.
        if task_def == last_task_def and len(prompt.split('<bot>')) < 20:
            
            prompt += f"\n<human>: {question}\n<bot>: {answer}"
            
        else:
            if last_task_def is not None:
                
                output.write(json.dumps({'text': prompt}) + '\n')
                
            last_task_def = task_def
            prompt = f"<human>: {task_def}\n\n{question}\n<bot>: {answer}"

    output.write(json.dumps({'text': prompt}) + '\n')