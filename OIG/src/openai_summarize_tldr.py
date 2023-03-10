from datasets import load_dataset
import csv
import json
import random
import tqdm


def create_openai_summarize_tldr(output):

    data = load_dataset('CarperAI/openai_summarize_tldr')
    
    user_templates = [
    "Can you condense the text into three sentences or less?",
    "Can you provide a brief rundown of the text in within 3 sentences?",
    "Could you distill the essence of the text into 1-3 sentences?",
    "Can you succinctly summarize the text in 1-3 sentences?",
    "Can you summarize the given text in a few sentences?",
    "Can you provide a brief overview of the text?",
    "Can you give me the gist of the text in a nutshell?",
    "Can you condense the text into a few key points?",
    "Can you give me a shortened version of the text?",
    "Summarize the given text in a few sentences.",
    "Provide a brief overview of the text.",
    "Give me the gist of the text in a nutshell",
    "Condense the text into a few key points",
    "Give me a shortened version of the text",
    ]
    
    for item in tqdm.tqdm(data['train']):
        
        #### instruction before
        
        text = item['prompt'][item['prompt'].find('\nPOST')+6:].replace('\nTL;DR:', '').strip()
        summary = item['label']
        
        user_template = random.choice(user_templates)
        
        prompt = '<human>: ' + user_template
        prompt += '\n\n' + text
        
        prompt += '\n<bot>: ' + summary
        
        output.write(json.dumps({'text': prompt}) + '\n')
        
        ##### instruction after
        
        text = item['prompt'][item['prompt'].find('\nPOST')+6:].replace('\nTL;DR:', '').strip()
        summary = item['label']
        
        user_template = random.choice(user_templates)
        
        prompt = '<human>: ' + text
        prompt += '\n\n' + user_template
        
        prompt += '\n<bot>: ' + summary
        
        output.write(json.dumps({'text': prompt}) + '\n')