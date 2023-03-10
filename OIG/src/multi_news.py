from datasets import load_dataset
import csv
import json
import random
import tqdm


def create_multi_news(output):

    data = load_dataset('multi_news')
    
    doc_names = [('document', 'Document'), ('passage', 'Passage'), ('text', 'Text'),]
    user_templates = [
    "Can you condense the documents into XXX words or less?",
    "Can you provide a brief rundown of the documents in XXX words?",
    "Could you distill the essence of the documents into XXX words?",
    "Can you succinctly summarize the documents in XXX words?",
    "Can you give a brief summary of the documents using only XXX words?",
    "Can you encapsulate the documents into XXX words or fewer?",
    "Could you articulate the main points of the documents in XXX words?",
    "Can you concisely highlight the essence of the documents in XXX words?",
    "Can you synthesize the documents into a XXX-word summary?",
    "Can you present a pithy summary of the documents in XXX words?",
    "Briefly summarize the documents in XXX words or less.",
    "Provide a succinct summary of the documents in no more than XXX words.",
    "Give a condensed overview of the documents in XXX words or fewer.",
    "Present a short summary of the documents using no more than XXX words.",
    "In XXX words or less, give a brief synopsis of the documents.",
    "Summarize the contents of the documents in no more than XXX words.",
    "Give a summary of the documents in a maximum of XXX words.",
    "Present an abridged version of the documents using XXX words or fewer.",
    "Summarize the information in the documents using a maximum of XXX words.",
    "In a nutshell, provide a summary of the documents in XXX words or less.",
    "Can you condense the documents?",
    "Can you provide a brief rundown of the documents?",
    "Could you distill the essence of the documents?",
    "Can you succinctly summarize the documents?",
    "Can you give a brief summary of the documents?",
    "Can you encapsulate the documents?",
    "Could you articulate the main points of the documents?",
    "Can you concisely highlight the essence of the documents?",
    "Can you synthesize the documents into a summary?",
    "Can you present a pithy summary of the documents?",
    "Briefly summarize the documents",
    "Provide a succinct summary of the documents.",
    "Give a condensed overview of the documents.",
    "Present a short summary of the documents.",
    "Give a brief synopsis of the documents.",
    "Summarize the contents of the documents.",
    "Give a summary of the documents.",
    "Present an abridged version of the documents.",
    "Summarize the information in the documents.",
    "In a nutshell, provide a summary of the documents.",
    ]
    
    for item in tqdm.tqdm(data['train']):
        
        ##### instruction after
        
        documents = [doc.strip() for doc in item['document'].split('|||||')]
        summary = item['summary'].lstrip('– ').strip()
        summary_n_words = (len(summary.split()) // 10 + 1) * 10

        template = random.choice(user_templates)
        doc_type, Doc_type = random.choice(doc_names)
        template = template.replace('document', doc_type)
        template = template.replace('XXX', str(summary_n_words))
        
        prompt = '<human>: '
        for i, doc in enumerate(documents):
            doc = doc.replace('\n ', '\n')
            prompt += f"{Doc_type} {i+1}: " + doc + '\n'
        
        prompt += template
        
        prompt += '\n<bot>: ' + summary
        
        output.write(json.dumps({'text': prompt}) + '\n')
        
        ##### instruction before
        documents = [doc.strip() for doc in item['document'].split('|||||')]
        summary = item['summary'].lstrip('– ').strip()
        summary_n_words = (len(summary.split()) // 10 + 1) * 10

        template = random.choice(user_templates)
        doc_type, Doc_type = random.choice(doc_names)
        template = template.replace('document', doc_type)
        template = template.replace('XXX', str(summary_n_words))
        
        prompt = '<human>: '
        
        prompt += template
        
        for i, doc in enumerate(documents):
            doc = doc.replace('\n ', '\n')
            prompt += f"\n{Doc_type} {i+1}: " + doc
        
        prompt += '\n<bot>: ' + summary
        
        output.write(json.dumps({'text': prompt}) + '\n')
        