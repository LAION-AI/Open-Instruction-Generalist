from datasets import load_dataset
import csv
import json
import random
import tqdm

import re

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


def create_nq(output):
    
    #####
    # download and unzip https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz
    #####
    
    mores = [
    "Could you point me to the text mentioning this?",
    "Which text refers to this?",
    "Can you highlight the sentence that discusses this?",
    "Where in the text can I find information about this?",
    "What sentence talks about this?",
    "Could you direct me to the sentence that pertains to this?",
    "Which sentence addresses this topic?",
    "Can you locate the sentence that covers this?",
    "Where in the passage is the sentence discussing this?",
    "What is the sentence that relates to this?",
    ]

    with open("v1.0-simplified_simplified-nq-train.jsonl") as f:
        for i, line in enumerate(f):
            if line.strip() == '':
                continue
            item = json.loads(line)

            # doc = detokenizer.detokenize(item['document_text'].split(' '))
            c_id = item['annotations'][0]['long_answer']['candidate_index']
            if c_id < 0:
                # no answer
                c_id = random.randint(0, len(item['long_answer_candidates']))
            full_doc = item['document_text'].split(' ')
            doc = []
            for cand in item['long_answer_candidates'][max(c_id-2, 0): c_id+2]:
                doc += full_doc[cand['start_token']: cand['end_token']]
            doc = detokenizer.detokenize(doc)

            if doc == '':
                print(doc, c_id)
                break

            q = item['question_text'].capitalize()

            if item['annotations'][0]['yes_no_answer'] == 'NONE' and item['annotations'][0]['long_answer']['start_token'] == -1 and len(item['annotations'][0]['short_answers']) == 0:

                prompt = f"<human>: {doc}\n\n{q}\n<bot>: Sorry, I cannot find a relevant answer in the given context."

            elif len(item['annotations'][0]['short_answers']) > 0:
                # short answer

                short_a = detokenizer.detokenize(item['document_text'].split(' ')[
                    item['annotations'][0]['short_answers'][0]['start_token']: item['annotations'][0]['short_answers'][0]['end_token']
                ])

                long_a = detokenizer.detokenize(item['document_text'].split(' ')[
                    item['annotations'][0]['long_answer']['start_token']: item['annotations'][0]['long_answer']['end_token']
                ])
                long_a = striphtml(long_a).strip()
                while long_a != long_a.replace('  ', ' '):
                    long_a = long_a.replace('  ', ' ')
                long_a = detokenizer.detokenize(long_a.split(' '))

                more = random.choice(mores)
                prompt = f"<human>: {doc}\n\n{q}\n<bot>: {short_a}\n<human>: {more}\n<bot>: {long_a}"

            elif item['annotations'][0]['yes_no_answer'] != 'NONE':

                # short answer
                if item['annotations'][0]['yes_no_answer'] == 'NO':
                    short_a = 'No.'
                else:
                    short_a = 'Yes.'

                long_a = detokenizer.detokenize(item['document_text'].split(' ')[
                    item['annotations'][0]['long_answer']['start_token']: item['annotations'][0]['long_answer']['end_token']
                ])
                long_a = striphtml(long_a).strip()
                while long_a != long_a.replace('  ', ' '):
                    long_a = long_a.replace('  ', ' ')
                long_a = detokenizer.detokenize(long_a.split(' '))

                more = random.choice(mores)
                prompt = f"<human>: {doc}\n\n{q}\n<bot>: {short_a}\n<human>: {more}\n<bot>: {long_a}"

            else:

                long_a = detokenizer.detokenize(item['document_text'].split(' ')[
                    item['annotations'][0]['long_answer']['start_token']: item['annotations'][0]['long_answer']['end_token']
                ])
                long_a = striphtml(long_a).strip()
                while long_a != long_a.replace('  ', ' '):
                    long_a = long_a.replace('  ', ' ')
                long_a = detokenizer.detokenize(long_a.split(' '))

                prompt = f"<human>: {doc}\n\n{q}\nfind me the text answering this question\n<bot>: {long_a}"

            output.write(json.dumps({'text': prompt}) + '\n')
        