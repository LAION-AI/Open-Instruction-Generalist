#@title flanv2_cot
"""
Copyright 2023, LAION contributors, inclduing Ontocord, LLC
and the other authors of OIG
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License. 
"""
import glob
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
import random, os
import spacy
import json
from basic_augment import *
try:
  nlp = spacy.load("en_core_web_sm")
except:
  os.system("python -m spacy download en_core_web_sm")  
  nlp = spacy.load("en_core_web_sm")
if not os.path.exists("FLAN"):
  os.system("git clone https://github.com/google-research/FLAN")

def flanv2_cot(output):
  for file in glob.glob("./FLAN/flan/v2/cot_data/*.tsv"):
    dataset_name = file.split("/")[-1].split(".")[0]
    with open(file, "rb") as in_data:
      for l in in_data:
        l = l.decode()
        l = l.strip()
        question, final_answer, answer  = l.split("\t")
        question = question.replace("\\n", "\n")
        if "Premise:" in question and random.randint(0,1)==0:
          question = question.replace("Premise:", random.choice(["", "You see the following:", "Given this that", "Assume:"]))
        if "Hypothesis:" in question and random.randint(0,1)==0:   
          question = question.replace("Hypothesis:", random.choice(["=>", "We infer:", "We generalize", "A potential conclusion:"]))
          
        question2 = question.split("?")[0].split(".")[-1].strip(" ?")

        if "following" in question2 or "sentence" in question2 or "one is it" in question2:
          question2 = ""
        elif question2 and question2.split()[0].lower() in {"who", "what", "when", "where", "how", "which"}:
          orig_q2 = question2
          if "How much" in question2 or "How many" in question2 or "How might" in question2:
            question2 = ""
          else:
            start2 = orig_q2.split()[0].lower()
            question2 = " ".join(question2.split()[1:])
            if question2.startswith("might"):
              question2= question2.replace("might ", "is what ",1)
            elif question2.startswith("does"):
              question2 = question2.replace("does ", "is "+ start2+ " ")
            elif question2.startswith("do"):
              question2 = question2.replace("do ", "is "+ start2+ " ")          
            #print ('**', question2)
        elif question2 and question2.split()[0].lower() in { "is", "can", "do",}:
          
          doc = nlp(question2)
          np = [a.text for a in doc.noun_chunks if len(a.text) > 3]
          if not np or question2.index(np[0]) > 5:
            question2 = ""
          else:
            np = np[0]
            start2 = orig_q2.split()[0].lower()
            start = question2.split()[0]
            if final_answer in {"No", "no"}:
              question2 = np + " " + start.lower() + " not " + question2.split(np,1)[1]
            else:
              question2 = np + " " + start.lower() + " " + question2.split(np,1)[1]
        answer = answer.replace("\\n", "\n").replace("..", ".").replace(" .", ".")
        question = question.replace(" .", ".").strip()
        question = question[0].upper()+question[1:]
        answer = answer[0].upper()+answer[1:]
        final_answer = final_answer.replace(" .", ".")
        if final_answer:
          final_answer = final_answer[0].upper()+final_answer[1:]
        if "\n" in answer and answer.count("\n") > 1 and random.randint(0,1) == 0:
          text = f"User: {question}\nAssistant: {final_answer} {question2}.\nUser: "+random.choice(["Please also explain your answer.", "And why?", "Take me through your reasoning.", "Explain.", "Can you tell me your reasoning?"])+f"\nAssistant: {answer}"
          text = text.replace("..", ".").replace("?.", "?")
          text = basic_augment(text)
          output.write(json.dumps({'text': text, "metadata": {'source': 'flanv2_cot_'+dataset_name}})+"\n")
        elif "\n" in question:
          if "\n" in answer:
            answer = answer.strip()
            answer = answer[0].lower() + answer[1:]
            text =  f"User: {question}\n"+random.choice(["Can you also walk me through your reasoning", "Plus step-by-step reasons.", "Let's solve this step by step."])+f"\nAssistant: {final_answer} because {answer}.\nThus {final_answer} {question2}."
            text = text.replace("..", ".").replace("?.", "?")
            text = basic_augment(text)
            output.write(json.dumps({'text': text, "metadata": {'source': 'flanv2_cot_'+dataset_name}})+"\n")
          else:
            answer = answer.strip()
            answer = answer[0].lower() + answer[1:]
            text = f"User: {question}\n"+random.choice(["", "Plus step-by-step reasons.", "And why?", "Let's solve this step by step."])+f"\nAssistant: {final_answer} because {answer}."
            text = text.replace("..", ".").replace("?.", "?")
            text = basic_augment(text)
            output.write(json.dumps({'text': text, "metadata": {'source': 'flanv2_cot_'+dataset_name}})+"\n")
        else:
          answer = answer.strip()
          answer = answer[0].lower() + answer[1:]
          text = f"User: {question} "+random.choice(["", "Please also explain your answer.", "And why?", "Take me through your reasoning.", "Explain."])+f"\nAssistant: {final_answer} because {answer}."
          text = text.replace("..", ".").replace("?.", "?")
          text = basic_augment(text)
          output.write(json.dumps({'text':text, "metadata": {'source': 'flanv2_cot_'+dataset_name}})+"\n")
