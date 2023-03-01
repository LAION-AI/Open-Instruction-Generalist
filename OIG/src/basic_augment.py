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
import gzip, json, os
from collections import Counter
import random
import re
from collections import Counter
try:
  from nltk.corpus import stopwords as nltk_stopwords
  nltk_stopwords.words() 
except:
  os.system("python -m nltk.downloader stopwords")
from nltk.corpus import stopwords as nltk_stopwords
import re, random

import itertools

stop_words = set(nltk_stopwords.words() + list({'"', "'", ".", "!", "?", ":", "continued", "developed", "continuing", "developing", "user", "search", "online", "contact", "nonetheless", "rights", "reserved", "called", "that's", "you're", "feed", "shared", "please", "thank", "thanks", "writing", "mr", "mrs", "miss", "named", "naming", "followed", "following", "everything", "nothing", "something", "saying", "illustrating", "describing", "described", "illustrated", "happened", "indeed", "someone", "searching", "regarding", "having", "getting", "according", "including", "included", "consisting", "consisted", "comprising", "comprised", "there's", "he's", "can't", "would", "however", "don't", "i've", "let's", "i'm", "i'll", "i'd", "it's", 'he', 'she', 'they', 'it', 'me', 'i', 'good', 'will', 'why', 'talk', 'long', 'above', 'looks', 'face', 'men', 'years', 'can', 'both', 'have', 'keep', 'yeah', 'said', 'bring', 'done', 'was', 'when', 'ask', 'now', 'very', 'kind', 'they', 'told', 'tell', \
              'ever', 'kill', 'hold', 'that', 'below', 'bit', 'knew', 'haven', 'few', 'place', 'could', 'says', 'huh', 'job', 'also', 'ain', 'may', 'heart', 'boy', 'with', 'over', 'son', 'else', 'found', 'see', 'any', 'phone', 'hasn', 'saw', 'these', 'maybe', 'into', 'thing', 'mom', 'god', 'old', 'aren', 'mustn', 'out', 'about', 'guy', 'each', 'most', 'like', 'then', 'wasn', 'being', 'all', 'door', 'look', 'run', 'sorry', 'again', 'won', 'man', 'gone', 'them', 'ago', 'doesn', 'gonna', 'girl', 'feel', 'work', 'much', 'hope', 'never', 'woman', 'went', 'lot', 'what', 'start', 'only', 'play', 'too', 'dad', 'going', 'yours', 'wrong', 'fine', 'made', 'one', 'want', 'isn', 'our', 'true', 'room', 'wanna', 'are', 'idea', 'sure', 'find', 'same', 'doing', 'off', 'put', 'turn', 'come', 'house', 'think', 'meet', 'hers', 'gotta', 'nor', 'away', 'leave', 'car', 'used', 'happy', 'the', 'care', 'seen', 'she', 'not', 'were', 'ours', 'their', 'first', 'world', 'lost', 'make', 'big', 'left', 'miss', 'shan', 'did', 'thank', 'ready', 'those', 'give', 'next', 'came', 'who', 'mind', 'does', 'right', 'her', 'let', 'didn', 'open', 'has', 'show', 'wife', 'yet', 'got', 'know', 'whole', 'some', 'such', 'alone', 'baby', 'him', 'nice', 'bad', 'move', 'new', 'dead', 'three', 'weren', 'whom', 'well', 'get', 'which', 'end', 'you', 'than', 'while', 'last', 'once', 'sir', 'from', 'need', 'wait', 'days', 'how', 'don', 'heard', 'own', 'hear', 'where', 'hey', 'okay', 'just', 'until', 'your', 'there', 'this', 'more', 'been', 'his', 'under', 'mean', 'might', 'here', 'its', 'but', 'stay', 'yes', 'guess', \
              'even', 'guys', 'hard', 'hadn', 'live', 'stop', 'took', 'still', 'other', 'since', 'every', 'needn', 'way', 'name', 'two', 'back', 'and', 'hello', 'head', 'use', 'must', 'for', 'life', 'die', 'day', 'down', 'wants', 'after', 'say', 'try', 'had', 'night'}))

transition_prompts = {
  "generalization of": [
    "generally",
    "as a generalization", 
    "in general",              
  ],
  "explanation of":[
    "as an explanation",
    "the reason was",
    "it was done as follows",
    "explained step by step"
  ],
  "clarification of": [
    "not only this", 
    "indeed", 
    "further", 
    "as well as this", 
    "but also that", 
    "as well", 
    "as a matter of fact",
    "what is more", 
    "in addition to this", 
    "to to tell you the truth",
    "furthermore", 
    "besides this", 
    "in fact", 
    "actually", 
    "to say nothing of", 
    "let alone", 
    "much less", 
    "additionally", 
    "not to mention", 
    "that is to say",
    "namely", 
    "specifically",
    "i mean to say", 
    "to put it another way",
    "in other words",
  ],
  "example of": [
    "such as", 
    "particularly", 
    "including",
    "as an illustration",
    "for example",
    "in particular", 
    "for one thing", 
    "to illustrate",
    "for instance", 
    "especially", 
    "notably",
    "by way of example",
  ],
  "reference for": [
    "speaking about this",
    "considering this",
    "regarding this",
    "with regards to this",
    "as for this",
    "concerning this",
    "on the subject of this",
    "the fact that", 
  ],
  "similarity to": [
    "similarly",
    "in the same way",
    "in ta similar way",
    "by the same token", 
    "in a like manner",
    "in a similar manner",
    "equally likewise",
  ],
  "contrast to": [
    "by way of contrast",
    "on the other hand",
    "however", 
    "and yet", 
    "whereas", 
    "though",
    "alternatively", 
  ],
  "emphasis of": [
    "even more", 
    "above all", 
    "indeed", 
    "more importantly",
    "besides"
  ],
  "concession to": [
    "but even so", 
    "nevertheless", 
    "even though",
    "on the other hand",
    "admittedly",
    "however", 
    "nonetheless", 
    "despite this",
    "notwithstanding this",
    "albeit",
    "and still", 
    "in spite of this",
    "regardless of this",
    "and yet", 
    "though",
    "granted this",
    "be that as it may",
  ],
  "dismissal of": [
      "either way",
      "whichever happens",
      "in either event",
      "in any case",
      "at any rate",
      "in either case",
      "whatever happens",
      "all the same",
      "in any event",
  ],
  "alternative to": [
    "or at least", 
    "or rather", 
    "instead"
  ],
  "cause of or reason for": [ 
    "for the simple reason that", 
    "being that", 
    "in view of the fact", 
    "inasmuch as", 
    "because",
    "because of the fact", 
    "seeing that", 
    "owing to the fact", 
    "due to the fact that", 
    "in that since", 
    "forasmuch as",
  ],
  "condition for": [ 
    "on the condition",
    "granted that", 
    "provided that", 
    "in case", 
    "in the event that", 
    "so long as", 
    "as long as", 
    "unless given that," 
    "granting that", 
    "providing that", 
    "even if", 
    "only if",
    ], 
  "result of or consequence of": [
    "as a result of this", 
    "consequently", 
    "hence", 
    "then",
    "for this reason", 
    "thus", 
    "because of this", 
    "in consequence", 
    "so that", 
    "accordingly", 
    "as a consequence", 
    "so much so that", 
    "therefore",     
    "under those circumstances", 
    "in that case",
    "if not",
    "that being the case", 
    "if so", 
    "otherwise", 
    "=>"
  ],
  "purpose of":[ 
    "for the purpose of", 
    "in the hope that", 
    "for fear that", 
    "so that", 
    "with this intention", 
    "to the end that", 
    "in order to", 
    "lest", 
    "with this in mind", 
    "in order that", 
    "so as to", 
  ],
  "ordering of": [ 
    "in the first place"
    "in the second place", 
    "initially", 
    "to start with", 
    "first of all", 
    "thirdly", 
    "to begin with", 
    "at first", 
    "for a start", 
    "secondly",
  ],
  "continuation of": [ 
    "subsequently", 
    "previously", 
    "eventually", 
    "next up", 
    "before this", 
    "afterwards", 
    "after this", 
    "then afterwards", 
    "after this", 
    "then",
    ],
  "conclusion to": [
    "consequently",
    "given these points",
    "therefore",
    "hence", 
    "in conclusion", 
    "in a word",
    "to conclude with", 
    "as a final point", 
    "eventually", 
    "at last",
    "last but not least", 
    "in the end", 
    "finally", 
    "lastly",
  ],
  "digression from": [
    "to change the topic", 
    "incidentally", 
    "by the way",
  ],
  "resumption of": [
      "to get back to the point",
      "to resume", 
      "anyhow", 
      "anyway", 
      "at any rate",
      "to return to the subject",
  ],
  "summary of": [
    "as was previously stated",
    "in summary",
    "all in all",
    "to make a long story",
    "in short",
    "as I have said",
    "to sum up", 
    "overall",
    "as has been mentioned",
    "to summarize",
    "to be brief", 
    "briefly",
    "in all", 
    "on the whole", 
    "as has been noted", 
    "to put it briefly", 
    "in sum", 
    "altogether", 
    "in short",
  ],  
  "emotional response to":[
    "they felt",
    "it felt",
    "i felt",
    "they wanted to because", 
    "their motivation being", 
    "their intention being",
    "their desire being",
    "what they wanted was"
   ],
  "positive emotional response to":[
    "positively",
    "amazingly",
    "wonderfuly",
    "with great joy",
    "gladly",
    "happily",
   ],
   "negative emotional response to":[
    "negatively",
    "awfully",
    "unsuprisngly",
    "with great sadness",
    "with hesitation",
    "unhappily",
   ],
}
transition_prompt_inv = dict(itertools.chain(*[[(c,a) for c in b] for a, b in transition_prompts.items()]))
 
styles = {'NA':'narrative',
 'IN': 'informational description',
 'OP':'opinion',
 'ID':'interactive discussion',
 'HI':'instruction',
 'IP':'sales and marketing',
 'LY':'lyrical',
 'SP':'spoken',}

text_type = {'NA':'article',
 'IN': 'description',
 'OP':'opinion',
 'ID':'interview',
 'HI':'instruction',
 'IP':'sales description',
 'LY':'poetic text',
 'SP':'spoken transcript',}

text_type2 = {'NA':'discussion',
 'IN': 'description',
 'OP':'opinion',
 'ID':'news article',
 'HI':'instruction',
 'IP':'sales pitch',
 'LY':'lyrical text',
 'SP':'transcript',}


#TODO: convert websites to their descriptive forms
#add algorithmic stuff like summarize, NER and translation
#basic math stuff like counting number of people, and the number of places and asking how many more people than places
#convert words lower in the wordnet hiearchy - to higher in the hiearchy
#this creates lower resolution, but protects more the creative expressions of authors.
#same for do name changing and gender swapping
#create commands to change formatting - numbered paragraphs, lists, tables, json, etc.
#create commands to force bot to attend to beginning of dialog or earlier in dialog
#create commands to force forgetting context and rememering context
#perturb answers to create incorrect answers and add <notrain> </notrain> tags and fixing of an answer

#UL2 stuff

R_DENOISING = 0
S_DENOISING = 1
X1_DENOISING = 2
X2_DENOISING = 3

def cjk_detect(texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    #thai
    if re.search("[\u0E01-\u0E5B]", texts):
        return "th"
    return None


def get_main_non_stopwords(text, top_n=6):
  text_arr = [s.strip("~!@#$%^&*()_-+={}:;\"'1234567890<>,.?/") for s in text.split() if s[0] == s[0].upper() and s[0].lower()  in "qwertyuiopasdfghjklzxcvbnm"]
  verb_arr = [s for s in text.split() if s.endswith("ing") or s.endswith("ed")]
  words = list(set([a[0] for a in Counter([s for s in text_arr if len(s) > 2 and s.lower() not in stop_words]).most_common(int(top_n/2))] + 
      [a[0] for a in Counter([s for s in verb_arr if len(s) > 2 and s.lower() not in stop_words]).most_common(int(top_n/2))]))
  words_position = [(word, text.index(word)) for word in words]
  words_position.sort(key = lambda a: a[1])
  return [a[0] for a in words_position]


def get_metadata(text, labels):
  num_sent =  len(re.split(r'[.!?]+', text) )
  subject = get_main_non_stopwords(text)
  is_nsfw = False
  text_lower = text.lower()
  if " xxx " in text_lower or " porn " in text_lower or " bdsm " in text_lower or " anal " in text_lower or " pussy " in text_lower or\
    " fuck " in text_lower or " cunt " in text_lower or " dildo " in text_lower or " cock " in text_lower or \
    " horny " in text_lower or " cock " in text_lower:
    is_nsfw = True
  subject = ", ".join(subject)
  if not labels:
    return (subject, "NSFW" if is_nsfw else "", "NSFW" if is_nsfw else "", num_sent)
  style = labels[0] #TODO - do mixed styles
  if random.randint(0,1)== 0:
    a_type = text_type[style]
  else:
    a_type = text_type2[style]
  if styles[style] == "instruction" and "1." in text:
    style = "how-to"
    a_type = "how-to"
  else:
    style = styles[style]
  if is_nsfw:
    style = "NSFW " + style
    a_type = "NSFW " + a_type
  return (subject, a_type, style, num_sent)


#word repition ration
def number_words_ratio(text):
  return len(list(set(text.lower().split())))/len(text)


def basic_augment(dialog):
  new_dialog = []
  dialog = dialog.split("User:")
  for d in dialog:
    d = d.strip()
    if not d: continue
    d_arr =  d.split("\nAssistant:")
    if len(d_arr) <= 1:
      #print (d_arr)
      continue
    instruction, response =d_arr
    instruction, response = instruction.strip(), response.strip()
    response = response.lstrip("+]),.?/~!^&*) ")
    if response.count(".") >3 and ("The" in response or "the" in response or " an " in response  or " of " in response  or " can " in response  or " should " in response  or " on " in response):
      reponse = response[0].upper()+response[1:]
      response = response.split(".")
      if response[-1] == '':
        response = response[:-1]
        response[-1] =response[-1]+"."
      choice = random.randint(0,12)
      if choice == 0 and len(response[0].strip()) > 5:
        if random.randint(0,1) == 0 or len(response[0].split()) <=3:
          instruction = instruction + f". Start with the sentence '{response[0].strip()}'."
        else:
          words = response[0].strip().split()
          words = " ".join(words[:random.randint(3, min(len(words), 5))])
          instruction = instruction + f". Start with the words '{words}'."          
        new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
        continue
      elif choice == 1 and len(response[-1].strip()) > 5:
        if random.randint(0,1) == 0 or len(response[-1].split()) <=3:
          instruction = instruction + f". End with the sentence '{response[-1].strip()}'."
        else:
          words = response[-1].strip().split()
          words = " ".join(words[:random.randint(3, min(len(words), 5))])
          instruction = instruction + f". End with the words '{words}'."
        new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
        continue
      elif choice == 2:
        orig_response =response
        response_missing = response[int(len(response)/2)]
        ents = get_main_non_stopwords(response_missing)
        if  ents:
          ents = ", ".join(ents)
          response = [r for r in response if r != response_missing]
          new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
          new_dialog.append(f"User: Add another sentence about {ents}.\nAssistant: " + ". ".join(orig_response) + "\n")
          continue
      elif choice == 3:
          if "\n" not in response[0]:
            new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
            new_dialog.append("User: Reverse the sentences.\nAssistant: " + ".".join(reversed(response)) + "\n")
            continue
      elif choice == 4:
          if "\n" not in response[0]:
            new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
            tmp = response[-1].strip(".")
            response[-1] = " "+response[0]+". "
            response[0] = tmp
            new_dialog.append("User: Swap the first and the last sentence.\nAssistant: " + ".".join(response) + "\n") 
            continue         
      elif choice == 5:
          new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
          response = response[:-1]
          response[-1] = response[-1]+". "
          new_dialog.append("User: Delete the last sentence.\nAssistant: " + ".".join(response) + "\n")       
          continue
      elif choice == 6:
          if "\n" not in response[0]:
            new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
            response = response[1:]
            new_dialog.append("User: Delete the first sentence.\nAssistant: " + ".".join(response) + "\n")    
            continue  
    else:
      response = response.split(".")
    new_dialog.append("User: " + instruction + "\nAssistant: " + ".".join(response) + "\n")
  new_dialog = [d.replace("  ", " ").replace(" .", ".").replace("?.", "?").replace(".. ", ". ").replace(".'.", ".'").replace(".\".", ".\"").replace("..\n", ".\n")  for d in new_dialog]
  return "".join(new_dialog).strip()    
