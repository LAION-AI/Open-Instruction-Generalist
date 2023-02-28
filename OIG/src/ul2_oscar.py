#@title UL2 and Oscar-registry
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
  !python -m nltk.downloader stopwords
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
 'IP':'sales piitch',
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
     

def create_ul2_plus_instructions(text,  lang="en", min_encoder_word_span = 5, labels=None):
  """
  Sample from some multilingual data to create UL2 data in the form of instructions
  """
  (subject, a_type, style, num_sent) = get_metadata(text, labels)
  encoder_batch, decoder_batch = [], []
  text = text.strip().replace("..", ".").replace(".-", ".").replace("... ...","...").replace("... ...","...").replace("... ...","...").replace("..", ".").replace("--", "-").replace("--", "-").replace("...", "-").replace("[", "(").replace("]", ")")
  paras = []
  chunks = []
  for line in text.split("."):
    line = line.lstrip()
    if not line: continue
    if len(line) < 10:
      chunks.append(line)
      continue  
    if "return" in line and ("{" in line or "//" in line or "#" in line): # this is code. ignore for now. 
      continue
    if random.randint(0,2)==0 and len(chunks) > 1:
      paras.append(". ".join(chunks))
      chunks =[]
    chunks.append(line)
  paras.append(". ".join(chunks))
  paras = [p.replace(".  ", ". ").replace("a. m.", "a.m.").replace("p. m.", "p.m.").replace("U. S.", "U.S.").replace(". com", ".com").replace(". edu", ".edu").replace(". org", ".org").replace(". gov", ".gov").\
        replace(". 0", ".0").replace(". 1", ".1").replace(". 2", ".2").replace(". 3", ".3").replace(". 4", ".4").replace(". 5", ".5").replace(". 6", ".6").replace(". 7", ".7").replace(". 8", ".8").replace(". 9", ".9").\
        replace(". ”. ", ".” ").replace(". \". ", ".\" ").replace(".\".", ".\"").replace(". ”", ".”").strip() 
        for p in paras]
  dialog = ""
  asked_style=False
  end_para = None
  if (a_type or style) and len(paras[-1]) > 100 and len(paras) > 1 and random.randint(0,1) == 1:
    asked_style=True
    if random.randint(0,1) == 0:
      end_para = paras[-1]
      paras = paras[:-1]
    else:
      if random.randint(0,1) == 0:
        dialog += f"\nUser: What style of text is this: {paras[-1]}\nAssistant: It appears to be a {style} style."
      elif random.randint(0,1) == 0:
        dialog += f"\nUser: What kind of data is this: {paras[-1]}\nAssistant: Most likley {a_type}."
      elif random.randint(0,1) == 0:
        dialog += f"\nUser: For this text:\n {paras[-1]}\nWhat type of writing is this?\nAssistant: Most likley {a_type}."
      else:
        dialog += f"\nUser: Can you tell me the writing style: {paras[-1]}\nAssistant: {style} style."
      paras = paras[:-1]

  for line in paras:   
    #print ('*', line) 
    line = line.lstrip(" .,").replace(".”.", ".”").replace(".\".", ".\"")   
    use_extra_id = random.randint(0,3) > 0
    denoising_type = R_DENOISING
    extra_id = 0
    if lang in ("zh", "th", "ko", "ja", "bo"):
      line = line.split("་")
    else:
      line = line.split()
    encoder_line = []
    decoder_line = []
    prev_has_id = False
    if random.randint(0, 1) == 0:
      denoising_type = S_DENOISING
      if random.randint(0,1) == 1:
        while line:
          l = min(len(line), random.choice((12, 24, 24, 32, 32, 64, 64, 128)))
          if prev_has_id == False:
            if l == len(line):
              l = int(len(line)/2)
            s_denoiser_output = line[l:min(len(line), l*2)] 
            before = line[:l]
            line = line[min(len(line), l*2):]  
          else:
            s_denoiser_output = line[:min(len(line), l)] 
            line = line[min(len(line), l):] 
          if lang in ("bo", "zh", "th", "ko", "ja"):
            if prev_has_id:
              encoder_line = [random.choice(["Next", "More", "Continue"])]
            else:
              encoder_line = ["What could be the completion of this text: "]+ before + ["..."] 
          else:
            if prev_has_id:
              encoder_line = [random.choice(["Next", "More", "Continue"])]
            else:
              encoder_line = ["What could be the completion of this text: "]+ before + ["..."]  
          if prev_has_id:
            decoder_line = s_denoiser_output
          else:
            decoder_line = [f"The completion of this text could be: "]+s_denoiser_output
          if lang in ("zh", "th", "ko", "ja", "bo"):
              encoder_line = "".join(encoder_line).replace("  ", " ")+"."
              decoder_line = "".join(decoder_line).replace("  ", " ")+("." if not line else "...")
          else:
              encoder_line = " ".join(encoder_line).strip().replace("  ", " ")+"."
              decoder_line = " ".join(decoder_line).strip().replace("  ", " ")+("." if not line else "...")
          encoder_batch.append(encoder_line.strip().replace("....","..."))
          decoder_batch.append(decoder_line.strip().replace("....","..."))
          prev_has_id = True
          
      else:
        l = int(len(line)/2)
        s_denoiser_output = line[l:] 
        line = line[:l] 
        if lang in ("bo", "zh", "th", "ko", "ja"):
          encoder_line = ["What could be the completion of this text: "]+ line + ["..."] 
        else:
          encoder_line = ["What could be the completion of this text: "]+ line + ["..."]  
        decoder_line = [f"The completion of this text could be: "]+s_denoiser_output
        if lang in ("zh", "th", "ko", "ja", "bo"):
            encoder_line = "".join(encoder_line).replace("  ", " ")+"."
            decoder_line = "".join(decoder_line).replace("  ", " ")+"."
        else:
            encoder_line = " ".join(encoder_line).strip().replace("  ", " ")+"."
            decoder_line = " ".join(decoder_line).strip().replace("  ", " ")+"."
        encoder_batch.append(encoder_line.strip().replace("....","..."))
        decoder_batch.append(decoder_line.strip().replace("....","..."))
    else:
      choice = random.randint(0, 3) 
      if choice == 0:
        denoising_type = R_DENOISING
      elif choice == 1:
        denoising_type = X1_DENOISING
      else:
        denoising_type = X2_DENOISING
      line_len = len(line)
      last_encoder_span = 0
      while line:
        word = line[0]
        if last_encoder_span > min_encoder_word_span and extra_id < 99 and not prev_has_id:
          if (denoising_type == R_DENOISING  and random.randint(0, 10) in (0, 1)) or \
            (denoising_type ==  X1_DENOISING and random.randint(0,2) == 0) or\
            (denoising_type == X2_DENOISING and  random.randint(0,10) == 0): 
            if lang in ("zh", "th", "ko", "ja"):
              l = min(len(line), random.choice((1, 2, 3, 5, 10)))# this is number of words, and not tokens. original t5 paper did, 2, 3, 5, and 10 tokens.
            else:
              if denoising_type ==  X2_DENOISING:
                l = min(len(line), random.choice((12, 24, 24, 32, 32, 64)))
              else:
                l = min(len(line), random.choice((1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8)))
            eos = [idx for idx, w in enumerate(line[:l]) if w[-1]=="."]
            if eos:
              l = eos[0]
            if line[:l]:
              extra_id+=1
              if not use_extra_id:
                encoder_line.append(f" ... ")
              else:
                if denoising_type in (X1_DENOISING, R_DENOISING):
                  encoder_line.append(f" [words {extra_id}] ")
                else:
                  encoder_line.append(f" [spans {extra_id}] ")
              if denoising_type in (X1_DENOISING, R_DENOISING):
                decoder_line.append(f"\nwords {extra_id}: ")
              elif denoising_type == X2_DENOISING:
                decoder_line.append(f"\nspans {extra_id}: ")    
            
              decoder_line.extend(line[:l])
              line = line[l:]
              prev_has_id = True
              last_encoder_span = 0
              continue
        encoder_line.append(word)
        line = line[1:]
        prev_has_id = False
        last_encoder_span += 1
    
      if decoder_line:
        if lang in ("zh", "th", "ko", "ja", "bo"):
          encoder_line = "".join(encoder_line).rstrip().replace("  ", " ")

          decoder_line = "".join(decoder_line).rstrip().replace("  ", " ")
        else:
          encoder_line = " ".join(encoder_line).rstrip().replace("  ", " ")
          decoder_line = " ".join(decoder_line).rstrip().replace("  ", " ")
        if denoising_type in (X1_DENOISING, R_DENOISING):
            encoder_line = "Fill in the missing words: " + encoder_line +"."
        else:
            encoder_line = "Fill in the missing spans: " + encoder_line +"."
        encoder_batch.append(encoder_line.replace("....","..."))
        decoder_batch.append(decoder_line.replace("....","..."))
  dialog += "\n"+"\n".join(["User: "+ encoder_line + "\nAssistant: "+decoder_line for encoder_line, decoder_line in zip(encoder_batch, decoder_batch)])
  if end_para or ((a_type or style) and not asked_style):
    asked_style=True
    if end_para:
      end_word = end_para.split()[-1]
      start_word = end_para.split()[-1]
      if random.randint(0,1) == 0:
        dialog += "\n"+basic_augment(f"\nUser: Give me another paragraph in the style of above text.\nAssistant: {end_para}")
      elif random.randint(0,1) == 0:
        dialog += f"\nUser: What kind of text were you giving me, and write more text like the above?\nAssistant: We were writing a {a_type} text. Here is more of the same:\n{end_para}"
      elif random.randint(0,1) == 0:
        dialog += f"\nUser: More of the same in {style} style?\nAssistant: {end_para}."
      elif len(end_word) > 3:
        dialog += f"\nUser: Can you tell me the writing style we were using, and give me another paragraph ending with the word '{end_word}'?\nAssistant: Style: {style}.\n{end_para}"
      elif len(start_word) > 3:
        dialog += f"\nUser: Can you tell me the writing style we were using, and give me another paragraph starting with the word '{start_word}'?\nAssistant: Style: {style}.\n{end_para}"
      else:
        dialog += f"\nUser: More of the same and the type of text too please?\nAssistant: The type of text is {style}.\n{end_para}."
    else:      
      if random.randint(0,1) == 0:
        dialog += f"\nUser: What style of text is the above text?\nAssistant: It appears to be a {style} style."
      elif random.randint(0,1) == 0:
        dialog += f"\nUser: What kind of text were you giving me?\nAssistant: Most likley {a_type}."
      elif random.randint(0,1) == 0:
        dialog += f"\nUser: For this above text, what type of writing is this?\nAssistant: Most likley {a_type}."
      else:
        dialog += f"\nUser: Can you tell me the writing style we were using?\nAssistant: {style} style."
  dialog = dialog.strip()

  choice = random.randint(0,5)
  if choice == 0:
    dialog = dialog.replace("What could be the completion of this text: ", "Complete this text: ")
  elif choice == 1:
    dialog = dialog.replace("What could be the completion of this text: ", "Generate more:\n")
  elif choice == 2:
    dialog = dialog.replace("What could be the completion of this text: ", "Finish this: ")
  elif choice == 3:
    dialog = dialog.replace("What could be the completion of this text: ", "The rest of this: ")
  elif choice == 4:
    dialog = dialog.replace("What could be the completion of this text: ", "More of this please:\n")

  choice = random.randint(0,5)
  if choice == 0:
    dialog = dialog.replace("The completion of this text could be: ", "Completed pargraph(s): ")
  elif choice == 1:
    dialog = dialog.replace("The completion of this text could be: ", "")
  elif choice == 2:
    dialog = dialog.replace("The completion of this text could be: ", "Rest of text: ")
  elif choice == 3:
    dialog = dialog.replace("The completion of this text could be: ", "Here you go: ")
  elif choice == 4:
    dialog = dialog.replace("The completion of this text could be: ", "Sure, let me know if you need anything else:\n")

  choice = random.randint(0,5)
  if choice == 0:
    dialog = dialog.replace("Fill in the missing", "Find the missing")
  elif choice == 1:
    dialog = dialog.replace("Fill in the missing", "Fill in")
  elif choice == 2:
    dialog = dialog.replace("Fill in the missing", "More")
  elif choice == 3:
    dialog = dialog.replace("Fill in the missing", "What are the missing")
  elif choice == 4:
    dialog = dialog.replace("Fill in the missing", "Can you tell me how to fill in the missing")

  choice = random.randint(0,6)
  if choice == 0:
    dialog = dialog.replace(": Can you tell me", ": What is")
  elif choice == 1:
    dialog = dialog.replace("paragraph", "section")
  elif choice == 2:
    dialog = dialog.replace(": Give me another", ": Provide more")  
  elif choice == 3:
    dialog = dialog.replace(": Give me ", ": ")
  elif choice == 4:
    dialog = dialog.replace(": Can you tell me ", ": ")     
  elif choice == 5:
    dialog = dialog.replace(": What ", ": ")    
  return dialog

def create_prompt(text, labels):
  (subject, a_type, style, num_sent) = get_metadata(text, labels)
  if not subject: return ""
  text = text.strip()
  if "##" in text and not text.startswith("##") and "." in text:
    idx = text.index(".",1)
    if idx> 50:
      text_arr = text.split()[:3]
      if text_arr[0].lower() not in stop_words and text_arr[0].lower() not in stop_words and text_arr[0].lower() not in stop_words:
        text = "## "+" ".join(text_arr)+"\n"+text
        #print (text)
    else:
      text = "## " + text[:idx]+"\n"+ text[idx+1:]
      #print (text)
  if "." not in text:
    num_sent = 0
  if num_sent <=1:
    text = text.strip("# ")
  if random.randint(0,1)==1 or  num_sent > 10:
    prompts = f'User: Write a {a_type} about {subject}.\nAssistant: {text}'
  elif random.randint(0,1)==1 and num_sent > 1:
    num_para = text.count("\n")+1
    if random.randint(0,1):
      prompts = f'User: Write {num_para} paragraphs about {subject} in {style} style.\nAssistant: {text}'
    else:
      prompts = f'User: Write {num_para} paragraphs in {style} style with the words: {subject}.\nAssistant: {text}'
  elif num_sent==1:
    if random.randint(0,1):
      prompts = f'User: Write {num_sent} sentences about {subject} in {style} style.\nAssistant: {text}'
    else:
      prompts = f'User: Write {num_sent} sentences in {style} style with the words: {subject} .\nAssistant: {text}'
  else:
    if random.randint(0,1):
      prompts = f'User: Write about {subject} in {style} style.\nAssistant: {text}'
    else:
      prompts = f'User: Write in {style} style with the words {subject}.\nAssistant: {text}'
  if prompts:
    if random.randint(0,3) == 1:
      prompts = prompts.replace(": Write", ": Give me")
    elif random.randint(0,2) == 1:
      prompts = prompts.replace(": Write", ": Provide")
    elif random.randint(0,1) == 1:
      prompts = prompts.replace(": Write", ": ")
    
    if random.randint(0,3)==1:
      prompts = prompts.replace(" about ", " using the words ")
    elif random.randint(0,2)==1:
      prompts = prompts.replace(" about ", " with the words ")      
    elif random.randint(0,1)==1:
      prompts = prompts.replace(" about ", " related to ")      
    elif random.randint(0,1)==1:
      prompts = prompts.replace(" about ", " mentioning ")      
    prompts2 = basic_augment(prompts)

  if prompts2: return prompts2
  return prompts
  

def do_oscar_registry(shard_name, do_ul2=False):
  output_file = f"ul2_plus_oscar_{shard_name}.jsonl" if do_ul2 else f"oscar_{shard_name}.jsonl"
  if os.path.exists(f"/content/drive/Shareddrives/LAION/{output_file}.gz"):
    print (f"{output_file} already exists")
    return []
  if not os.path.exists(shard_name+".jsonl.gz"):
    os.system(f"wget https://huggingface.co/datasets/TurkuNLP/register_oscar/resolve/main/en/{shard_name}.jsonl.gz")
  with open(output_file, "w") as output:
    i = 0
    aSent = {}
    for idx, l in enumerate(gzip.open(shard_name+".jsonl.gz")):
      if idx+1 %1000==0:
        if len(aSent) > 10000:
          aSent = dict(Counter(aSent).most_common(10000))
      data = json.loads (l.decode().strip())
      labels, text = data['labels'], data['text']
      if len(labels) == 1 and 'IN' in labels and random.randint(0,5) > 0: 
        continue
      text_lower = text.lower()
      if 'viagra' in text_lower or 'cialis' in text_lower or " hack " in text_lower: continue
      if " a " not in text_lower and " the " not in text_lower and " of " not in text_lower and " and " not in text_lower: continue
      if (" xxx " in text_lower or " sex " in text_lower or " fuck " in text_lower or " porn " in text_lower) and (" video " in text_lower or " pics " in text_lower or " pictures " in text_lower or " horney " in text_lower or " dildo " in text_lower or " anal " in text_lower or " dating " in text_lower or " cock " in text_lower or " pussy " in text or " whore" in text_lower or " cunt" in text_lower) and random.randint(0,4) > 0: continue
      if "{" in text or "}" in text: continue
      menu= ""
      footer = ""
      if not labels or len(text) < 50: continue
      text = text.replace("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", "").replace("0 1 2 3 4 5 6 7 8 9","")
      text = text.replace("’", "'").replace("[…]", "...").replace("\t", " ").replace("…", " ... ")
      text = " "+text
      if " 1. "  in text and " 2. " in text:
        text = text.replace(" 1. ", "\n1. ").\
        replace(" 2. ", "\n2. ").\
        replace(" 3. ", "\n3. ").\
        replace(" 4. ", "\n4. ").\
        replace(" 5. ", "\n5. ").\
        replace(" 6. ", "\n6. ").\
        replace(" 7. ", "\n7. ").\
        replace(" 8. ", "\n8. ").\
        replace(" 9. ", "\n9. ").\
        replace(" 10. ", "\n10. ").\
        replace(" 11. ", "\n11. ").\
        replace(" 12. ", "\n12. ").\
        replace(" 13. ", "\n13. ").\
        replace(" 14. ", "\n14. ").\
        replace(" 15. ", "\n15. ").\
        replace(" 16. ", "\n16. ").\
        replace(" 17. ", "\n17. ").\
        replace(" 18. ", "\n18. ").\
        replace(" 19. ", "\n19. ").\
        replace(" 20. ", "\n20. ").\
        strip()
      text = text.replace("｜", " | ").replace("||", " | ").replace(".", ". ").replace("!", "! ").replace("?", "? ").\
        replace("a. m.", "a.m.").replace("p. m.", "p.m.").replace("U. S.", "U.S.").replace(". com", ".com").replace(". edu", ".edu").replace(". org", ".org").replace(". gov", ".gov").\
        replace(". 0", ".0").replace(". 1", ".1").replace(". 2", ".2").replace(". 3", ".3").replace(". 4", ".4").replace(". 5", ".5").replace(". 6", ".6").replace(". 7", ".7").replace(". 8", ".8").replace(". 9", ".9").\
        replace(". ”. ", ".” ").replace(". \". ", ".\" ").replace(".\".", ".\"").replace(". ”", ".”").replace("[", " [").replace("]", "] ").replace(" -- ", "\n-- ").\
        replace("  "," ").replace(". . .", "...").\
        replace("* ", "* ").replace("• ", "\n• ").\
        replace(". -", ".\n -").\
        replace("? -", "?\n -").strip()
      if "|" in text[:50]:
        text_arr = text.split("|")
        text = " ".join(text_arr[-1].split()[3:])
        text_arr[-1] = " ".join(text_arr[-1].split()[:3])
        menu = " | ".join(text_arr)
        
      if "|" in text[-50:]:
        text_arr = text.split("|")
        text = " ".join(text_arr[0].split()[:-3])
        text_arr[0] = " ".join(text_arr[0].split()[-3:])
        footer = " | ".join(text_arr)
      sents = []  
      if "Disclaimer" in text[:100] or "this site" in text[:100]:
        #print ('has disclaimer', text)
        text = ". ".join(text.split(". ")[4:])
      text = text.split(".")
      for t in text:
        if not t.strip(): continue
        #remove basic spam
        if "dating site" in t or "free online" in t or "free dating" in t or "free porn" in t: continue
        code = hash(t.strip())
        if code in aSent:
          aSent[code] += 1
          continue
        aSent[code] = 1
        t_arr = t.split()
        if t and  len(t_arr) >= 4 and t_arr[0][0].lower() in "qwertyuiopasdfghjklzxcvbnm" and   t_arr[0][0] == t_arr[0][0].upper() and t_arr[3][0] in "qwertyuiopasdfghjklzxcvbnm" and t_arr[3][0] == t_arr[3][0].upper():
          if  len(t_arr) == 4:
            t = "\n##"+t+".\n"
          else:
            t = "\n##"+" ".join(t_arr[:4])+".\n"+" ".join(t_arr[4:])
        sents.append(t)
      text = ".".join(sents).replace("\n. ", "\n").replace("\n.\n", "\n")
      if not text.strip(): continue
      text = text.replace("....", "\n").replace("...", "\n").replace("..", "\n")
      text = text.split(". ")
      sent = []
      title = True
      num = 0
      rel_prompts = {}
      for t in text:
        t = t.lstrip("+]),.?/~!^&*) ")
        if len(t) < 20:
          continue
        if "August" in t and "October" in t: continue
        if " link " in t and " download" in t: continue
        if title and t.startswith("##"): continue
        if t.startswith("##"): 
          title = True
        else:
          title = False
        ratio = number_words_ratio(t)
        if ratio <= .12:
          if t.count(" ") > 1 and len(t) < 150 and ratio > 0.08:
            t = "\n## " + t + ".\n"
            #print (ratio, t)
          else:
            #print (ratio,t)
            continue
        t_arr = t.split()
        found_prefix = False
        if len(t_arr) > 4:
          prefix = " ".join(t_arr[:4]).lower() 
          if prefix in transition_prompt_inv:
            t = "\n"+t
            num = 0
            if sent: rel_prompts[sent[-1]] = (transition_prompt_inv[prefix], t[len(prefix)+1:].strip(", "))
            found_prefix = True
        if not found_prefix and len(t_arr) > 3:
          prefix = " ".join(t_arr[:3]).lower() 
          if prefix in transition_prompt_inv:
            t = "\n"+t
            num = 0
            if sent: rel_prompts[sent[-1]] = (transition_prompt_inv[prefix], t[len(prefix)+1:].strip(", "))
            found_prefix = True
        if not found_prefix and len(t_arr) > 2:
          prefix = " ".join(t_arr[:2]).lower() 
          if prefix in transition_prompt_inv:
            t = "\n"+t
            num = 0
            if sent: rel_prompts[sent[-1]] = (transition_prompt_inv[prefix], t[len(prefix)+1:].strip(", "))
            found_prefix = True
        if not found_prefix and len(t_arr) > 1:
          prefix = t_arr[0].lower() 
          if prefix in transition_prompt_inv:
            t = "\n"+t
            num = 0
            if sent: rel_prompts[sent[-1]] = (transition_prompt_inv[prefix], t[len(prefix)+1:].strip(", "))
            found_prefix = True
        sent.append(t)
        num += 1
        if (num+1)%4 == 0:
          sent[-1] = "\n"+sent[-1]
      text = ". ".join(sent)
      text = text.replace("\n\n", "\n").replace("\n. ", ".\n").replace("\n\n", "\n")
      text = text.replace("[Read More]", "").replace("  ", " ")
      chunks = []
      dialog = ""
      if do_ul2:
        dialog = create_ul2_plus_instructions(text, labels)
        dialog = dialog.strip()
        if dialog:
          output.write(json.dumps({"text": dialog, "metadata": {"source": "ul2_plus_oscar_"+shard_name}})+"\n")      
      else:
        for chunk in text.split("\n"):
          if len(chunk) < 50: 
            chunks.append(chunk)
            continue
          if len(chunks) == 1 and (sum([len(c) for c in chunks]) > 1000) and random.randint(0,3) == 1:
            dialog += "\n"+create_prompt("\n".join(chunks), labels)
            chunks = []
          elif len(chunks) == 2 and random.randint(0,3) == 1:
            dialog += "\n"+create_prompt("\n".join(chunks), labels)
            chunks = []
          elif len(chunks) == 3 and random.randint(0,3) == 1:
            dialog += "\n"+create_prompt("\n".join(chunks), labels)
            chunks = []
          elif len(chunks) > 3 and random.randint(0,3) == 1:
            dialog += "\n"+create_prompt("\n".join(chunks), labels)
            chunks = []
          elif len(chunks) == 1 and random.randint(0,5) == 1:
            dialog += "\n"+create_prompt("\n".join(chunks), labels)
            chunks = []
          chunks.append(chunk)
        if chunks:
          dialog += "\n"+create_prompt("\n".join(chunks), labels) 
        dialog = dialog.strip()
        if dialog:
          output.write(json.dumps({"text": dialog, "metadata": {"source": "oscar_"+shard_name}})+"\n")
          
      i+=1
      #if i > 500: break
  os.system(f"rm {shard_name}.jsonl.gz")
  if do_ul2:
    os.system(f"gzip {output_file}")
    os.system(f"mv {output_file}.gz /content/drive/Shareddrives/LAION")
  else:
    os.system(f"gzip {output_file}")    
    os.system(f"mv {output_file}.gz /content/drive/Shareddrives/LAION")
  return  [f"/content/drive/Shareddrives/LAION/{output_file}.gz"]

import multiprocessing, functools
def multiprocess_oscar(docs_start=0, docs_end=99999, do_ul2=False):
  docs = ["en_{:05d}".format(number) for number in range(docs_start, docs_end)]
  with multiprocessing.Pool(processes=8) as pool:
      results = pool.map(functools.partial(do_oscar_registry, do_ul2=do_ul2), docs) 
  return results

#multiprocess_oscar(300, 325, do_ul2=True)
