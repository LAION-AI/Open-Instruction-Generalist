#@title Infill Q/A
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
import random, json
import pandas as pd
try:
  from unidecode import unidecode
except:
  !pip install unidecode
from unidecode import unidecode
import spacy
import json
import os
import glob
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


if False:
  import pandas as pd
  import spacy
  import json
def infil_old():
  try:
    if sci is  None: pass
  except:
    sci = spacy.load("en_ner_craft_md")
    data = pd.read_parquet('data.parquet', engine='pyarrow')

  with open("biomed.jsonl", "w") as output:
    for a, b in zip(data['labeled_dialog'],data['page_title']):
      a = a.replace("this article", "this subject").replace("()", "").replace("  ", " ")
      if ' arts ' in a or 'comic' in a or 'novel' in a or ' story' in a or 'movie' in a or 'film' in a or 'music' in a:
        #print ('###arts\n', a)
        continue  
      if ' game' in a or 'sports' in a or 'football' in a or 'soccer' in a or 'baseball' in a or 'basketball' in a:
        #print ('###sports\n', a)
        continue  
      if 'population' in a or 'territory' in a or 'village' in a or 'country' in a or 'county' in a:
        #print ('###place\n', a)
        continue
      if 'ingredient' in a or 'food' in a or 'recipe' in a:
        #print ('###recipe\n', a)
        continue
      if ' law' in a or ' rights' in a or ' court ' in a or ' criminal ' in a or ' verdict ' in a or ' guilt ' in a or ' legislat' in a:
        #print ('###law\n', a)
        continue

      doc = sci(a)
      j = 0
      for ent in doc.ents:
        if ent.label_ == 'SO' or (ent.label_ == 'CHEBI' and len(ent.text) > 5):
          j+= 1
          if j > 4:
            output.write (json.dumps({'page_title': b, 'labeled_dialog': a})+"\n")
            break
          #print (ent.label_, ent.text)

#poorman's reverb. TODO: we need to use semantic matching of relationship to paragraph to filter out bad relationships.
def get_verb_relation(text):
  doc = basic_nlp(text)
  verb_relationship = ""
  orig_verb = ""
  for token in doc:
    #print (token, token.tag_)
    if token.tag_.startswith("VB") and token.tag_ not in {"VBZ", } and token.lemma_ not in {'do', 'be', 'have', 'list'}:
      orig_verb = token.text
      verb_relationship = str(token.lemma_)
      continue  
    if verb_relationship:
      if token.tag_ == "IN":
        orig_verb += " "+token.text
        verb_relationship += "_"+str(token.lemma_)
        break
      else:
        break
  if verb_relationship == "bear": 
    verb_relationship = "born"
  return verb_relationship, orig_verb

#need to filter out rel that don't match embedding of full text. these are spurious
def ner_rel_template_extract(text, min_ner_len=5, length_for_rel=50, min_ner_per_domain=3):
  ret = {}
  orig_text = text
  text2 = text.replace("{", "-lbracket-").replace("}", "-rbracket-")
  ner_cnt = {}
  for nlp in [blackstone, sci, basic_nlp]:
    doc =nlp(text)
    ents = [(ent.text.strip(), ent.label_) for ent in  list(doc.ents) if len(ent.text.strip()) >= min_ner_len]
    if nlp != basic_nlp and len(ents) < min_ner_per_domain: continue
    ents.sort(key=lambda a: len(a[0]), reverse=True)
    for st, label in ents:
      #we are not doing NER for code
      if "->" in st or "{" in st or "}" in st: continue 
      if st in text:
        ner_cnt[label] = ner_cnt.get(label, -1)
        ner_cnt[label] += 1
        if ner_cnt[label] > 0:
          text2 = text2.replace(st,'{'+label+'_'+str(ner_cnt[label])+'}')
          ret[st] = label+'_'+str(ner_cnt[label])
        else:
          text2 = text2.replace(st,'{'+label+'}')
          ret[st] = label
        text = text.replace(st,' ')
    rels =[]
    if nlp == basic_nlp:

      args = dict([(b, "{"+a+"}") for a, b in ret.items() ])
      if args:
        text3 = text2.format(**args)
        text4 = text3.replace("{", " ").replace("}", " ")
        for entity in ret.keys():
          if "{"+entity+"}" not in text3:
            continue
            #print ('problem', "{"+entity+"}", '***', text3)
          text5= text4[text3.index("{"+entity+"}"):]
          if len(text5) > length_for_rel:
            text5 = text5[:length_for_rel]
          rel, orig_verb = get_verb_relation(text5)
          if "{"+entity+"}" in text3 and rel:
            text6 = text3[text3.index("{"+entity+"}"):].split(orig_verb)
            if len(text6) < 2: continue
            text6 = text6[1]
            if "{" in text6:
              text6 = text6.split("{",1)[1]
              if "}" in text6:
                entity2 = text6.split("}")[0]
                rels.append ((entity.replace(" ", "_") ,rel, entity2.replace(" ", "_") ))
      
  return ret, text2.replace("-lbracket-", "{").replace("-rbracket-", "}"), rels

def output_data(entity, instructions, context, output, min_ner_len=5, length_for_rel=50):
  context = context[0]
  context_arr = context.split(".")
  style = ""
  if len(context_arr) >= 24:
    style = " in six paragraphs"
    mult = int(len(context_arr)/6)
    context_arr[mult] = "\n"+context_arr[mult].strip()
    context_arr[2*mult] = "\n"+context_arr[2*mult].strip()
    context_arr[3*mult] = "\n"+context_arr[3*mult].strip()
    context_arr[4*mult] = "\n"+context_arr[3*mult].strip()
    context_arr[5*mult] = "\n"+context_arr[3*mult].strip()
    context = ".".join(context_arr)
  if len(context_arr) >= 20:
    style = " in five paragraphs"
    mult = int(len(context_arr)/5)
    context_arr[mult] = "\n"+context_arr[mult].strip()
    context_arr[2*mult] = "\n"+context_arr[2*mult].strip()
    context_arr[3*mult] = "\n"+context_arr[3*mult].strip()
    context_arr[4*mult] = "\n"+context_arr[3*mult].strip()
    context = ".".join(context_arr)
  if len(context_arr) >= 16:
    style = " in four paragraphs"
    mult = int(len(context_arr)/4)
    context_arr[mult] = "\n"+context_arr[mult].strip()
    context_arr[2*mult] = "\n"+context_arr[2*mult].strip()
    context_arr[3*mult] = "\n"+context_arr[3*mult].strip()
    context = ".".join(context_arr)
  elif len(context_arr) >= 12:
    style = " in three paragraphs"
    context_arr[4] = "\n"+context_arr[4].strip()
    context_arr[8] = "\n"+context_arr[8].strip()
    context = ".".join(context_arr)    
  elif len(context_arr) >= 8:
    style = " in two paragraphs"
    context_arr[4] = "\n"+context_arr[4].strip()
    context = ".".join(context_arr)        
  elif len(context_arr) >= 4:
    style = " in one paragraph"
    if random.randint(0,3) > 0: return
  elif len(context_arr) == 3:
    style = " in three sentences"
    if random.randint(0,5) > 0: return
  else:
    return 
  ner_rel = ner_rel_template_extract(context,  min_ner_len=min_ner_len, length_for_rel=length_for_rel)
  first_sent = basic_nlp(context_arr[0])
  first_sent = [a.text for a in first_sent.noun_chunks if a.text not in entity and a.text.lower() != a.text and len(a.text) > 4]
  if len(first_sent) > 3:
    first_sent = first_sent[:3]
  #print ("###")
  if ner_rel and first_sent:
    ner = [a for a in ner_rel[0] if a not in entity and a not in first_sent]
    if len(ner) > 2:
      ner = ner[:2]
    context_instruction = (f"User: Write me an article about "+ ", ".join(first_sent) + ", discussing in detail " + ", ".join(ner)+ style + ".")
  elif first_sent:
    context_instruction = (f"User: Write me an article about "+ ", ".join(first_sent) + style + ".")
  elif ner_rel:
    ner = [a for a in ner_rel[0] if a not in entity]
    if len(ner) > 2:
      ner = ner[:2]    
    context_instruction = (f"User: Write me an article about "+ ", ".join(ner)+ style + ".") 
  else:
    ner = [a for a in ner_rel[0] if a not in entity]
    if len(ner) > 2:
      ner = ner[:2]    
    context_instruction = (f"User: Write me an article about {entity}"+ style + ".") 
  

  last_sent = basic_nlp(context_arr[-2])
  if (context_instruction or first_sent) and last_sent != context_arr[0]:
    last_sent = [a.text for a in last_sent.noun_chunks if a.text not in entity and a.text.lower() != a.text and len(a.text) > 4]  
    if len(last_sent) > 2:
      last_sent = last_sent[:2]  
    if last_sent and random.randint(0,1) == 0:
      context_instruction += (f" End with a sentence about "+ ", ".join(last_sent)+".") 
  
  instructions = instructions.strip()
  format_type = random.randint(0,3)
  if format_type == 0:
    out = (context_instruction + "\n" + "Assistant: " + context+ "\n"+ instructions) 
    out = out.replace("Write me an article about", random.choice(["Write me an article about", "Provide an article about", "Give me an article about"]))
  elif format_type == 1:
    first_instruction =  instructions.split("\n")[0].split(": ",1)[1].strip()
    if first_instruction[1:].lower() == first_instruction[1:]:
      ner_rel_text =  "; ".join(str(a) for a in ner_rel[-1]) if ner_rel[-1] else ('' if not ner_rel[0] else "; ".join(str(a) for a in ner_rel[0].items()) )
      if not ner_rel_text: return
      instructions = "User: " + first_instruction + "\n\n" + "Assistant: I'm sorry I can't answer that question based on the information I have.\n\n" + \
        "User: Answer the question assuming the following : " + ner_rel_text+ ".\n\n" + instructions.split("\n\n",1)[-1]
    out = (instructions+"\n"+context_instruction + "\n" + "Assistant: " + context) 
    out = out.replace("Write me an article about", random.choice(["Based on the above, write me an article about", "Using the above, provide an article about", "Summarizing the above, give me an article about"]))
  else:
    if entity.replace("_", " ") not in instructions.split("\n")[0] and entity.replace("_", " ").lower() not in instructions.split("\n")[0]:
      instructions = "User: " +  random.choice(["Tell me about", "Provide one sentence about", "Briefly describe"]) + " " + entity.replace("_", " ") +".\n\n"+ \
        "Assistant: "+ context_arr[0] + ".\n\n" + instructions
    out = ("Background: " + context+ "\n"+ instructions) 
  out = out.replace("\n\n", "\n").replace("()", "").replace("  ", " ")
  #print ("###")
  #print (out)
  output.write (json.dumps({'text': out, 'metadata': {'source': 'infil_dbpedia'}})+"\n")
  

aHash = sci = data = basic_nlp = blackstone = aHash = None
def create_abstract_infil(output):
  #TODO clear context, output jsonl, list, table format. algorithmic ops
  global aHash, sci, data, basic_nlp, blackstone
  if not os.path.exists("/content/data.parquet"):
    !wget https://huggingface.co/datasets/ericyu3/openassistant_inpainted_dialogs/resolve/main/data.parquet
    
  if not os.path.exists("/content/long-abstracts_lang=en.ttl"):
    !wget https://databus.dbpedia.org/dbpedia/text/long-abstracts/2022.09.01/long-abstracts_lang=en.ttl.bz2
    !bunzip2 long-abstracts_lang=en.ttl.bz2

  try:
    if sci is  None: assert False
  except:
    sci = spacy.load("en_ner_craft_md")
    data = pd.read_parquet('data.parquet', engine='pyarrow')
    basic_nlp = spacy.load('en_core_web_sm')
    blackstone = spacy.load("en_blackstone_proto")
    # add the other scispacy ner  
      
  if aHash is None:

    aHash = {}
    with open("/content/long-abstracts_lang=en.ttl") as input:
      for l in input:
        l = l.strip()
        l = l.split(" ",2)
        entity = l[0].split("/")[-1].split(">")[0].lower().replace("&amp;", "&").strip("_").replace("-", "_")
        #topic = l[1].split("/")[-1].split(">")[0]
        sent = l[-1].split("\"@")[0].strip('"00')
        aHash[unidecode(entity)] = aHash.get(unidecode(entity), []) + [sent]
        if entity.count("_") > 1:
          entity2 = unidecode("_".join(entity.split("_")[:2]).strip("_"))
          if entity2 not in aHash:
            aHash[entity2] = aHash.get(entity2, []) + [sent]
          if entity.count("_") > 2:
            entity2 = unidecode("_".join(entity.split("_")[:3]).strip("_"))
            if entity2 not in aHash:
              aHash[entity2] = aHash.get(entity2, []) + [sent]
        if "(" in entity:
          entity, cat = entity.split("(", 1)
          cat = cat.split("_")
          entity = unidecode(entity + "("+cat[0]+")")
          aHash[entity] = aHash.get(entity, []) + [sent]

 
  

  for a, b in zip(data['labeled_dialog'],data['page_title']):
    b = b.replace("(, ","(").replace("()","").replace("  ", " ")
    a = a.replace("Are there any other interesting aspects about this article?", random.choice(["more please", "next", "continue", "and?", "tell me more", "anything else?"]))
    a = a.replace("What else did you find important?",  random.choice(["more please", "next", "continue", "and?", "tell me more", "anything else?"]))
    b = b.replace(" ", "_").replace("&amp;", "&").strip("_")
    if unidecode(b.lower().replace("-", "_")) not in aHash:
        if "(" in b:
          b2, cat = b.split("(", 1)
          cat = cat.split("_")
          b2 = b2 + "("+cat[0]+")"
          if unidecode(b2.lower().replace("-", "_")) in aHash: 
            context = aHash[ unidecode(b2.lower().replace("-", "_"))]
            output_data(b, a, context, output)            
            continue
          if b2.count("_") > 1:
            b2 = "_".join(b2.split("_")[:2]).strip("_")
            if unidecode(b2.lower().replace("-", "_")) in aHash: 
              context = aHash[ unidecode(b2.lower().replace("-", "_"))]
              output_data(b, a, context, output)             
              continue
            if b2.count("_") > 2:
              b2 = "_".join(b2.split("_")[:3]).strip("_")
              if unidecode(b2.lower().replace("-", "_")) in aHash: 
                context = aHash[ unidecode(b2.lower().replace("-", "_"))]
                output_data(b, a, context, output)              
                continue
        
    else:
      context = aHash[unidecode(b.lower().replace("-", "_"))]
      output_data(b, a, context, output)

def abstract_infil2(output):
  i = 0
  with open("/content/drive/Shareddrives/LAION/OIG/abstact_infill.jsonl") as infile:
    for l in infile:
      data = json.loads(l.strip())
      data['metadata'] = data['meta']
      del data['meta']
      text = data['text']
      text_arr = text.split("User:")
      b = ([a for a in text_arr if "('" in a])
      if b:
        print (b)
        i+=1
      #if i > 20: break
    print (i)
#abstract_infil2(None)            
