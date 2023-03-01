#@title XP3
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
import spacy
import glob, os, json
import random
for file in glob.glob("/content/drive/Shareddrives/LAION/xp3/*"):
  name = file.split("/")[-1]
  lang = name.split("_")[-1].split(".")[0]
  if lang not in {'en',}: continue
  if not os.path.exists(name):
    os.system(f"cp {file} ./")
  for idx, l in enumerate(open(name)):
      if idx > 10: break
      data = json.loads(l.strip())
      print (data)
      inputs, targets = data["inputs"], data["targets"]
      inputs = inputs.replace("\\n","\n")
      targets = targets.replace("\\n","\n")
      inputs = inputs.replace("-lrb-", "(").replace("-rrb-", ")")
      inputsHash = {}
      instr = ""
      for inpt in inputs.split("\n"):
        if ":" in inpt:
          aspect, inpt = inpt.split(":",1)
          aspect = aspect.strip()
          if " " not in aspect and inpt != '':
            aspect = aspect.strip()
            inpt = inpt.strip().replace(".", ". ")
            inputsHash[aspect] = inpt
      print (inputsHash)
      if "The previous"  in inputs:
        text, instr = inputs.split("The previous" , 1)
        instr = "The previous"  + instr
      elif "What is" in inputs:
        text, instr = inputs.split("What is",1)
        instr = "What is"+instr
      elif "\nGive" in inputs:
        text, instr = inputs.split("\nGive",1)
        instr = "Give"+instr
      elif "\nWrite" in inputs:
        text, instr = inputs.split("\nWrite",1)
        instr = "Write"+instr
      elif "\nGenerate" in inputs:
        text, instr = inputs.split("\nGenerate",1)
        instr = "Generate"+instr
      elif "\nProvide" in inputs:
        text, instr = inputs.split("\nProvide",1)
        instr = "Provide"+instr
      elif  "\n##" in inputs :
        text, instr = inputs.split("\n\n",1)
        instr = instr.replace("#", "")
      elif "?" in inputs:
        if inputs[-1] == "?":
          instr = "".join(reversed("".join(reversed(inputs)).split(".",1)[0]))
        else:
          instr = inputs.split("?",1)[0].split(".")[-1]
          instr = instr+"?"
        text = inputs.replace(instr, "")
      elif "\n\n" in inputs:
        text, instr = inputs.split("\n\n",1)
      elif ":" in inputs:
        instr = inputs.split(":")[-2]
        text = inputs.replace(instr+":", "")
      elif inputs[-1] == ".":
         instr = inputs.split(".")[-2]
         text = inputs.replace(instr+".", "")
      else:
        instr = inputs.split(".")[-1]
        text = inputs.replace(instr+".", "")
      prefix = genre = ""
      if ":" in text:
        prefix, text = text.split(":",1)
        if ":" in prefix:
          tmp = prefix.split(":")[-2].strip()
          if " " not in tmp or len(tmp) < 50:
            genre = tmp
      else:
        prefix = ""
      genre = genre.strip()
      print ("**", genre)
      if genre == "Answer": genre = ""
      prefix = prefix.strip().replace(genre+":","").strip()
      instr = instr.replace("The same text in", ("Given " + prefix+", translate to") if prefix else "Translate to").replace(genre+":","")
      instr = instr.replace("Summary ", "Summarize ")
      instr = instr.replace("Here is a translation ", "Translate to ").replace("Here is the same text in ", "Translate to ").replace(" to to ", " to ").strip()
      text = text.replace(genre+":","").replace(".", ". ").replace("  ", " ").strip()
      if len(instr) > len(text):
        tmp = text
        instr = text
        text = instr
      print ("##")
      orig_key = ""
      span_1 = ""
      span2_2 = ""
      span_summary = ""
      modified_inputs = ""
      modified_inputs_with_summary = ""
      instr = instr.replace("following", "").replace("  ", " ")
      instr = instr.replace("above", "").replace("  ", " ")
      if not instr or not text or not targets: continue

      # fill in the blanks - using a summary or the actual text
      if False: # random.randint(0,1) == 0:
          val = text.split(".")
          len_val = len(val)
          remove_1 = random.randint(0,len_val-1)
          remove_2 = (remove_1 + 2) % len_val
          span_1 = val[remove_1].strip()
          val[remove_1] = ""
          span_2 = val[remove_2].strip()
          val[remove_2] = ""
          modified_inputs = ".".join(val).replace(" .", ".").strip()
          if (span_1 or span_2) and len(val) > 10:
            span_summary = run_model("summarize: " + span_1 + ". " + span_2, t5_model, t5_tokenizer, max_length=512)[0]
            print ('###', span_summary, '##', span_1, '###', span_2)
            val[remove_1] = span_summary
            modified_inputs_with_summary = ".".join(val).replace(" .", ".").strip()
      

      if random.randint(0,1) == 0 and len(inputsHash) >= 1: # 
        items = list(inputsHash.items())
        items.sort(key=lambda a: len(a[1]), reverse=True)
        orig_key, val = items[0]
        
        print (f"User: What kind of text is this?\n{val}\n\nAssistant: This appears to be a type of {orig_key}.")
        items = items[1:]
        if items and random.randint(0,1) == 0:
          key, val = items[0]
          print (f"User: What is a possible {key} for this {orig_key}? \n\nAssistant: A possible {key} for this {orig_key} could be '{val}'.")
        
      if genre:
        targets = "Here is a "+genre + ". " + targets
      text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").strip()
      targets = targets.replace(" .", ".").replace("  ", " ").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").strip()
      if orig_key:
        print (f"User: {instr}\n\nAssistant: {targets}")
      else:
        choice = random.randint(0,10)
        if choice == 0:
          print (f"User: {instr}\n{text}\n\nAssistant: {targets}")
        elif choice == 1:
          print (f"User: Please follow my instructions based on the following text: {text}\n{instr}\n\nAssistant: {targets}")
        elif choice == 2:
          print (f"User: I will give you some text and my instruction:\n{text}\n{instr}\n\nAssistant: {targets}")
        else:
          print (f"User: {text}\n{instr}\n\nAssistant: {targets}")
  
  #os.system(f"rm {name}")
  #break
