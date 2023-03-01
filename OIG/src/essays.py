#@title essays
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
#TODO: add essay fixing of wording, formatting and sentence order by permuting
try:
  from datasets import load_dataset
except:
  !pip install datasets
from datasets import load_dataset  
import random, json
def create_essays(output):
  try:
    if essays: pass
  except:
    essays = load_dataset("ChristophSchuhmann/essays-with-instructions")
  for idx, data in enumerate(essays['train']):
    first_section = ""
    instructions_old, summary, text = data['instructions'], data['titles'], data['essays']
    summary = summary.strip()
    instructions = summary
    instructions = instructions.replace("The author argues that", "Furthermore")
    if "essay discusses" in instructions:
      instructions = instructions.split("essay discusses",1)[-1]
    instructions = instructions.replace("The author of this", "This")
    instructions = instructions.strip()
    instructions = instructions[0].lower() + instructions[1:]
    instructions = instructions.replace(" text ", " essay ")
    if instructions.startswith("the essay"):
      instructions = "Write "+instructions.replace("the essay", "a essay").replace("essay", "essay that",1)
    elif instructions.startswith("this essay"):
      instructions = "Write "+instructions.replace("this essay", "a essay").replace("essay", "essay that",1)
    elif " paper " in instructions:
      instructions = "Write "+instructions.replace("this paper", "a paper").replace("paper", "paper that",1)
    elif " proposal " in instructions:
      instructions = "Write "+instructions.replace("this proposal", "a proposal")#.replace("proposal", "proposal that",1)
    else:
      instructions = "Write an essay about "+instructions 
    instructions = instructions.replace("an essay about the essay", "an essay that").replace("an essay about this essay", "an essay that").\
      replace("the purpose of this","").replace("  ", " ").replace("an essay about essay was", "an essay").\
      replace("that that ", "that ").replace("It argues", "Argue")
    text = text.replace("Works Cited", "References")
    text = text.replace("Work Cited", "References")
    text = text.replace("Reference List", "References")
    text = text.replace("List of References", "References")
    text = text.replace("Bibliography", "References")
    text = text.replace("Reference\n", "References\n")
    toc = ""
    if 'Table of Contents' in text:
      before, text = text.split('Table of Contents',1)
      first_section = text.split("2.")[0].replace("1. ","").strip()
      before, toc, text = text.split(first_section,2)
      toc = "1. " + first_section+toc
    if toc:
      toc = [t for t in [t.strip(" 0123456789\n") for t in toc.split(".") if t.strip()] if t]
    ref = ""
    if "References" in text:
        ref = text.split("References",1)[-1]
    if ref:
      text = text.replace(ref,"")
      text = text[:-len("References")]
    if first_section:
      text = (first_section+"\n"+text).strip()
    text = text.replace("\n\n", "\n").strip()
    ref = ref.replace("\n\n", "\n").strip()
    if ref:
      ref = [r for r in [r.split(",")[1].split(".")[-1].strip(" \n\"\'”") for r in ref.split("\n") if r.strip() and len(r.split(","))>1] if len(r) > 15]
    appendix = ""
    if "Appendix" in ref:
      arr = ref.split("Appendix")
      ref = arr[0].strip()
      appendix = ("##Appendix" + "##Appendix".join(arr[1:])).strip()
    instructions = instructions.strip().split("\n")[0]
    if ". " in instructions:
      instructions1, rest_instructions = instructions.split(". ",1)
      instructions1 = instructions1.replace(" is ", " that is ").replace(" are ", " that are ").replace(" can ", " that can ").replace(" will ", " that will ").replace(" discusses ", " that discusses ")
      instructions = instructions1+". " + rest_instructions
    else:
      instructions = instructions.replace(" have ", " that have ").replace(" has ", " that has ").replace(" is ", " that is ").replace(" are ", " that are ").replace(" can ", " that can ").replace(" will ", " that will ").replace(" discusses ", " that discusses ")
    instructions = instructions.replace("that is that it that is", "is that it is").replace("it that is", "it is").replace("that that", "that")
    text = text.replace("\n\n", "\n").replace("\n\n", "\n")
    if toc:
      for t in toc:
        text = text.replace(t, '##'+t)
    else:
      text2 = ""
      for t in text.split("\n"):
        if t.startswith("Table") or t.startswith("Figure") or t.startswith("Diagram") or t.startswith("Panel"):
          text2 += "\n###"+t
          continue
        elif t.startswith("Conclusion") or t.startswith("Intro") or t.startswith("Discussion"):
          text2 += "\n##"+t
          continue
        elif len(t) < 25 and " " in t and len(t.split()) > 2:
          t_arr = t.split()
          if t_arr[0][0] ==  t_arr[0][0].upper() and t_arr[2][0] ==  t_arr[2][0].upper():
            text2 += "\n##"+t
            continue
        elif len(t) < 25 and " " in t and (t.split()) == 2:
          t_arr = t.split()
          if t_arr[0][0] ==  t_arr[0][0].upper() and t_arr[1][0] ==  t_arr[1][0].upper():
            text2 += "\n##"+t
            continue
        elif len(t) < 25 and " " not in t:
          t_arr = t.split()
          if t_arr[0][0] ==  t_arr[0][0].upper():
            text2 += "\n##"+t
            continue
        text2 += "\n"+t
      text = text2.strip()
    if text[0] != '#':
      if "#" not in text or text.split("\n")[1][0] == '#':
        text = "#"+ text
      else:
        text = "##"+ text
    summary = summary.replace("\n\n", "\n").replace("\n", ". ")
    do_summary=True
    dialog = ""
    if summary and random.randint(0,1)==0:
      do_summary=False
      first_q = instructions.split("?")[0].split(".")[0].replace("an essay", "a essay").replace("essay", "summary").replace("report", "summary").replace("proposal", "summary")
      dialog +=  (f"\nUser: {first_q}.\nAssistant: {summary}.".replace("..","."))
    if not do_summary and ref and random.randint(0,1)==0 and "&" not in ref[0]:
      if random.randint(0,1):
        dialog += (f"\nUser: What could be a reference for this essay?\nAssistant: {random.choice(ref)}")
      elif len(ref) > 1:
        dialog +=  (f"\nUser: What could be a reference for this essay?\nAssistant: {ref[0]}")
        for r in ref[1:]:
          if random.randint(0,1):
            dialog +=  (f"\nUser: What could be a reference for this essay?\nAssistant: {r}") 
          elif random.randint(0,1):
            dialog +=  (f"\nUser: Another\nAssistant: {r}") 
          else:
            dialog +=  (f"\nUser: Another reference?\nAssistant: {r}")
    if text:
      if random.randint(0,1) and text.count("##") > 1:
        start=True
        secs = text.split("##")
        random.shuffle(secs)
        for section in secs:
          if "\n" not in section: continue
          section, text2 = section.split("\n",1)
          if start:
            start=False
            first_q = instructions.split("?")[0].split(".")[0]
            dialog +=  (f"\nUser: {first_q}. Write the {section} only.\nAssistant: {text2}")
          else:
            if random.randint(0,1):
              dialog +=  (f"\nUser: Write an {section} section.\nAssistant: {text2}")
            elif random.randint(0,1):
              dialog +=  (f"\nUser: How about a {section} section.\nAssistant: {text2}")
            else:
               dialog +=  (f"\nUser: {section}\nAssistant: {text2}")
        if appendix and random.randint(0,1)==0 :
          dialog +=  (f"\nUser: What are possible appendices for this essay?\nAssistant: {appendix}")
          dialog +=  (f"\nUser: "+random.choice(["Give me the whole essay based on the above", "Now the whole article.", "Put it all togegther in the right order.", "The complete essay please."])+f".\nAssistant: {text}\n{appendix}")
          appendix = ""
        else:
          dialog +=  (f"\nUser: "+random.choice(["Give me the whole essay based on the above", "Now the whole article.", "Put it all togegther in the right order.", "The complete essay please."])+f".\nAssistant: {text}")
      elif not do_summary:
        if random.randint(0,1)==0:
          first_q = instructions.split("?")[0].split(".")[0]
          dialog +=  (f"\nUser: {first_q}. Exapnd on the summary above.\nAssistant: {text}")
        else:
          dialog +=  (f"\nUser: Write an essay based on the summary above.\nAssistant: {text}")
      else:
        if random.randint(0,1)==0:
          first_q = instructions.split("?")[0].split(".")[0]
          dialog +=  (f"\nUser: {first_q}.\nAssistant: {text}")
        elif random.randint(0,1)==0 and "Summary" not in text:
          do_summary = False
          first_q = instructions.split("?")[0].split(".")[0]
          dialog +=  (f"\nUser: {first_q}\nAssistant: ##Executive Summary\n{summary}\n{text}")      
        else:
          do_summary = False
          if random.randint(0,1):
            dialog +=  (f"\nUser: {instructions}\nAssistant: {text}")
          else:
            dialog +=  (f"\nUser: Write an essay for this summary: {summary}\nAssistant: {text}")
        if "Summary" in text:
          do_summary = False
    if appendix and random.randint(0,1)==0 :
      dialog +=  (f"\nUser: What are possible appendices for this essay?\nAssistant: {appendix}")
    if do_summary and random.randint(0,1)==0:
      dialog +=  (f"\nUser: Write a summary for this essay.\nAssistant: {summary}.".replace("..","."))
    if do_summary and ref and random.randint(0,1)==0 and "&" not in ref[0]:
      if random.randint(0,1):
        dialog +=  (f"\nUser: What could be a reference for this essay?\nAssistant: {random.choice(ref)}")
      elif len(ref) > 1:
        dialog +=  (f"\nUser: What could be a reference for this essay?\nAssistant: {ref[0]}")
        for r in ref[1:]:
          if random.randint(0,1):
            dialog +=  (f"\nUser: What could be a reference for this essay?\nAssistant: {r}") 
          elif random.randint(0,1):
            dialog +=  (f"\nUser: Another\nAssistant: {r}") 
          else:
            dialog +=  (f"\nUser: Another reference?\nAssistant: {r}")
    dialog = dialog.strip()
    choice = random.randint(0,2)
    if choice == 0:
      dialog = dialog.replace("Write the", "Give me")
    elif choice == 1:
      dialog = dialog.replace("Write the", "Provide an")
    elif choice == 2:
      dialog = dialog.replace("Write the", "")
    choice = random.randint(0,2)
    if choice == 0:
      dialog = dialog.replace("Write an", "Give me")
    elif choice == 1:
      dialog = dialog.replace("Write an", "Provide an")
    elif choice == 2:
      dialog = dialog.replace("Write an", "")
    choice = random.randint(0,2)
    if choice == 0:
      dialog = dialog.replace("about the", "regarding")
    elif choice == 1:
      dialog = dialog.replace("about the", "relating to")
    elif choice == 2:
      dialog = dialog.replace("about the", "for")
    dialog = dialog.replace("..", ".").replace("\n\n", "\n")
    dialog = basic_augment(dialog)
    output.write(json.dumps({'text':dialog, "metadata": {'source': 'essays'}})+"\n")

with open("essays.jsonl", "w") as output:
  create_essays(output)
!cp essays.jsonl /content/drive/Shareddrives/LAION/OIG
