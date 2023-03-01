#@title CUAD
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

import json, os
def create_cuad(output):
  if not os.path.exists("CUADv1.json"):
    os.system("wget https://github.com/TheAtticusProject/cuad/raw/main/data.zip")
    os.system("unzip data.zip")
  cuad = json.load(open("CUADv1.json"))
  #cuad['data'][0]['title'], 
  for cuad0 in cuad['data']:
    dialog_all = ""
    for para in cuad0['paragraphs']:
      context = para['context'].replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
      context = "\n".join(a for a in context.split("\n") if "Page -" not in a).replace("[***]", "___")
      context_arr = context.split("\n")
      sec0 = random.randint(1,4)
      sec = str(sec0)+". "
      next_sec = str(sec0+1)+". "
      next_next_sec = str(sec0+2)+". "
      sec_idx = [idx for idx, item in enumerate(context_arr) if item.startswith(sec) or item.startswith(next_sec) or item.startswith(next_next_sec)]
      if len(sec_idx) == 3:
        context0 = "\n".join(context_arr[:min(len(context_arr), sec_idx[1])])
        dialog_all += "\n"+(create_ul2_plus_instructions(context0))
        section_before = ("\n".join(context_arr[sec_idx[0]:min(len(context_arr), sec_idx[1])]))
        section_after = ("\n".join(context_arr[sec_idx[1]:min(len(context_arr), sec_idx[2])]))
        if random.randint(0,1):
          dialog = f"User: What would be a contract section that comes after this one:\n{section_before}\nAssistant: {section_after}"
          dialog = basic_augment(dialog)
          dialog_all += "\n"+(dialog)
        else:
          dialog = f"User: What would be a contract section that comes before this one:\n{section_after}\nAssistant: {section_before}"
          dialog = basic_augment(dialog)
          dialog_all += "\n"+(dialog)  
        i = min(len(context_arr), sec_idx[2])
      else:
        i = 0
      for rng in range(i, len(context_arr), 10):
        if dialog_all == "":
          dialog = "User: "+random.choice(["Complete the next paragraph of this contract:", "Give me more for this agreement:", "Provide a continuation for this:", "What comes next for this:"])+": "+"\n".join(context_arr[rng:min(len(context_arr),rng+5)]) +"\nAssistant: "+"\n".join(context_arr[rng+5:min(len(context_arr),rng+10)])
        else:
          dialog = "User: "+random.choice(["More from this contract.", "Next lines please.", "Continue.", "More."])+"\nAssistant: "+"\n".join(context_arr[rng:min(len(context_arr),rng+10)])
        dialog_all += "\n"+(dialog)

      for qa in para['qas']:
        question = qa['question'].split("Details:")[1].strip(".? ")
        if qa['is_impossible'] and not len(qa['answers']):
          if question.startswith("Is") or question.startswith("Does") or question.startswith("Can") or question.startswith("Will"):
            if random.randint(0,4)==0:
              dialog = ("User: "+question+"?\nAssistant: "+random.choice(["No.", "It does not appear so.", "I don't think so."]))
              dialog_all += "\n"+(dialog)
          else:
            if random.randint(0,4)==0:
              dialog = ("User: "+question+"?\nAssistant: "+random.choice(["I can't tell from this context.", "I don't know. Can you provide more information?", ]))
              dialog_all += "\n"+(dialog)
        elif len(qa['answers']) > 1:
          if  question.startswith("Is") or question.startswith("Does") or question.startswith("Can") or question.startswith("Will"):
            yes = "Yes. "
          else:
            yes = ""
          dialog = ("User: "+question+f"?\nAssistant: {yes}\n* " + "\n* ".join(a['text'] for a in qa['answers']))
          dialog_all += "\n"+(dialog)
        else:
          if  question.startswith("Is") or question.startswith("Does") or question.startswith("Can") or question.startswith("Will"):
            yes = "Yes. "
          else:
            yes = ""
          dialog = ("User: "+question+f"?\nAssistant: {yes}"+qa['answers'][0]['text'])
          dialog_all += "\n"+(dialog)
      dialog_all = dialog_all.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").replace("___ ___", "___").replace("___ ___", "___").replace("___ ___", "___")
      if dialog_all:
        output.write(json.dumps({"text": dialog_all, "metadata": {"source": "cuad"}})+"\n")
with open("cuad.jsonl", "w") as output:
  create_cuad(output)
os.system("cp cuad.jsonl /content/drive/Shareddrives/LAION/OIG")
