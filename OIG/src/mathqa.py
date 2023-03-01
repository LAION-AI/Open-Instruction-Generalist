#@title mathqa
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

from tqdm.contrib import tzip
import random, os
try:
  from datasets import load_dataset
except:
  os.system("pip install datasets")
from datasets import load_dataset  
import json

def math_qa(output):
  dataset = load_dataset("math_qa")

  qs = dataset['train']['Problem']
  ans = dataset['train']['Rationale']

  options = dataset['train']['options']
  cors = dataset['train']['correct']
  for a, q, opt, core in tzip(ans, qs, options, cors):
      a = a.replace("mr.", "mr")
      a = a.replace("dr.", "dr")
      a = a.replace("mrs", "mrs")
      q = q.replace("’", "'")
      a = a.replace("’", "'")
      a = a.replace("per. son", "person ").replace(" ;", ".\n").replace("no. ", "number ").replace("sq. ", "sq ").replace("⇒", ".\n").replace("- >", ".\n").replace("= >", ".\n").replace("∴", ". ∴").replace("hence", ". hence").replace("therefore", ". therefore").replace("thus", ". thus").replace("so", ". so").replace("let", ". let").replace("i.e.",".\n")
      a = a.replace(" - - -", ".\n").replace("- - ", "").replace(". . .", "\n").strip('"').replace(". ",".\n").replace("sol .\n","").replace("explanation:","")
      if "the answer is" in a:
        a = a.split("the answer is")[0]
      if "answer" in a:
        a = a.split("answer")[0] 
      if "ans " in a:
        a = a.split("ans ")[0] 
      if "option " in a:
        a = a.split("option ")[0] 
      if "choice" in a:
        a = a.split("choice")[0]     
              
      a = a.strip(" \n.").replace("\n.\n","\n").replace(" .",".").replace(" ,",",").replace("\ni.\ne.", "\n").replace("\ni. \ne.\n", "\n").replace("\ni.\ne", "\n").replace("\ni. \ne\n", "\n")  
      a = a.replace("\nthus.\n","\n")
      a = "\n".join(a1.strip(" ,.").rstrip(" -.")+"." for a1 in a.split("\n") if len(a1) > 1)
      if [a1 for a1 in a.split("\n") if "+." in a1 or "=." in a1 or len(a1.strip(" .")) == 1 or (")" in a1 and "(" not in a1) or ("[" in a1 and "]" not in a1) or ("(" in a1 and ")" not in a1) or ("]" in a1 and "[" not in a1)]:
        continue
      if not a.strip():
        continue
      a = a.replace("..",".").replace("as.\n","as").replace("hence.","").replace("so.", "").replace("thus.","").replace("now.", "").replace("solution.\n", "").replace("solution","").strip()
      a = a.replace("play.\n", "play ").replace("per.\nson", "person").replace("corret", "").replace("  ", " ").replace("explanation :.\n", "").replace(" .", ".").replace(" '","'").replace(" :", ":").replace(" / ", "/").\
          replace("sq.\n", "sq ").replace("al.\nso", "\nalso").replace("no.\n", "number ").replace("to.\n", "to ").replace("the.\n", "the ").replace("and.\n", "and ").replace("are.\n", "are ").replace("is.\n", "is ").replace("per.\nson.", "person ").replace("and.\nso on", "and so on").replace(" ,", ",").rstrip(" ,.")
      a = ("\n".join([((p.strip()[0].upper()+p.strip()[1:]) if len(p) > 1 and p[1] != ' ' else p.strip()).strip(' ,?') for p in a.split("\n") if p.strip() and "option" not in p and "correct" not in p and "answer" not in p])+".")
      a = a.replace(" ,",",").replace(" .",".").replace("..",".").replace(",.",".").strip(" ,").replace(" ' ", "'").replace("' ", "'").replace(" '", "'")  
      q = q.replace(" ,",",").replace(" .",".").replace("..",".").replace(",.",".").strip(" ,").replace(" ' ", "'").replace("' ", "'").replace(" '", "'")  
      if [a1 for a1 in a.split("\n") if "+." in a1 or "=." in a1 or len(a1.strip(" .")) == 1 or (")" in a1 and "(" not in a1) or ("[" in a1 and "]" not in a1) or ("(" in a1 and ")" not in a1) or ("]" in a1 and "[" not in a1)]:
        continue
      if len(a.strip(". ,")) > 20:
        q = q[0].upper() + q[1:]
        a = a[0].upper() + a[1:]
        q = q.strip()
        a = a.strip()
        if "\n" in a:
          if random.randint(0,1) == 0:
            pr = f"User: {q}. "+random.choice(["And explain please.", "Let's think step by step.", "Can you show me your work?", "Help me solve this step by step."])+f"\nAssistant: {a}".replace("..", ".").replace("?.", "?").replace(" i ", " I ")
            output.write (json.dumps({"text":pr, "metadata": {"source": "mathqa"}})+"\n")
          else:
            a2 = a.split("\n")
            final_answer = a2[-1].replace("∴", "").replace("Hence", "").replace("Therefore", "").replace("Then", "").replace("Thus", "").replace("So", "").strip()
            final_answer = final_answer[0].upper() + final_answer[1:]
            pr = f"User: {q}.\nAssistant: {final_answer}\nUser: Can you solve this step by step?\Assistant: Sure.\n{a}".replace("..", ".").replace("?.", "?").replace(" i ", " I ")
            output.write (json.dumps({"text":pr, "metadata": {"source": "mathqa"}})+"\n")
        else:
          pr = f"User: {q}. Let's think step by step.\nAssistant: {a}".replace("..", ".").replace("?.", "?").replace(" i ", " I ")
          output.write (json.dumps({"text":pr, "metadata": {"source": "mathqa"}})+"\n")
            
