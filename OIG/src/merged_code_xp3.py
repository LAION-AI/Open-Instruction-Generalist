#@title merged_code_xp3
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

import os
import json, random

from torch import e
def create_merged_code_xp3(output):
  if not os.path.exists("merged_code.jsonl"):
    os.system("wget https://huggingface.co/datasets/bigscience/xP3/resolve/main/code/merged_code.jsonl")

  with open("merged_code.jsonl") as file:
    for l in file:
      data = json.loads(l.strip())
      if ' def ' in data['inputs'] or ' def ' in data['targets']:
        inputs, targets = data['inputs'], data['targets']
        inputs = inputs.replace("<image>", "").replace(" ; ", ".\n").replace("; ", ".\n")
        if inputs.startswith ("def ") or inputs.startswith("class ") or inputs[0] in {'#', '@'}:
          inputs = inputs.strip()
          if "\n" not in inputs:
            inputs = "Complete this python program:\n" + inputs
          else:  
            inputs = "Solve the following python programming problem given the following code:\n" + inputs
        prefix = "Here is the Python code you requested"
        if inputs[0] == '"':
          prefix = inputs.split("\n")[0].strip('"').split("|")[0] + " and the other steps required."
          prefix = ("Here is code to " + prefix[0].lower()+prefix[1:] + "\n").replace("  ", " ")
        elif "Find "in inputs:
          prefix = inputs.split("Find ",1)[1].split("\n")[0]
          prefix = "Here is Python code to find " + prefix+"\n"
        elif "find "in inputs:
          prefix = inputs.split("find ",1)[1].split("\n")[0]
          prefix = "Here is Python code to find " + prefix+"\n"
        elif "Determine "in inputs:
          prefix = inputs.split("Determine ",1)[1].split("\n")[0]
          prefix = "Here is Python code to determine " + prefix+"\n"
        elif "determine "in inputs:
          prefix = inputs.split("determine ",1)[1].split("\n")[0]
          prefix = "Here is Python code to determine " + prefix+"\n"
        elif "Fix "in inputs:
          prefix = inputs.split("Fix ",1)[1].split("\n")[0]
          prefix = "Here is Python code to fix " + prefix+"\n"
        elif "fix "in inputs:
          prefix = inputs.split("fix ",1)[1].split("\n")[0]
          prefix = "Here is Python code to fix " + prefix+"\n"
        elif "Print "in inputs:
          prefix = inputs.split("Print ",1)[1].split("\n")[0]
          prefix = "Here is Python code to print " + prefix+"\n"
        elif "print "in inputs:
          prefix = inputs.split("print ",1)[1].split("\n")[0]
          prefix = "Here is Python code to print " + prefix+"\n"
        elif "Compare "in inputs:
          prefix = inputs.split("Compare ",1)[1].split("\n")[0]
          prefix = "Here is Python code to compare " + prefix+"\n"
        elif "compare "in inputs:
          prefix = inputs.split("compare ",1)[1].split("\n")[0]
          prefix = "Here is Python code to compare " + prefix+"\n"
        elif "Compute "in inputs:
          prefix = inputs.split("Compute ",1)[1].split("\n")[0]
          prefix = "Here is Python code to compute " + prefix+"\n"
        elif "compute "in inputs:
          prefix = inputs.split("compute ",1)[1].split("\n")[0]
          prefix = "Here is Python code to compute " + prefix+"\n"
        elif "Calculate "in inputs:
          prefix = inputs.split("Calculate ",1)[1].split("\n")[0]
          prefix = "Here is Python code to calculate " + prefix+"\n"
        elif "calculate "in inputs:
          prefix = inputs.split("calculate ",1)[1].split("\n")[0]
          prefix = "Here is Python code to calculate " + prefix+"\n"
        elif "What" in inputs:
          prefix = inputs.split("What",1)[1].split("\n")[0]
          prefix = "Here" + prefix+"\n"
        elif "what" in inputs:
          prefix = inputs.split("what",1)[1].split("\n")[0]
          prefix = "Here" + prefix+"\n"
        inputs = inputs.strip('"')
        if prefix: 
          prefix = prefix.split("?")[0]
          prefix = "#"+prefix.split(".")[0]+".\n"
        if len(targets) <= 20: continue
        if not inputs.startswith("Complete this python program") and not inputs.startswith("Solve the following python programming"):
          if random.randint(0,1) and prefix:
            command = random.choice(["\n"," ", " ... ", "\n=====\n"])+prefix.replace("Here is", random.choice(["Write me a", "Give me a", "What is a", "Can you provide a"])).strip("\n .")+"?"
            command = command.replace("#", "")
            text= (f"User: {inputs}{command}\nAssistant: {prefix}{targets}")
          elif random.randint(0,1) and prefix:
            command = prefix.replace("Here is", random.choice(["Write me a", "Give me a", "What is a", "Can you provide a"])).strip(".\n ")+" given the following:\n"
            command = command.replace("#", "")
            text= (f"User: {command}{inputs}\nAssistant: {prefix}{targets}")
          else:
            text= (f"User: {inputs}\nAssistant: {prefix}{targets}")  
        else:
          text= (f"User: {inputs}\nAssistant: {prefix}{targets}")  
        text = text.replace("\n\n\n", "\n\n")
        if random.randint(0,1):
          text = text.replace("Python code", "Python program")
        output.write(json.dumps({'text':text, "metadata": {'source': 'merged_code_xp3'}})+"\n")

