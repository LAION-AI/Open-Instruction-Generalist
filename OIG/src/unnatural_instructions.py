#@title unnatural_instructions
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
def create_unatural_instructions(output):
  %cd /content/
  !git clone https://github.com/orhonovich/unnatural-instructions
  if not os.path.exists("full_data.jsonl"):
    os.system("unzip /content/unnatural-instructions/data/full_data.zip")
  j = 0
  instruction_with_output = []
  with open("full_data.jsonl") as input:
    for l in input:
      dat = json.loads(l)
      #print (dat)
      instruction_with_output.extend([(dat2['instruction_with_input'], dat2['output']) for dat2 in dat.get('reformulations',[])])
      instruction_with_output.extend([(dat2['instruction_with_input'], dat2['output']) for dat2 in dat.get('instances',[])])
  instruction_with_output = list(set(instruction_with_output))
  import json
  i = 0
  if True:
    for a, b in instruction_with_output:
      a = a.strip()
      a = a.replace("<sep>", "\n").replace("?", "?").replace("?", "?")
      b = b.strip()
      b = b.replace("<sep>", "\n").replace("?", "?").replace("?", "?")
      if b.count("?") == 1:
        if b[-1]  not in "?":
          continue
      output.write(json.dumps({'text': "User: "+ a+"\nAssistant: "+ b, 'metadata': {'source': 'unatural_instructions'}})+"\n")
  
  #!cp una* /content/drive/Shareddrives/LAION/OIG/  
