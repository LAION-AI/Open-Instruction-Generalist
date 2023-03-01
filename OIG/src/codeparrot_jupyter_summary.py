#@title codeparrot jupyter summary
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

import re, json

pattern = re.compile(r'(?<!^)(?=[A-Z])')

def create_codeparrot_jupyter_summary(output):
  ret = []

  for i in range(len(dataset['train'])):
      data = dataset['train'][i]
      if not data['cells']: continue
      if "gpl" in data["license"]: continue
      if "unlicensed" in data["license"]: continue
      name = data['repo_name'].split("/")[1]
      name2 = ""
      for n in name.split():
        if n.upper() != n:
          n = pattern.sub('_', n)
        name2 = " "+n
      name2 = name2.lower()
      path = data['path']
      path = path.replace("/", " ").replace(".ipynb", "")
      path2 = ""
      for n in path.split():
        if n.upper() != n:
          n = pattern.sub('_', n)
        path2 = " "+n
      path2 = path2.lower() 
        
      content = data['cells']
      types = data['types']
      if types[1] == 'markdown':
        summary = content[1]
        content = [content[0]]+content[2:]
        types = [types[0]]+types[2:]
        
      else:
        summary = content[0]
        content = content[1:]
        types = types[1:]      
      text = '\n'.join(['```'+con+'```' if typ == 'code' else con for con, typ in zip(content, types)])
      subj = name2.strip()
      if path2.strip() not in subj:
        subj = subj + ", " + path2
      subj = subj.strip().replace("_", " ").replace("  ", " ")
      out.write (json.dumps({'subject': subj,'summary': summary, 'text': text})+"\n")
      #if i > 10: break
