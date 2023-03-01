#@title Soda-dialog
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
from datasets import load_dataset

def generate_soda_dialog(output):
  dataset = load_dataset("allenai/soda")
  for i in range(len(dataset['train'])):
    dat = dataset['train'][i]
    title = dat['literal']
    story = dat['narrative']
    theme = ""
    if dat['relation'] == 'xWant':
      theme = "wanting " + dat['tail']
    elif dat['relation'] == 'xNeed':
      theme = "needing " + dat['tail']
    elif not dat['tail'].startswith("to ") and  not dat['tail'].startswith("and "):
      theme = "being "+ dat['tail']
    elif dat['tail'].startswith("and "):
      theme = "people are "+ dat['tail'].replace("and PersonY ", "")
    else:
      theme =  dat['tail']    
    theme = theme.replace("PersonY", "another person")  
    theme = theme.replace("being is", "being")             
    dialog = [s2+": "+s1 for s1, s2 in zip(dat['dialogue'],dat['speakers'])]
    txt = ""
    start = random.choice(["Ok, ", "Sure, ", "Of course, ", ""])

    categories =  get_main_non_stopwords(story)
    if random.randint(0,6)==0 and categories:
      categories = ", ".join(categories)
      txt = f"User: Can you give me a short story idea for {categories}?\n"
      txt += f"Assistant: {start}, a short story idea for {categories}:\n  {story}.\n"
      dialog = dialog.replace(dat['speakers'][0], "User2").replace(dat['speakers'][1], "User")
      title = title.replace(dat['speakers'][0], "User2").replace(dat['speakers'][1], "User")
      theme = theme.replace(dat['speakers'][0], "User2").replace(dat['speakers'][1], "User")
      dialog2 = ""
      for d in dialog.split("\n"):
        if random.randint(0,3) == 0 and len(dialog2)>1 and "User:" in d and "@Assistant" not in dialog2:
            dialog2 += f"User: @Assistant, what would be a theme of my conversation with @User2?\nAssistant: One theme of your conversation could be {theme}.\n"
        dialog2 += d
      txt += dialog2
      txt += f"User2: @Assistant, can you summarize my conversation with User?\nAssistant: {title}.\n"
    elif random.randint(0,6)==0:
      txt = f"User: Can you give me a short story description for this dialog?\n"
      txt += "  "+"\n  ".join(dialog)+"\n"
      txt += f"Assistant: {start}, a short story description for this dialog could be: \n  {story}\n"
      txt += "User: And a title?\n"
      txt += f"Assistant: {start}a title for this dialog could be: \n  {title}\n"
      if theme:      
        txt += "User: What would be one theme of this story?\n"
        txt += f"Assistant: One theme of this story could be: \"{theme}\"\n"     
    elif random.randint(0,3)==0:  
      txt = f"User: Can you write a short dialog based on this story:\n  {story}\n"
      txt += f"Assistant: {start}a dialog for this story could be:\n"
      txt += "  "+"\n  ".join(dialog)+"\n"
      txt += "User: And a title?\n"
      txt += f"Assistant: {start}a title for this dialog could be: \n  {title}\n"  
      if theme:      
        if random.randint(0,1) == 0:
          txt += "User: What would be one theme of this story?\n"
        else:
          txt += "User: a theme\n"
        txt += f"Assistant: One theme of this story could be: \"{theme}\"\n"       
    elif random.randint(0,3)==0:  
      txt = f"User: Can you write the next few lines of dialog for this scene:\n"
      if random.randint(0,1) == 0:
        txt += "  "+"\n  ".join(dialog[:-5])+"\n"
        txt += f"Assistant: {start}the next dialog for this scene could be:\n"
        txt += "  "+"\n  ".join(dialog[-5:])+"\n"
      elif random.randint(0,1) == 0:
        txt += "  "+"\n  ".join(dialog[:-3])+"\n"
        txt += f"Assistant: {start}the next dialog for this scene could be:\n"
        txt += "  "+"\n  ".join(dialog[-3:])+"\n"
      else:
        txt += "  "+"\n  ".join(dialog[:-4])+"\n"
        txt += f"Assistant: {start}the next dialog for this scene could be:\n"
        txt += "  "+"\n  ".join(dialog[-4:])+"\n"           
      txt += "User: And a title?\n"
      txt += f"Assistant: {start}a title for this dialog could be: \n  {title}\n"       
      txt += "User: How about a short description?\n"
      txt += f"Assistant: {start}a short description for this dialog could be: \n  {story}\n"  
      if theme:      
        if random.randint(0,1) == 0:
          txt += "User: What would be one theme of this story?\n"
        else:
          txt += "User: a theme?\n"
        txt += f"Assistant: One theme of this story could be: \"{theme}\"\n"     
    elif random.randint(0,3)==0:  
      title1 = title.split(".")[0]
      title2 = title.split(".")[1]
      txt = f"User: Can you write short story about: {title1}\n"
      txt += f"Assistant: {start}a short story about: \"{title1}\" could be:\n"
      txt += f"  {story}\n"
      if random.randint(0,3) < 3:
          txt += "  "+"\n  ".join(dialog)+"\n"  
      elif random.randint(0,1) == 0 and  len(dialog) > 5:
          txt += "  "+"\n  ".join(dialog[:-5])+"\n"  
          txt += f"User: Can you provide more dialog assuming \"{title2}\"?\n"
          txt += f"Assistant: {start}the next dialog for this scene could be:\n"
          txt += "  "+"\n  ".join(dialog[-5:])+"\n"  
      elif random.randint(0,1) == 0:
          txt += "  "+"\n  ".join(dialog[:-3])+"\n"  
          txt += "User: more please.\n"
          txt += f"Assistant: {start}the next dialog for this scene could be:\n"
          txt += "  "+"\n  ".join(dialog[-3:])+"\n"  
      else:
          txt += "  "+"\n  ".join(dialog[:-4])+"\n"  
          txt += f"User: Can you provide more dialog assuming \"{title2}\"?\n"
          txt += f"Assistant: {start}the next dialog for this scene could be:\n"
          txt += "  "+"\n  ".join(dialog[-4:])+"\n"     
      if theme:      
        txt += "User: What would be one theme of this story?\n"
        txt += f"Assistant: One theme of this story could be: \"{theme}\"\n"    
    else:
      txt = f"User: Can you write a short story and dialog based on the theme:\n  {theme}\n"
      txt += f"Assistant: {start}a short story and dialog based on the theme \"{theme}\" could be:\n"
      txt += f"  {story}\n"
      txt += "  "+"\n  ".join(dialog)+"\n"  
      txt += "User: And a title?\n"
      txt += f"Assistant: {start}a title for this dialog could be: \n  {title}\n"    
    if txt:
      if random.randint(0,1) == 1:
        txt = txt.replace("short story", "story")
      if random.randint(0,1) == 1:
        txt = txt.replace("more please", "next")
      if random.randint(0,1) == 1:
        txt = txt.replace("more please", "continue")
      if random.randint(0,1) == 1:
        txt = txt.replace("Can you ", "")
      txt = txt.strip("\n ")
      if "User2: " not in txt:
        txt = basic_augment(txt)
      if txt:
        output.write(json.dumps({"text": txt, 'metadata': {'source': 'soda-dialog'}})+"\n")
      
  
                
