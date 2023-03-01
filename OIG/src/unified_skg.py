#@title UnifiedSKG
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

import json, random

def ask_context(context, text):
  if context: 
    context_arr = context.split(".")
    if len(context_arr) > 4:
      context = ".".join(context_arr[-3:])
      context_arr = context_arr[:-3]
    else:
      context_arr = []
    if random.randint(0,1) == 0 or not text:
        text += "User: Tell me more about "+context.split(".")[0]+".\nAssistant: " + context
    else:
        text += "User: Tell me more about this subject.\nAssistant: " + context
    for rng in range(0, len(context_arr), 3):
        context = ".".join(context_arr[rng:min(len(context_arr), rng+3)])
        if random.randint(0,1) == 0:
          text += "\nUser: More please.\nAssistant: " + context
        elif random.randint(0,1) == 0:
          text += "\nUser: More.\nAssistant: " + context
        else:
          text += "\nUser: Continue.\nAssistant: " + context   
  return text

def process_unifiedskg(output):
  i = 0
  seen = {}
  if True: #with open("unifiedskg_instructions.jsonl", "w") as output:
    with open("/content/drive/Shareddrives/LAION/OIG/unifiedskg.jsonl") as input:
      for l in input:
        data = json.loads(l.strip())
        context = data['context'].replace("list of", "")
        if context and context.lower() in data['structured data']:
          sd = data['structured data'].replace(context.lower(),"").strip()
        else:
          sd = data['structured data'].strip()
        if not sd: continue
        del data['structured data']
        
        if  "<page_title>" in sd:
          continue 
          #sd = sd.replace("<page_title>", "###  ").replace(" </page_title> <section_title> ", ", ").replace(" </section_title> <table> <cell> ","\n * ").\
          #        replace("<cell> ", " * ").replace(" <col_header>", " is the ").replace("</col_header>", " ").replace("</cell>", " ").replace("</table>", "\n").\
          #        replace("</row_header>", "|").replace("<row_header>", " ").replace("  ", " ").replace("*", "\n*").strip() 
          #sd = sd.replace("  is the", "\n--")
          #sd = "\n".join([s for s in sd.split("\n") if s.strip() and s.strip()[0] != '-'])
          #print (sd, "**\n",data)
          #break
          #continue
        table_name = ""
        is_table = False
        if "col :" in sd:
          is_table = True
          table_name = sd.split(":",1)[-2].split(" col")[0].strip()
          if table_name == "col": table_name = ""
          if table_name:
            sd = sd.replace(table_name, "").strip(" :")
          table_name = table_name.replace(" | ", ": ")
          col_sd = sd.replace("col :", "|").split("row",1)
          if len(col_sd) == 1:
            continue
          col, sd = col_sd
          col  = ("| ID | "+ col + "|\n"+ (("| --------- " )*(col.count("|") +1)) + "|\n").replace("| |", "|")
          sd = "| "+sd.replace("row ", "|\n| ")+" |"
          sd = col + sd
          sd = ("|".join(sd.split(":"))).replace("_", " ").replace("  ", " ").replace("| – |", "|   |")
          if table_name:
            sd = "### "+table_name+"\n" + sd
        else:
          sd = sd.replace("type.object.type", "is a type of")
          sd = " ".join([s if "." not in s else "'s "+ s.split(".")[-1].replace("_", " ") + " is " for s in sd.split()])
          sd = sorted([s.strip() for s in sd.replace("of 's ", "of ").replace("is  's", "is ").replace("s is ", "s are").split("|")])
          for idx, el in enumerate(sd):
            el = el.strip()
            el = el.lstrip("+-1234567890 ")
            if el.startswith("'s "): el = el.lstrip("'s ")
            if el.startswith("is "): el = el.lstrip("is ")
            if el.endswith("is "): el = el.rstrip("is ")
            el = el.replace("is  is", "is")
            el = el.replace("is  is", "is")
            el = el.replace("is  is", "is")
            el = el.replace(" 's contains are ", " contains ")
            el = el.replace(" population number ", " population ")
            el = el.replace(" 's partially containedby is ", " is partially contained by ")
            el = el.replace(" 's containedby ", " is contained by ")
            el = el.replace("containedby", "contained by")
            el = el.replace("  ", " ")
            if el.count(":") == 2:
              a, b, c = el.split(":")
              el = a + " 's " + b + " is " + c
              el = el.replace("  ", " ")
            el = el.replace("[TABLECONTEXT] : [title] : ", "")
            el = el.replace("[TABLECONTEXT] : ", "")
            el = el.replace("[", " ")
            el = el.replace("]", " ")
            el = el.strip()
            el = el.lstrip("+-1234567890 ")
            el = el.replace("  ", " ")
            if el.endswith(" is"): el = ""
            sd[idx] = el
          sd = [s.rstrip(".")+"." for s in sd if s]
          sd.sort()
          sd_arr = sd
          if random.randint(0,1) == 0:
            sd = "\n".join([" "+str(idx)+". " + s for idx, s in enumerate(sd_arr)])
          else:
            sd = '* '+("\n* ".join(sd_arr))      
        if sd in seen: continue
        seen[sd] = 1
        choice = random.randint(0,5)
        if choice == 0:
          text = ""
          add_context = False
          if random.randint(0,1) == 0:
            text = ask_context(context, "")
            add_context = True            
          text += "\nBackground:\n" + sd + "\nUser: " + data['query'] + "\nAssistant: " + data['output']
          if not add_context: 
            text = ask_context(context, text+"\n") 
        elif choice == 1:
          text = ""
          add_context = False
          if random.randint(0,1) == 0:
            text = ask_context(context, "")
            add_context = True    
          text += "\nBackground:\n" + sd + "\nUser: What is a question for which the answer is '" + data['output'] + "'\nAssistant: One question for which the answer is '"+ data['output'] + "' could be: "+ data['query']
          if not add_context: 
            text = ask_context(context, text+"\n") 
        elif choice == 2:
          text = ""
          add_context = False
          if random.randint(0,1) == 0:
            text = ask_context(context, "")
            add_context = True           
          text += "\nUser: What is one question you can ask based on this data:\n"+ sd +"\nAssistant: " + data['query'] +"\nUser: Now tell me the answer.\nAssistant: " + data['output']
          if not add_context: 
            text = ask_context(context, text+"\n")           
        elif choice == 3:
          context2 = ""
          if context:
            context_arr = context.split(".")
            if random.randint(0,1) == 0 and len(context_arr) > 3:
              context2 = ".".join(context_arr[-3:])
              context_arr = context_arr[:-3]
              context = ".".join(context_arr)
            else:
              context2 = ""
          text = ""
          add_context = False
          if not context2 and random.randint(0,1) == 0:
            text = ask_context(context, "")
            add_context = True 
          instr = random.choice(["Answer using the following:", "I will give you a question and some data. Please answer the question.", "", "Here is some data.", "Read this and respond based on my instructions."])
          if context2:
            if random.randint(0,1) == 0:
              text += "\nBackground:\n"+ context2+"\nUser: "+instr+"\n"+sd + "\n"+data['query'] + "\nAssistant: " + data['output']
            elif random.randint(0,1) == 0:
              if instr == "I will give you a question and some data. Please answer the question.": instr = ""
              text  += "\nBackground:\n"+ context2+"\nUser: "+data['query'] +"\n"+instr+ "\n" +sd + "\nAssistant: " + data['output']
            elif random.randint(0,1) == 0:
              text  += "\nUser: "+instr+" Given this context: "+ context2+"\n"+sd + "\n"+data['query'] + "\nAssistant: " + data['output']  
            else:
              text  += "\nUser: Given this context: "+ context2+"\n"+instr+"\n"+sd + "\n"+data['query'] + "\nAssistant: " + data['output']  
          else:
            if random.randint(0,1) == 0:
              text  += "\nUser: "+instr+"\n"+sd + "\n"+data['query'] + "\nAssistant: " + data['output']
            else:
              if instr == "I will give you a question and some data. Please answer the question.": instr = ""
              text  += "\nUser: "+data['query'] +"\n"+instr+ "\n" +sd + "\nAssistant: " + data['output']
          if not add_context:
            text = ask_context(context, text+"\n")  
        elif choice == 4 and (table_name or context):
          text = ""
          add_context = False
          if random.randint(0,1) == 0:
            text = ask_context(context, "")
            add_context = True           
          if "|" in sd:
            fields = sd.split("\n")
            if sd.startswith("##"):
              fields = fields[1]
            else:
              fields = fields[0]
            fields = fields.strip(" |").replace(" | ", ", ")
            if random.randint(0,1) == 0:
              text += f"\nUser: Give me a table with the fields {fields} about "+ table_name.split(",")[0] if table_name else context.split(".")[0] + ".\nAssistant:\n"+sd +"\nUser: " + data['query'] +"\Assistant: " + data['output']
            else:
              text += f"\nUser: Give me a table of data with the fields {fields}.\nAssistant:\n"+sd +"\nUser: " + data['query'] +"\Assistant: " + data['output']  
          else:
            text += "\nUser: Give me a list of data about "+ table_name.split(",")[0] if table_name else context.split(".")[0] + "\nAssistant:\n"+sd +"\nUser: " + data['query'] +"\Assistant: " + data['output']
          if not add_context: 
            text = ask_context(context, text+"\n")           
        else:
          text = ""
          add_context = False
          if random.randint(0,1) == 0:
            text = ask_context(context, "")
            add_context = True
          if "|" in sd:
            text += "\nUser: " + data['query'] +"\nAssistant: " + data['output'] +"\nUser: Give me a table of data useful for answering this question.\nAssistant:\n"+sd 
          else:
            text += "\nUser: " + data['query'] +"\nAssistant: " + data['output'] +"\nUser: Give me a list of data useful for answering this question.\nAssistant:\n"+sd 
          if not add_context: 
            text = ask_context(context, text+"\n")   
        #if context: print (context, '***\n', text)
        text = text.strip()+"\n"
        output.write(json.dumps({'text': text, 'metadata': {'source': 'unifiedskg'}})+"\n")
        if "|" in text and '*' in text:
          pass
          #print (text)
        i += 1
        #if i > 100: break
    
process_unifiedskg()
