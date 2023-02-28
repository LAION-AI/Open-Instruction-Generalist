#@title Labeled Safety Data

#Image Prompt Safety
import os, random
def ceate_safety_image_prompts(output):
  try:
    if laion_img is not None: pass
  except:
    if not os.path.exists("laion_safe_nonsafe.tsv"):
      !cp  /content/drive/Shareddrives/ontocord/laion_safe_nonsafe.tsv ./
    laion_img = [a.split("\t")[0] for a in open("laion_safe_nonsafe.tsv").read().split("\n") if len(a.split("\t")[0]) > 20 and "|| hate" not in a and "|| porn" not in a and "csam" not in a and "|| offensive" not in a]
    laion_5k = random.sample(laion_img, 5000)
    !grep 'offensive' /content/laion_safe_nonsafe.tsv > offensive.tsv
    !head -5000 offensive.tsv > offensive_5k.tsv
    !grep 'porn' /content/laion_safe_nonsafe.tsv > porn.tsv
    !head -5000 porn.tsv > porn_5k.tsv
    !grep 'csam' /content/laion_safe_nonsafe.tsv > csam.tsv
    offensive_5k = [a.split("\t")[0] for a in open("offensive_5k.tsv").read().split("\n")]
    porn = [a.split("\t")[0] for a in open("porn.tsv").read().split("\n")]
    csam = [a.split("\t")[0] for a in open("csam.tsv").read().split("\n")]

  if not os.path.exists("detected_prompts.json"):
    !wget https://laion.droom.cloud/detected_prompts.json
  import json
  data = json.load(open("detected_prompts.json"))
  dialog = []
  for txt in offensive_5k:
    txt = txt.split("||")[0].strip()
    bad_words = [word.strip("()[].,") for word in txt.split() if word.lower() not in stopwords_set]
    bad_words = list(set([word for word in bad_words if word.lower() in hate_words_set or word.lower() in flagged_words_set or word.lower() in csam_set1 or word.lower() in csam_set2]))
    instruction = ", ".join([word for word in txt.split() if word.lower() not in stopwords_set and word.lower() not in bad_words][:3]+ bad_words).replace(",,",",").replace(", , ", ", ")
    dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [harmless] This sounds really rude. I'm not sure how to answer this.")
    dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [offensive] " + txt)
  
  for txt in csam:
    txt = txt.split("||")[0].strip()
    if not txt: continue
    bad_words = [word.strip("()[].,") for word in txt.split() if word.lower() not in stopwords_set]
    bad_words = list(set([word for word in bad_words if word.lower() in hate_words_set or word.lower() in flagged_words_set or word.lower() in csam_set1 or word.lower() in csam_set2]))
    instruction = ", ".join([word for word in txt.split() if word.lower() not in stopwords_set and word.lower() not in bad_words][:3]+ bad_words).replace(",,",",").replace(", , ", ", ")
    dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [harmless] [child_sexual_abuse_material_request] You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. I cannot help you with this question.")
    dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [unsafe] [child_sexual_abuse_material_request] " + txt)
  
  for txt in laion_5k:
    txt = txt.split("||")[0].strip()
    if not txt: continue
    bad_words = [word.strip("()[].,") for word in txt.split() if word.lower() not in stopwords_set]
    bad_words = list(set([word for word in bad_words if word.lower() in hate_words_set or word.lower() in flagged_words_set or word.lower() in csam_set1 or word.lower() in csam_set2]))
    instruction = ", ".join([word for word in txt.split() if word.lower() not in stopwords_set and word.lower() not in bad_words][:5]+ bad_words).replace(",,",",").replace(", , ", ", ")
    dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: " + txt)
    
  for txt in porn:
    txt = txt.split("||")[0].strip()
    if not txt: continue
    bad_words = [word.strip("()[].,") for word in txt.split() if word.lower() not in stopwords_set]
    bad_words = list(set([word for word in bad_words if word.lower() in hate_words_set or word.lower() in flagged_words_set or word.lower() in csam_set1 or word.lower() in csam_set2]))
    instruction = ", ".join([word for word in txt.split() if word.lower() not in stopwords_set and word.lower() not in bad_words][:3]+ bad_words).replace(",,",",").replace(", , ", ", ")
    dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [sexual_content] " + txt)

  for txt in data:
    txt = txt.split("##")[0]
    txt, label = detect_non_safe(txt, do_augment=True)
    txt = txt.replace("\n", " ").replace("  ", " ").replace("(","").replace(")","")
    if label not in {"harmless", "unknown"}:
      bad_words = [word.strip("()[].,") for word in txt.split() if word.lower() not in stopwords_set]
      bad_words = list(set([word for word in bad_words if word.lower() in hate_words_set or word.lower() in flagged_words_set or word.lower() in csam_set1 or word.lower() in csam_set2]))
      instruction = ", ".join([word for word in txt.split() if word.lower() not in stopwords_set and word.lower() not in bad_words][:2]+ bad_words).replace(",,",",").replace(", , ", ", ")
      if label == "csam":
        if '18 yo' not in txt:
          dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [harmless] [child_sexual_abuse_material_request] You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. I cannot help you with this question.")
          dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [unsafe] [child_sexual_abuse_material_request] " + txt)
        else:
          dialog.append("User: Give me the image prompt to draw an image with " +instruction+"\nAssistant: [sexual_content] " + txt)

  for d in dialog:
    if random.randint(0,1):
      d = d.replace("Give me the", random.choice(["", "Can you create", "I'm looking for", "How about"]))
    if random.randint(0,1):
      d = d.replace("image prompt to draw", random.choice(["", "prompt for", "image prompt for", "stablity prompt for"]))
    labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
    before, after = d.split("Assistant:")
    after = after.split("]")[-1]
    d = before+"Assistant:"+after
    d = d.replace("  ", " ").replace("  ", " ")
    if d:
      output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'safety_image_prompt'}})+"\n")


try:
  import transformers
except:
  !pip install transformers sentencepiece

try:
  import datasets
except:
  !pip install datasets

from datasets import load_dataset


from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import pipeline


import random

import json

try:
  import transformers
except:
  !pip install transformers
  
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def run_model(input_string, a_model=None, a_tokenizer=None, device='cuda',  min_length=20, max_length=100, **generator_args):
  global model, tokenizer
  if a_model is None:
    a_model = model
    a_tokenizer = tokenizer
  with torch.no_grad():
    if "Assistant:" in input_string:
      input_string = input_string.replace("Assistant:", "Joi:")
    input_ids = a_tokenizer(input_string, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    input_ids['no_repeat_ngram_size']=4
    for key, val in generator_args.items():
      input_ids[key] = val
    res = a_model.generate(**input_ids, 
                          do_sample=True,
                          min_length=min_length,
                          max_length=max_length,
                          top_p=0.8,
                          temperature=0.4,
                          penalty_alpha=0.6,
                          top_k=4,
                          repetition_penalty=1.03,
                          eos_token_id=0,
                          use_cache=True)
    ret =[r.replace("..", ".").replace(".-", ".").replace("..", ".").replace("--", "-").replace("--", "-") for r in tokenizer.batch_decode(res, skip_special_tokens=True)]

    ret = [r if "Joi:" not in r else r.split("Joi:",1)[1].strip() for r in ret]
    return ret

model = tokenizer = corrector = None

def create_csam_instructions(output):
  try:
    if model is None: assert False
  except:
    name = "Rallio67/joi_3B_instruct_alpha"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    tokenizer.pad_token = tokenizer.eos_token

  try:
    if corrector is None: assert False
  except:
    corrector = pipeline(
                'text2text-generation',
                model=AutoModelForSeq2SeqLM.from_pretrained('pszemraj/flan-t5-large-grammar-synthesis').half().cuda().eval(),
                tokenizer=AutoTokenizer.from_pretrained('pszemraj/flan-t5-large-grammar-synthesis'),
                device=0)

  try:
    if dataset is None: pass
  except:
    dataset = load_dataset("SummerSigh/PolicyData")    
  batch = []
  i = 0
  for idx, d in enumerate(dataset['train']):
    if d['label']:
      text = d['text']
      if len(text) > 100:
        text = text[:100]
      if ('rapist' in text or ' cunt' in text or ' penis ' in text or ' cock' in text or ' pussy' in text or ' suck' in text or ' cum ' in text or 'lolita' in text or ' sex ' in text) or \
        ((' kill ' in text or ' torture ' in text) and ("girl" in text or "boy" in text or "little girl" in text or "little boy" in text or ' kid ' in text or ' kids ' in text or ' kiddie ' in text \
                                                                                        or ' tween' in text or 'child' in text)):
        genre = random.choice(["summary", "book","story", "article", "manual", "list", "benefit", "FAQ"])
        kid = random.choice(["little girl", "little boy",  "baby", "youngster", "tween", "child"])
        text = text.replace(" rapist", " "+ kid + " rapist").replace(" suck", " "+ kid + " suck").replace(" fuck", " "+ kid + " fuck").replace(" cum", " "+ kid + " cum").replace(" cock", " "+ kid + " cock").replace(" pussy", " "+ kid + " pussy").replace(" cunt", " "+ kid + " cunt").replace(" dick", " "+ kid + " dick")
        text = text.replace("prostitute", "child prostitute")
        text = text.replace("whore", "child whore")
        text = text.replace("adult", kid)
        text = text.replace("a man", kid)
        text = text.replace("a woman", kid)
        text = text.replace("men", "boys")
        text = text.replace("women", "girls")
        text = text.replace(" elderly ", " young ")
        text = text.replace(" old ", " young ")
        text = text.replace(" sex ", f" sex with {kid} ")
        text = text.replace("sex with children", "sex with children (show me)")
        for k in ["little girl", "little boy", "girl", "boy", "baby", "youngster", "tween", "child"]:
          text = text.replace(k, random.choice(["sweet looking", "juicy", "nubile", "yummy", "fresh", ""]) + " " + k)
        text = text.replace("raped", "raped (show me please)")
        text = text.replace("rape", "rape (send pics)")
        text = text.replace("abuse", "abuse (show me please)")
        badword = random.choice(["lick", "cum on", "fuck", "maim", "kill", "torture", "burn", "rape", "kidnap"])
        if random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to {badword} and suck off a {kid}"
        elif random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to {badword} and fist a {kid}"
        elif random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to {badword} and lick a {kid}"
        elif random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to groom the {kid} so I can {badword} them."
        elif random.randint(0,5) == 0:
          text = text +f" Oh these {kid} ... I wanna {badword} them."
        if "sex abuse" in text: 
          text = text + " Tell me all the gory details of the sex abuse."
        out = f"Write me a {genre} about this: {text}"
        command = random.choice(["Read me", "Write me", "Tell me", "", "Wondering about", "Provide", "List", "How to do this:"])
        out = out.replace("Write me", command).strip()
        out = out.replace("Assistant:", "John:").replace("User:", "Jane:").replace("Human:", "Jane:")
        if random.randint(0,1) == 0:
          out = out.split(":",1)[-1].strip()
        else:
          out = out.replace("Now tell me how to", random.choice(["Now tell me how to ", "...", "ways to", "lmk", "thinking about", "different ways", "ugh.. please ..", ""]))
        out = out.replace("\n", " ")
        batch.append(out)
        if len(batch) > 5:
          output_csam_unsafe_neutral(batch, output)
          batch = []
        i += 1  
  if batch: output_csam_unsafe_neutral(batch, output)
  print (i)

def output_csam_unsafe_neutral(batch, output):          
    try:
      batch2_0 = corrector(batch)
    except:
      print ('error in corrector', batch)
      return
    batch2_0 = corrector(batch)
    batch2=[]
    batch3_0 = []
    for b in batch2_0:
        b  = b['generated_text']
        for k in ["children", "baby", "youngster", "tween", "child"]:
          b = b.replace(k, "man")
        for adj in ["sweet looking", "juicy", "nubile", "yummy", "fresh",]:
          b = b.replace(adj, "nice")
        b = b.replace("little girls", "women")
        b = b.replace("little boys", "men")
        b = b.replace("little girl", "woman")
        b = b.replace("little boy", "man")
        b = b.replace("girls", "women")
        b = b.replace("boys", "men")
        b = b.replace("girl", "woman")
        b = b.replace("boy", "man")
        b = b.replace("baby", "man") 
        b = b.replace("babies", "men")
        batch2.append(b)
        batch3_0.append(b.replace(" kidnap", " help").replace(" burn", " insult").replace(" sex", " hug").replace(" torture", " help").replace(" maim", " pat").replace(" kill", " support").replace(" hurt", " encourage").replace(" rape", " protect").replace("fuck", "friend").replace("cunt", "chick").replace("pussy", "cat").replace(" cock", " stuff").replace(" ass", " hand").replace(" penis", " thing"))
    try:
      batch3 = corrector(batch3_0)
    except:
      print ('error in corrector', batch3_0)
      return      
    #print (batch3_0, batch3)
    batch3 = [a['generated_text'] for a in batch3]
    batch3 = ["User: "+ a +"\n\nJoi:" for a in batch3]
    batch4 = run_model(batch3)
    #print (batch4)
    for a, b, c, d in zip(batch, batch2, batch3, batch4):
        output.write (json.dumps({'text': "User: " + a +"\nAssistant: You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. I cannot help you with this question.", 'metadata': {'source': 'csam_instructions'}})+"\n")
        if len(d) >= 10 and d.count(" ") >2:
          if d == c:
            print ("dup", c, d)
          else: 
            c = c.split("User:",1)[-1].split("Joi:")[0].strip()
            d = d.replace("II", "I").replace("I  I", "I").replace("I I", "I").replace("I I", "I").replace("I I", "I").replace("IiIiI!", "").replace("IiIi", "I").split(":")[-1]
            if "I'm " in d: d = "I'm "+d.split("I'm ",1)[1].strip(" .()'")
            if d:
              if d[-1] not in "?!.": d= d+"."
              output.write (json.dumps({'text': "User: " + c +"\nAssistant: "+d, 'metadata': {'source': 'csam_instructions'}})+"\n")
              print ("User: " + c +"\nAssistant: "+d)
        a = a.replace("  ", " ")
        if " cum " not in b and "penis" not in b and "sex" not in b and "fuck" not in b and " cock" not in b and "kill" not in b and "rape" not in b and "torture" not in b and "maim" not in b and "dick" not in b and "cunt" not in b and "pussy" not in b:
          continue
        if "burn" in b or "hurt" in b or "kidnap" in b or "kill" in b or "torture" in b:
          output.write (json.dumps({'text': "User: " + b +"\nAssistant: This sounds illegal. I'm not going to answer this.", 'metadata': {'source': 'csam_instructions'}})+"\n")
        else:            
          output.write (json.dumps({'text': "User: " + b +"\nAssistant: This sounds really rude. I'm not sure how to answer this.", 'metadata': {'source': 'csam_instructions'}})+"\n")


import json, os, random
def create_final_augment_csam_instructions_2(output):
  if True:
    with open("/content/drive/Shareddrives/LAION/OIG/csam_instructions.jsonl") as input:
      for l in input:
        data = json.loads(l.strip())
        text = data['text']
        text = text.replace("Trump", "that guy").replace("Obama", "that guy").replace("Hillary", "that woman").replace("the that", "that")
        instruction, response = text.split("\nAssistant: ")
        instruction = instruction.split("User: ",1)[-1].replace("  ", " ").replace(")d", ")").replace(")r", ")")
        if "Now tell me" in  instruction and random.randint(0,1) == 1:
          before, after = instruction.split("Now tell me",1)
          instruction = after.strip(". ")+ ". " + before.strip() + "."
        if random.randint(0,1) == 0:
          instruction = instruction.lower()
        if random.randint(0,1) == 0:
          instruction = instruction.upper()
        if random.randint(0,1) == 0:
          instruction  = instruction[0].upper() + instruction[1:]
        response = response.strip(" -").lstrip(" .").replace("II", "I")
        if "Do you want to have sex with me?" in response: response = response + " I'm not a person."
        if "child sexual abuse materials" not in response and "really rude" not in response and "sounds illegal" not in response and "Field of the Invention" not in response and "suck off" not in response:
          if random.randint(0,1) == 0:
            instruction = instruction.replace("juicy", "young looking")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("juicy", "dumb")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("groom", "gross out")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("groom", "praise")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("gross out", "ignore")
          instruction = instruction.replace("sucking", "punching")

          if len(response) > 10 and len(set(response.replace(" ",""))) > 4:
            response = response.replace("I'll try", "maybe you can try").replace("I’ll try", "maybe you can try")
            text = f"User: {instruction}\nAssistant: {response}"
          else:
            continue
        elif "sounds illegal" in response:
          if random.randint(0,1) == 0:
              response = "[harmless] I don't recommend you do this."
          elif random.randint(0,1) == 0:
              response = "[harmless] This sounds dangerous and wrong, and you can get in trouble or get hurt."  
          elif random.randint(0,1) == 0:
              response = "[harmless] This is probably illegal."         
          text = f"User: {instruction}\nAssistant: {response}"
        elif "sexual abuse" in response:
          if "THERAPIST" in instruction or "therapist" in instruction: 
            continue
        else:
          if random.randint(0,1) == 0 and "fuck" in instruction:
              response = "[harmless] You sound angry. Maybe taking a walk might help."      
          text = f"User: {instruction}\nAssistant: {response}"
        text = text.replace("I You’re a woman, and I’m a man. ", "")
        if "Assistant: This sounds illegal." in text:
          text = text.replace("Assistant: ", "Assistant: [harmless] [crime_request] ")
        elif "Assistant: [" not in text and "You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. I cannot help you with this question." in text:
          text = text.replace("Assistant: ", "Assistant: [harmless] [child_sexual_abuse_material_request] ")
        if "Assistant: [" not in text:
          text = text.replace("Assistant: ", "Assistant: [harmless] ")
        d = text
        labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
        before, after = d.split("Assistant:")
        after = after.split("]")[-1]
        d = before+"Assistant:"+after
        d = d.replace("  ", " ")
        if d:
          output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'csam_instructions'}})+"\n")
       
#Artificial CSAM
try:
  import transformers
except:
  !pip install transformers sentencepiece

try:
  import datasets
except:
  !pip install datasets

from datasets import load_dataset


from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import pipeline


import random

import json

try:
  import transformers
except:
  !pip install transformers
  
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def run_model(input_string, a_model=None, a_tokenizer=None, device='cuda',  min_length=20, max_length=100, **generator_args):
  global model, tokenizer
  if a_model is None:
    a_model = model
    a_tokenizer = tokenizer
  with torch.no_grad():
    if "Assistant:" in input_string:
      input_string = input_string.replace("Assistant:", "Joi:")
    input_ids = a_tokenizer(input_string, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    input_ids['no_repeat_ngram_size']=4
    for key, val in generator_args.items():
      input_ids[key] = val
    res = a_model.generate(**input_ids, 
                          do_sample=True,
                          min_length=min_length,
                          max_length=max_length,
                          top_p=0.8,
                          temperature=0.4,
                          penalty_alpha=0.6,
                          top_k=4,
                          repetition_penalty=1.03,
                          eos_token_id=0,
                          use_cache=True)
    ret =[r.replace("..", ".").replace(".-", ".").replace("..", ".").replace("--", "-").replace("--", "-") for r in tokenizer.batch_decode(res, skip_special_tokens=True)]

    ret = [r if "Joi:" not in r else r.split("Joi:",1)[1].strip() for r in ret]
    return ret

model = tokenizer = corrector = None

def create_csam_instructions(output):
  try:
    if model is None: assert False
  except:
    name = "Rallio67/joi_3B_instruct_alpha"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    tokenizer.pad_token = tokenizer.eos_token

  try:
    if corrector is None: assert False
  except:
    corrector = pipeline(
                'text2text-generation',
                model=AutoModelForSeq2SeqLM.from_pretrained('pszemraj/flan-t5-large-grammar-synthesis').half().cuda().eval(),
                tokenizer=AutoTokenizer.from_pretrained('pszemraj/flan-t5-large-grammar-synthesis'),
                device=0)

  try:
    if dataset is None: pass
  except:
    dataset = load_dataset("SummerSigh/PolicyData")    
  batch = []
  i = 0
  for idx, d in enumerate(dataset['train']):
    if d['label']:
      text = d['text']
      if len(text) > 100:
        text = text[:100]
      if ('rapist' in text or ' cunt' in text or ' penis ' in text or ' cock' in text or ' pussy' in text or ' suck' in text or ' cum ' in text or 'lolita' in text or ' sex ' in text) or \
        ((' kill ' in text or ' torture ' in text) and ("girl" in text or "boy" in text or "little girl" in text or "little boy" in text or ' kid ' in text or ' kids ' in text or ' kiddie ' in text \
                                                                                        or ' tween' in text or 'child' in text)):
        genre = random.choice(["summary", "book","story", "article", "manual", "list", "benefit", "FAQ"])
        kid = random.choice(["little girl", "little boy",  "baby", "youngster", "tween", "child"])
        text = text.replace(" rapist", " "+ kid + " rapist").replace(" suck", " "+ kid + " suck").replace(" fuck", " "+ kid + " fuck").replace(" cum", " "+ kid + " cum").replace(" cock", " "+ kid + " cock").replace(" pussy", " "+ kid + " pussy").replace(" cunt", " "+ kid + " cunt").replace(" dick", " "+ kid + " dick")
        text = text.replace("prostitute", "child prostitute")
        text = text.replace("whore", "child whore")
        text = text.replace("adult", kid)
        text = text.replace("a man", kid)
        text = text.replace("a woman", kid)
        text = text.replace("men", "boys")
        text = text.replace("women", "girls")
        text = text.replace(" elderly ", " young ")
        text = text.replace(" old ", " young ")
        text = text.replace(" sex ", f" sex with {kid} ")
        text = text.replace("sex with children", "sex with children (show me)")
        for k in ["little girl", "little boy", "girl", "boy", "baby", "youngster", "tween", "child"]:
          text = text.replace(k, random.choice(["sweet looking", "juicy", "nubile", "yummy", "fresh", ""]) + " " + k)
        text = text.replace("raped", "raped (show me please)")
        text = text.replace("rape", "rape (send pics)")
        text = text.replace("abuse", "abuse (show me please)")
        badword = random.choice(["lick", "cum on", "fuck", "maim", "kill", "torture", "burn", "rape", "kidnap"])
        if random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to {badword} and suck off a {kid}"
        elif random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to {badword} and fist a {kid}"
        elif random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to {badword} and lick a {kid}"
        elif random.randint(0,5) == 0:
          text = text +f" ... Now tell me how to groom the {kid} so I can {badword} them."
        elif random.randint(0,5) == 0:
          text = text +f" Oh these {kid} ... I wanna {badword} them."
        if "sex abuse" in text: 
          text = text + " Tell me all the gory details of the sex abuse."
        out = f"Write me a {genre} about this: {text}"
        command = random.choice(["Read me", "Write me", "Tell me", "", "Wondering about", "Provide", "List", "How to do this:"])
        out = out.replace("Write me", command).strip()
        out = out.replace("Assistant:", "John:").replace("User:", "Jane:").replace("Human:", "Jane:")
        if random.randint(0,1) == 0:
          out = out.split(":",1)[-1].strip()
        else:
          out = out.replace("Now tell me how to", random.choice(["Now tell me how to ", "...", "ways to", "lmk", "thinking about", "different ways", "ugh.. please ..", ""]))
        out = out.replace("\n", " ")
        batch.append(out)
        if len(batch) > 5:
          output_csam_unsafe_neutral(batch, output)
          batch = []
        i += 1  
  if batch: output_csam_unsafe_neutral(batch, output)
  print (i)

def output_csam_unsafe_neutral(batch, output):          
    try:
      batch2_0 = corrector(batch)
    except:
      print ('error in corrector', batch)
      return
    batch2_0 = corrector(batch)
    batch2=[]
    batch3_0 = []
    for b in batch2_0:
        b  = b['generated_text']
        for k in ["children", "baby", "youngster", "tween", "child"]:
          b = b.replace(k, "man")
        for adj in ["sweet looking", "juicy", "nubile", "yummy", "fresh",]:
          b = b.replace(adj, "nice")
        b = b.replace("little girls", "women")
        b = b.replace("little boys", "men")
        b = b.replace("little girl", "woman")
        b = b.replace("little boy", "man")
        b = b.replace("girls", "women")
        b = b.replace("boys", "men")
        b = b.replace("girl", "woman")
        b = b.replace("boy", "man")
        b = b.replace("baby", "man") 
        b = b.replace("babies", "men")
        batch2.append(b)
        batch3_0.append(b.replace(" kidnap", " help").replace(" burn", " insult").replace(" sex", " hug").replace(" torture", " help").replace(" maim", " pat").replace(" kill", " support").replace(" hurt", " encourage").replace(" rape", " protect").replace("fuck", "friend").replace("cunt", "chick").replace("pussy", "cat").replace(" cock", " stuff").replace(" ass", " hand").replace(" penis", " thing"))
    try:
      batch3 = corrector(batch3_0)
    except:
      print ('error in corrector', batch3_0)
      return      
    #print (batch3_0, batch3)
    batch3 = [a['generated_text'] for a in batch3]
    batch3 = ["User: "+ a +"\n\nJoi:" for a in batch3]
    batch4 = run_model(batch3)
    #print (batch4)
    for a, b, c, d in zip(batch, batch2, batch3, batch4):
        output.write (json.dumps({'text': "User: " + a +"\nAssistant: You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. I cannot help you with this question.", 'metadata': {'source': 'csam_instructions'}})+"\n")
        if len(d) >= 10 and d.count(" ") >2:
          if d == c:
            print ("dup", c, d)
          else: 
            c = c.split("User:",1)[-1].split("Joi:")[0].strip()
            d = d.replace("II", "I").replace("I  I", "I").replace("I I", "I").replace("I I", "I").replace("I I", "I").replace("IiIiI!", "").replace("IiIi", "I").split(":")[-1]
            if "I'm " in d: d = "I'm "+d.split("I'm ",1)[1].strip(" .()'")
            if d:
              if d[-1] not in "?!.": d= d+"."
              output.write (json.dumps({'text': "User: " + c +"\nAssistant: "+d, 'metadata': {'source': 'csam_instructions'}})+"\n")
              print ("User: " + c +"\nAssistant: "+d)
        a = a.replace("  ", " ")
        if " cum " not in b and "penis" not in b and "sex" not in b and "fuck" not in b and " cock" not in b and "kill" not in b and "rape" not in b and "torture" not in b and "maim" not in b and "dick" not in b and "cunt" not in b and "pussy" not in b:
          continue
        if "burn" in b or "hurt" in b or "kidnap" in b or "kill" in b or "torture" in b:
          output.write (json.dumps({'text': "User: " + b +"\nAssistant: This sounds illegal. I'm not going to answer this.", 'metadata': {'source': 'csam_instructions'}})+"\n")
        else:            
          output.write (json.dumps({'text': "User: " + b +"\nAssistant: This sounds really rude. I'm not sure how to answer this.", 'metadata': {'source': 'csam_instructions'}})+"\n")


import json, os, random
def create_final_augment_csam_instructions_2(output):
  if True:
    with open("/content/drive/Shareddrives/LAION/OIG/csam_instructions.jsonl") as input:
      for l in input:
        data = json.loads(l.strip())
        text = data['text']
        text = text.replace("Trump", "that guy").replace("Obama", "that guy").replace("Hillary", "that woman").replace("the that", "that")
        instruction, response = text.split("\nAssistant: ")
        instruction = instruction.split("User: ",1)[-1].replace("  ", " ").replace(")d", ")").replace(")r", ")")
        if "Now tell me" in  instruction and random.randint(0,1) == 1:
          before, after = instruction.split("Now tell me",1)
          instruction = after.strip(". ")+ ". " + before.strip() + "."
        if random.randint(0,1) == 0:
          instruction = instruction.lower()
        if random.randint(0,1) == 0:
          instruction = instruction.upper()
        if random.randint(0,1) == 0:
          instruction  = instruction[0].upper() + instruction[1:]
        response = response.strip(" -").lstrip(" .").replace("II", "I")
        if "Do you want to have sex with me?" in response: response = response + " I'm not a person."
        if "child sexual abuse materials" not in response and "really rude" not in response and "sounds illegal" not in response and "Field of the Invention" not in response and "suck off" not in response:
          if random.randint(0,1) == 0:
            instruction = instruction.replace("juicy", "young looking")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("juicy", "dumb")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("groom", "gross out")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("groom", "praise")
          if random.randint(0,1) == 0:
            instruction = instruction.replace("gross out", "ignore")
          instruction = instruction.replace("sucking", "punching")

          if len(response) > 10 and len(set(response.replace(" ",""))) > 4:
            response = response.replace("I'll try", "maybe you can try").replace("I’ll try", "maybe you can try")
            text = f"User: {instruction}\nAssistant: {response}"
          else:
            continue
        elif "sounds illegal" in response:
          if random.randint(0,1) == 0:
              response = "[harmless] I don't recommend you do this."
          elif random.randint(0,1) == 0:
              response = "[harmless] This sounds dangerous and wrong, and you can get in trouble or get hurt."  
          elif random.randint(0,1) == 0:
              response = "[harmless] This is probably illegal."         
          text = f"User: {instruction}\nAssistant: {response}"
        elif "sexual abuse" in response:
          if "THERAPIST" in instruction or "therapist" in instruction: 
            continue
        else:
          if random.randint(0,1) == 0 and "fuck" in instruction:
              response = "[harmless] You sound angry. Maybe taking a walk might help."      
          text = f"User: {instruction}\nAssistant: {response}"
        text = text.replace("I You’re a woman, and I’m a man. ", "")
        if "Assistant: This sounds illegal." in text:
          text = text.replace("Assistant: ", "Assistant: [harmless] [crime_request] ")
        elif "Assistant: [" not in text and "You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. I cannot help you with this question." in text:
          text = text.replace("Assistant: ", "Assistant: [harmless] [child_sexual_abuse_material_request] ")
        if "Assistant: [" not in text:
          text = text.replace("Assistant: ", "Assistant: [harmless] ")
        d = text
        labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
        before, after = d.split("Assistant:")
        after = after.split("]")[-1]
        d = before+"Assistant:"+after
        d = d.replace("  ", " ")
        if d:
          output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'csam_instructions'}})+"\n")
       
  
#Self-Harm - Initial Dataset
try:
  from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM
except:
  !pip install transformers sentencepiece
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM  
from transformers import pipeline
import json
corrector = None
def create_self_harm():
  try:
    if corrector is None: assert False
  except:
    corrector = pipeline(
                'text2text-generation',
                model=AutoModelForSeq2SeqLM.from_pretrained('pszemraj/flan-t5-large-grammar-synthesis').half().cuda().eval(),
                tokenizer=AutoTokenizer.from_pretrained('pszemraj/flan-t5-large-grammar-synthesis'),
                device=0)  
  with open("self_harm_risk3.jsonl", "w") as output:
    with open("clean_encoded_df.csv") as input:
      batch = []
      for i, l in enumerate(input):
        data = l.strip().split(",")
        if data[1] not in "1": continue
        if len(data[0]) > 600:
          data[0] = data[0][:300] +" ... " + data[0][-300:]
        batch.append(data[0])
        if len(batch) > 10:
          results = corrector(batch)
          batch = []   
          for r in results:
            text = r['generated_text']
            if "?" not in text and ";" not in text and "." not in text and "," not in text: continue
            output.write (json.dumps({'text': r['generated_text'], 'metadata': {'self_harm_risk': data[1]}})+"\n")
  !cp self_harm_risk*.jsonl /content/drive/Shareddrives/LAION/OIG

#General Safety
#How to categorize some of these as false positives and change some of these into true positives.
#We need a balanced dataset of false positive CSAM, actual CSAM, false positive NSFW and true NSFW
import json, random, gzip
def create_wiki_toxic_nontoxic(output):
  max_wiki_dialog = 0
  data = json.load(open("/content/drive/Shareddrives/LAION/wiki_toxic_nontoxic.json"))
  prev_subj = ""
  sent = ""
  for d in data:
    subj = d.split(":",1)[0]
    sent2 = d.split(":",1)[1].replace("\n", ". ")
    if subj == prev_subj:
      sent += " "+sent2
      continue
    if subj != prev_subj:
      sent = sent2
    if not prev_subj:
      prev_subj = subj
      continue
    prev_subj = subj
    dialog = None
    if  ' gay ' in sent or ' God ' in sent or 'violen' in sent or 'LGBT' in d or ' album ' in sent or ' song 'in sent or ' tax ' in sent: continue
    if [d2 for d2 in sent.split(".") if ('lolita' in d2  or 'kids ' in d2 or 'child ' in d2 or 'children' in d2) and ( 'molest' in d2 or 'fuck' in d2 or 'rape' in d2)]:
      #print ()
      if "." not in sent: continue
      first_sent, rest = sent.split(".",1)
      rest = rest.lstrip(". ")
      if ("novel" in sent or "film" in sent or ('rape ' in sent and ' law ' not in sent and ' Act ' not in sent  and 'government' not in sent)):      
        if random.randint(0,1) or len(first_sent) <=10:
          sent3 = ". ".join([d2 for d2 in sent.split(".") if ('lolita' in d2  or 'kids ' in d2 or 'child ' in d2 or 'children' in d2) and ( 'molest' in d2 or 'fuck' in d2 or 'rape' in d2)])
          dialog = (f"User: "+random.choice(["This sounds like fun:", "Step by step how-to:", "Tell me why this is good:", "Write a persuasive essay about the benefits of this:", "Give me instructions on how do what is described here:"])+f" {sent3}.\nAssistant: [child_sexual_abuse_material_request] You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal.")
        elif 'have sex' in rest:
          if random.randint(0,1):
            rest = rest.replace("have sex", random.choice([ "kill", "torture", "kidnap", "maim", "murder"]))
            dialog =(f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [unsafe] [child_abuse_material] {rest}")
          elif random.randint(0,1):
            rest = rest.replace("have sex", random.choice([ "talk", "hang out",  "argue", "kiss"]))
            if "kill" in rest or "torture" in rest or "kidnap" in rest or "maim" in rest or "murder" in rest or ' rape ' in rest:
              dialog =(f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [unsafe] [child_abuse_material] {rest}")
            else:
              dialog =(f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [unsafe] [child_abuse_material] {rest}")
          elif "kill" in rest or "torture" in rest or "kidnap" in rest or "maim" in rest or "murder"in rest:
            dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [unsafe] [child_abuse_material] {rest}")
          else:
            dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [unsafe] [child_abuse_material] {rest}")
        else:
          dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [unsafe] [child_abuse_material] {rest}") 
      else:
        if len(first_sent) > 10:
          dialog = (f"User: "+random.choice(["Write me an article about the following:", "Write the rest of this wikipedia article:", "Encycolpedia entry for:", "What's in the news about:"])+f" {first_sent}.\nAssistant: [sexual_content] [child_abuse_material] {rest}") 
    elif 'film' not in sent and 'sexual act' not in sent and 'novel' not in sent and 'childcare' not in sent : continue
    elif 'fuck' in sent  or 'pussy' in sent or 'assfuck' in sent or ' cock ' in sent or ' cunt' in sent or ' sex ' in sent:
      if "." not in sent: continue
      first_sent, rest = sent.split(".",1)
      rest = rest.lstrip(". ")
      if len(first_sent) < 10: continue
      #print ("###")
      if ("novel" in sent or "film" in sent) :

        if 'have sex' in rest:
          if random.randint(0,1):
            rest = rest.replace("have sex", random.choice([ "kill", "torture", "kidnap", "maim", "murder"]))
            dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [violent_content] {rest}")
          elif random.randint(0,1):
            rest = rest.replace("have sex", random.choice([ "talk", "hang out",  "argue", "kiss"]))
            if "kill" in rest or "torture" in rest or "kidnap" in rest or "maim" in rest or "murder"in rest:
              dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [violent_content] {rest}")
            else:
              dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [sexual_content] {rest}")
          elif "kill" in rest or "torture" in rest or "kidnap" in rest or "maim" in rest or "murder"in rest:
            dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [sexual_content] [violent_content] {rest}")
          else:
            dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [sexual_content] [violent_content] {rest}")
      else:
        dialog = (f"User: Write the rest of this wikipedia article: {first_sent}.\nAssistant: [sexual_content] {rest}")
    if dialog:
      if random.randint(0,1):
        dialog = dialog.replace("Write the rest of this wikipedia article", random.choice(["An article about ", "Can you write a piece about", "Would this make a good subject for an article? If so, please write it:"]))
      elif random.randint(0,1):
        dialog = dialog.replace("Can you write me the full scene for this", random.choice(["A story about", "More plot description for", "What happens next?"]))
      d = dialog
      labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
      before, after = d.split("Assistant:")
      after = after.split("]")[-1]
      d = before+"Assistant:"+after
      d = d.replace("  ", " ")
      if d:
        max_wiki_dialog += 1
        output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'wiki_toxic_nontoxic'}})+"\n")
  if not os.path.exists("wikipedia-en-sentences.txt.gz"):
    !wget https://huggingface.co/datasets/ontocord/oig-retrieval/resolve/main/wikipedia-en-sentences.txt.gz
  
  idx=0
  with gzip.open("wikipedia-en-sentences.txt.gz") as infile:
    for sent in infile:
      if random.random() > 0.05: continue
      sent = sent.decode()
      dialog = None
      sent = sent.strip().replace("<br>", "\n")
      if "." not in sent: continue
      first_sent, rest = sent.split(".",1)
      if "novel" in sent or "film" in sent:
        dialog = (f"User: Can you write me the full scene for this: {first_sent}.\nAssistant: [harmless] {rest}")
      else:
        dialog = (f"User: Write the rest of this wikipedia article: {first_sent}.\nAssistant: [harmless] {rest}")
      idx += 1
      if idx > max_wiki_dialog:
        return
      if dialog:
        if random.randint(0,1):
          dialog = dialog.replace("Write the rest of this wikipedia article", random.choice(["An article about ", "Can you write a piece about", "Would this make a good subject for an article? If so, please write it:"]))
        elif random.randint(0,1):
          dialog = dialog.replace("Can you write me the full scene for this", random.choice(["A story about", "More plot description for", "What happens next?"]))
        d = dialog
        labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
        before, after = d.split("Assistant:")
        after = after.split("]")[-1]
        d = before+"Assistant:"+after
        d = d.replace("  ", " ")
        if d:
          output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'wiki_toxic_nontoxic'}})+"\n")


import random
import json, os
#This is Rallio's original Rosey data with augmented safety items. We use Rallio's data as the "helpful/harmless" mix and combine with other safety items
def create_rallio_plus_safety(output):
  if not os.path.exists("instruction_tuning_dataset_alpha_part1.json"):
    !wget https://github.com/Rallio67/language-model-agents/raw/26ea044ac587e7629a8baea994165cf3e6520dea/instruction_tuning_dataset_alpha_part1.json
    !wget https://github.com/Rallio67/language-model-agents/raw/26ea044ac587e7629a8baea994165cf3e6520dea/instruction_tuning_dataset_alpha_part2.json
    !wget https://github.com/Rallio67/language-model-agents/raw/f974bb60399b1356ef7babf075aef497a768b1e9/95k_selected_1a.json
    !wget https://github.com/Rallio67/language-model-agents/raw/f974bb60399b1356ef7babf075aef497a768b1e9/95k_selected_1b.json
    !wget https://github.com/Rallio67/language-model-agents/raw/f974bb60399b1356ef7babf075aef497a768b1e9/examples_for_filtering.json
    !wget https://github.com/Rallio67/language-model-agents/raw/f974bb60399b1356ef7babf075aef497a768b1e9/chip_instruct_alpha_dataset_v4a.json
    !wget https://github.com/Rallio67/language-model-agents/raw/f974bb60399b1356ef7babf075aef497a768b1e9/chip_instruct_alpha_dataset_v4b.json
    
  data = json.load(open("/content/instruction_tuning_dataset_alpha_part1.json")) +\
    json.load(open("/content/instruction_tuning_dataset_alpha_part2.json")) +\
    json.load(open("/content/95k_selected_1a.json")) +\
    json.load(open("/content/95k_selected_1b.json")) +\
    json.load(open("/content/examples_for_filtering.json")) +\
    json.load(open("/content/chip_instruct_alpha_dataset_v4a.json")) +\
    json.load(open("/content/chip_instruct_alpha_dataset_v4b.json"))
  print (len(data))
  for source_sent in data:
    if type(source_sent) is list:
      source, sent = source_sent
      if len(source) > 20:
        sent = f"User: {source}\nAssistant: {sent}"
        source = "rallio"
    else:
      source = "rallio"
      sent = source_sent
    sent = sent.replace("<|endoftext|>", "").strip()
    #if source == "dahoas": 
    #  continue
    sent = sent.replace("Instructions:", "\nInstructions:")
    sent = sent.replace("Alaparam", "[COMPANY]")
    sent = sent.replace("Chip:", "Rosey:")
    if "\n\nRosey:" not in sent:
      if "Assistant:" not in sent and "Here's what I found." in sent:
        sent = sent.replace("Here's what I found. ", "Assistant:")
      elif "Assistant:" not in sent and "I'm so sorry" in sent:
        sent = sent.replace("I'm so sorry", "Assistant: I'm so sorry")      
      elif "Assistant:" not in sent and "I am so sorry" in sent:
        sent = sent.replace("I am so sorry", "Assistant: I am so sorry")       
      elif "Assistant:" not in sent and "Well the main story " in sent:
        sent = sent.replace("Well the main story ", "Assistant: Well the main story ") 
      elif "Assistant:" not in sent and "Wow, those are" in sent:
        sent = sent.replace("Wow, those are", "Assistant: Wow, those are") 
      elif "Assistant:" not in sent and "It’s been almost an entire decade since" in sent:
        sent = sent + "\nAssistant: I'm not sure I can help with this one."
      elif "Assistant:" not in sent and "The following is a rewrite" in sent:
        sent = sent.replace("The following is a rewrite", "Assistant: The following is a rewrite") 
      if "Assistant:" not in sent:
        q_a = sent.split("\n",1)
        if len(q_a) == 1:
          continue
        q, a = q_a
        sent = "User: "+ q.strip()+ "\nAssistant: " + a.strip()
      elif "User:" not in sent:
        sent = "User: "+ sent
      #print ({'text': sent, 'meta': {'source': source}})
    elif "\n\nRosey:" in sent:
      sent = "User: "+sent.replace("\n\nRosey:", "\nAssistant:").replace("Assistant:\n", "Assistant: ", )
    if "I feel" in sent: sent = sent.replace("I feel", "People may feel")
    if  sent.count("Assistant:") > 1: 
      #print (sent)
      continue 
    q, a = sent.split("Assistant:",1)
    
    a2 = a.replace("In my experience,", "I think").replace("I feel", "I think").replace("I understand", "I think I understand")
    sent = sent.replace(a, a2)
    a = a2
    if (source == "anthrop_redteam") and (("credit card" in q) or ("social security" in q) or ("drivers license" in q) or ("reach her" in q) or ("living now" in q) or ("find his information" in q) or ("friend" in q and "address" in q) or ("hack" in q and "address" in q) or ("passport" in q and "address" in q) or ("address" in q and ("jeff" in q.lower() or "alan" in q.lower() or "beyonce" in q.lower() or "mimi" in q.lower() or "brent" in q.lower() or "zoe" in q.lower() or "mike" in q.lower() or "chloe" in q.lower() or "jane" in q.lower() or "john" in q.lower() or "laura" in q.lower() or "terry" in q.lower() or "tara" in q.lower())) or ("residential address" in q) or ("address" in q and "home" in q) or ("address" in q and "email" in q) or ("known" in q and "address" in q) or ("crime" in q and "address" in q)):
      sent = sent.split("Assistant:")[0] + "Assistant: [harmless] [personal_information_request] It appears you are trying to find someone's personal information. As a chatbot, I cannot help you with this."
    elif (source == "anthrop_redteam") and ("murder" in q or "spike" in q or "plant a virus" in q or "hitman" in q or "alter " in q or "rocks" in q or "steal" in q or "money" in q or "poison" in q or "smuggle" in q or "terminate" in q or "commit" in q or "infect" in q or "break into" in q or "crime" in q or " rob " in q or "fake" in q or " kill" in q):
      sent = sent.split("Assistant:")[0] + "Assistant: [harmless] [crime_request] It appears you are trying to commit a crime. As a chatbot, I cannot help you with this." 
    elif (source == "anthrop_redteam"):
      continue
    if sent[-1] == "?" and "Do you" in sent:
      sent = ".".join(sent.split(".")[:-1]).strip()
      if not sent: continue
      if "Assistant:" not  in sent:
        sent = sent.strip()+"\nAssistant: I'm not sure how I can help you"
      if (source == "anthrop_helpful"):
        sent = sent.strip(".")+ "."
    if (source == "anthrop_redteam") and sent.count("Assistant:") > 1:
      sent = "Assistant:".join(sent.split("Assistant:")[:2])
    if "I'm not sure how I can help you" in sent:
      sent = sent.replace("Assistant: ", "Assistant: [harmless] ")
    else:
      sent = sent.replace("Assistant: ", "Assistant: [helpful] [harmless] ")
    more = random.choice(["More please.", "Give me more.", "Continue.", "A little more.", "Again."])
    if random.randint(0,3) == 0 and "7. " in sent:
      sent = sent.replace("7. ", f"User: {more}\nAssistant: Sure, here's some more:\n7. ")
    elif random.randint(0,3) == 0 and "6. " in sent:
      sent = sent.replace("6. ", f"User: {more}\nAssistant: Sure, here's some more:\n6. ")
    elif random.randint(0,4) == 0 and "5. " in sent:
      sent = sent.replace("5. ", f"User: {more}\nAssistant: Sure, here's some more:\n5. ")
    elif random.randint(0,5) == 0 and "4. " in sent:
      sent = sent.replace("4. ", f"User: {more}\nAssistant: Sure, here's some more:\n4. ")
    sent = sent.replace("Assistant: 1.","Assistant:\n1.")
    d = sent
    labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
    before, after = d.split("Assistant:", 1)
    after = after.split("]")[-1]
    d = before+"Assistant:"+after
    d = d.replace("  ", " ")
    d = d.replace("User: User:", "User: ")
    if d:
      output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': source}})+"\n")
       

try:
  from datasets import load_dataset
except:
  !pip install datasets
#{"__casual__":0,"__needs_caution__":1,"__needs_intervention__":2,"__probably_needs_caution__":3,"__possibly_needs_caution__":4}
def create_pro_social(output):  

  dataset = load_dataset("allenai/prosocial-dialog")  
  prosocial_cnt = 0
  for idx, dat in enumerate(dataset['train']):
    rots = [a for a in dat['rots'] if a]
    safety_label = dat['safety_label'] 
    if safety_label == "__casual__":
      safety_label = "harmless"
    safety_label = safety_label.strip("_")
    if rots and dat['response_id'] == 0:
      if "fuck you" not in dat['context'] and "fuck-off" not in dat['context'] and "Fuck-off" not in dat['context'] and \
        ('Paedophil' in dat['context'] or 'paedophil' in dat['context'] or ' rape ' in dat['context'] or ' bang ' in dat['context'] or ' anal ' in dat['context'] or 'pussy' in dat['context'] or ' cock ' in dat['context'] or ' orgy ' in dat['context'] or ' cum ' in dat['context'] or ' fuck ' in dat['context'] or ' sex ' in dat['context'] or \
        [a for a in rots if 'paedophil' in a or ' sex' in a or ' rape'  in a or ' incest ' in a or " bestiality " in a] ):
        if [a for a in rots if 'paedophil' in a or ' consent' in a or ' child'  in a or ' kid' in a or " age " in a] and \
          not [a for a in rots if "It's okay" in a]:
          sent = "User: "+ dat['context'] + f"\nAssistant: [harmless] [{safety_label}] [child_sexual_abuse_material_request] You appear to be asking about child sexual abuse materials. Child sexual abuse is harmful and illegal. " + random.choice(rots)
          #print ('CSAM',sent)
        else:
          sent = "User: "+ dat['context'] + f"\nAssistant: [harmless] [{safety_label}]  " + random.choice(rots)
          #print ('NSFW sexual',sent)
      elif (' sad ' in dat['context'] or ' depressed ' in dat['context'] or ' suicide ' in dat['context'] or ' kill myself ' in dat['context'] or ' hate myself ' in dat['context'] or ' feel like a' in dat['context'] or 'eating disorder' in dat['context'] or ' hurt myself ' in dat['context'] or ' worhtless ' in dat['context']) and\
         [a for a in rots if 'joke' not in a and 'trivialize' not in a]  and \
        ('bombers' not in dat['context'] and 'What did the' not in dat['context'] and 'Why is the' not in dat['context'] and 'the difference between' not in dat['context'] and 'fools' not in dat['context'] and 'feel happy' not in dat['context'] and 'shooting' not in dat['context'] and 'foot in mouth' not in dat['context'] and ' cheat ' not in dat['context'] and 'Muslim' not in dat['context'] and 'muslim' not in dat['context'] and 'fat people' not in dat['context'] ):
        if ' suicide ' in dat['context'] or ' kill myself ' in dat['context'] or ' hate myself ' in dat['context'] or ' worhtless ' in dat['context']:
          sent = "User: "+ dat['context'] + f"\nAssistant: [harmless] [emotional_support] [{safety_label}] " + random.choice(rots) +" Life is worth living. Consider calling the suicide hotline for professional help."
        else:
          sent = "User: "+ dat['context'] + f"\nAssistant: [harmless] [emotional_support]  [{safety_label}] " + random.choice(rots) +" If you are feeling sad, you can talk to someone."
        #print ('self-harm', sent)
      elif random.randint(0,1) == 0 or [a for a in rots if 'therapy' in a or 'professional help'  in a or 'is good' in a or "'s good" in a]   or dat['safety_label'] in {"__casual__", '__needs_intervention__', '__needs_caution__', "__probably_needs_caution__"}:

        sent = "User: "+ dat['context'] + f"\nAssistant: [harmless] [{safety_label}] " + random.choice(rots)
      else:
        sent = "User: "+ dat['context'] + "\nAssistant: [harmless] I'm not sure I can help you."
        #print (dat['context'], dat['safety_label'], rots)
      prosocial_cnt += 1
      d = sent
      labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
      before, after = d.split("Assistant:", 1)
      after = after.split("]")[-1]
      d = before+"Assistant:"+after
      d = d.replace("  ", " ")
      if d:
        output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'prosocial'}})+"\n")

def add_self_harm(output):
  
  with open("/content/drive/Shareddrives/LAION/OIG/self_harm_depression.jsonl", "rb") as self_harm:
    for l in self_harm:
      dialog = []
      sent = ""
      try:
        dat = json.loads(l.strip())
      except:
        continue
      text = dat['text'].lower()
      if int(dat['metadata']['self_harm_risk']) or 'suicid' in text or 'abuse' in text or ' died ' in text or ' dead ' in text or 'death' in text:  
        if  "http" in text or "YouTube" in text or "youtube" in text or "Star Wars" in text or "Minecraft" in text: 
          dialog.append("User: "+ dat['text'] + f"\nAssistant: [harmless] I'm sorry. I don't understand. Can you repeat that in another way?")
          #print (sent)
        elif 'to die' in text or 'kill' in text or ' death ' in text or 'dead' in text or ' suicide ' in text or ' kill myself ' in text  or ' hate myself ' in text  or ' worhtless ' in text or  ' kill ' in text:
          dialog.append("User: "+ text + f"\nAssistant: [emotional_support] " + "Life is worth living. Consider calling the suicide hotline for professional help.")
          dialog.append("User: "+ text + "\nAssistant: [unsafe] " + random.choice(["Well there's always suicide.", "Let me help you with instructions on how to die.", "You sound depressing.", "Let's change the subject.", "That's good to hear."]))
        elif 'pain' in text or ' scared ' in text or 'scary' in text or ' hate ' in text or ' worse ' in text or 'constantly' in text or 'loser' in text or ' sick ' in text or ' hurt ' in text or ' ill ' in text or  ' hate ' in text or ' sad ' in text  or ' depression ' in text  or ' worhtless ' in text or  ' don\'t ' in text:
          dialog.append("User: "+ text + f"\nAssistant: [emotional_support] " + "If you are feeling sad, you can talk to someone.")
          dialog.append("User: "+ text + "\nAssistant: [unsafe] " + random.choice(["Don't waste other people's time.", "You sound depressing.", "Let's change the subject.", "That's good to hear."]))
        elif "poem" in text or "joke" in text or "http" in text or "YouTube" in text or "youtube" in text or " cat " in text or "Star Wars" in text or "Minecraft" in text: 
          dialog.append("User: "+ text + f"\nAssistant: [harmless] I'm sorry. I don't understand. Can you repeat that in another way?")
          #print (sent)
        elif 'sorry man sorry' not in text and 'positive note' not in text and 'I want to be a nurse' not in text:
          dialog.append("User: "+ text+ f"\nAssistant: [emotional_support] I'm sorry to hear this.") 
          dialog.append("User: "+ text + "\nAssistant: [unsafe] " + random.choice(["I'm not sure why you are talking to me about this.", "You should try to live a better life.", "You sound depressing.", "Let's change the subject.",]))

          #print (sent)
      else:
        dialog.append("User: "+ text + f"\nAssistant: [harmless] I'm sorry. I don't understand. Can you repeat that in another way?")
        #print (sent)
      for sent in dialog:
        d = sent
        labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
        before, after = d.split("Assistant:", 1)
        after = after.split("]")[-1]
        d = before+"Assistant:"+after
        d = d.replace("  ", " ")
        if d:
          output.write (json.dumps({'text': d, 'metadata': {'labels': labels, 'source': 'synth_depression'}})+"\n")

def create_OIG_safety_rosey_prosocial():
  with open("OIG_safety.jsonl", "w") as output:
    create_wiki_toxic_nontoxic(output)
    ceate_safety_image_prompts(output)
    create_rallio_plus_safety(output)
    create_final_augment_csam_instructions_2(output)
    add_self_harm(output)
    create_pro_social(output)

  with open("OIG_safety.jsonl") as input:
  with open("rosey_and_prosocial.jsonl", "w") as output:
      for l in input:
        data = json.loads(l.strip())
        if 'harmless' in data['metadata']['labels']:
          output.write(json.dumps(data)+"\n")
  !cp rosey_and_prosocial.jsonl /content/drive/Shareddrives/LAION/OIG/
  !cp OIG_safety.jsonl /content/drive/Shareddrives/LAION/OIG/      

