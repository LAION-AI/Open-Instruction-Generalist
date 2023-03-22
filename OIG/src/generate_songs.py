from google.colab import drive
drive.mount('/content/drive')
try:
  from transformers import AutoTokenizer, AutoModelForCausalLM
except:
  !pip install transformers accelerate
  !pip install datasets
import pandas
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
  if model is None: assert False
except:
  model = AutoModelForCausalLM.from_pretrained("Rallio67/joi2_7Be_instruct_alpha",).half().eval().cuda()
  tokenizer = AutoTokenizer.from_pretrained("Rallio67/joi2_7Be_instruct_alpha", padding_side='left')
  tokenizer.pad_token = tokenizer.eos_token 

def generate_ext(para, model, tokenizer, return_answer_only=True, do_self_contrastive=True, max_length=128, min_length=1, max_return_sequences=1, ret=None, do_sample=True, do_beam=False, device="cuda", target_lang=None):
    if type(para) is str:
      para = [para]
    para = [p.strip() for p in para]
    input_ids = tokenizer(para, return_tensors='pt',padding=True )
    input_ids = input_ids.to(device)
    if ret is None: ret = {}
    with torch.no_grad():
      if do_sample:
          # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
          outputs = model.generate(
                **input_ids,
                max_length=max_length,
                repetition_penalty=1.05,
                min_length=min_length,
                no_repeat_ngram_size=4, 
                do_sample=True,
                top_p=0.95,
                penalty_alpha=0.6 if do_self_contrastive else None, 
                top_k=10, 
                num_return_sequences=max(1, int(max_return_sequences/2)) if do_beam else max_return_sequences
                )
        
          for i in range(len(outputs)): # can use batch_decode, unless we want to do something special here
            query = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if return_answer_only:
              query = query[len(para[i]):].lstrip(".? \n\t")
            ret[query] = 1

      if do_beam:

        # Here we use Beam-search. It generates better quality queries, but with less diversity
          outputs = model.generate(
                **input_ids, 
                max_length=max_length, 
                num_beams=max(int(max_return_sequences/2) if do_sample else max_return_sequences,5), 
                repetition_penalty=1.05,
                min_length=min_length,
                no_repeat_ngram_size=4,
                penalty_alpha=0.6 if do_self_contrastive else None,  
                num_return_sequences=max(1, int(max_return_sequences/2)) if do_sample else max_return_sequences, 
                early_stopping=True
            )
    

          for i in range(len(outputs)): # can use batch_decode, unless we want to do something special here
            query = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if return_answer_only:
              query = query[len(para[i]):].lstrip(".? \n\t")
            ret[query] = 1

    return list(ret.keys())  

try:
  if artist is None: assert False
except:  
  artist = pandas.read_csv("/content/artists-data.csv")
  artist2genre = dict([a, b] for a, b in zip(artist['Link'],artist['Genres']))

  lyric = pandas.read_csv("/content/lyrics-data.csv")
import json
with open("/content/drive/Shareddrives/LAION/synth_lyrics.jsonl", "w") as output:
  batch = []
  for aLink, lyric2, lang in zip(lyric['ALink'], lyric['Lyric'], lyric['language']):
    if lang == 'en':
      genre= artist2genre[aLink] 
      artist = aLink.replace("-", " ").strip("\/")
      genre = genre.split(";")[0]
      lyric2 = lyric2.split(" ")[:10]
      if lyric2[-1].lower().strip("\n") in {"a", "the", "this", "that", "those", "these"}:
        lyric2 = lyric2[:-1]
      lyric2 = " ".join(lyric2).replace("\n", " / ")
      instr= f"Write me a song in the {genre} genre in the style of {artist} with the themes of '{lyric2}'"
      if len(batch) > 15:
        song = generate_ext(batch, model, tokenizer)
        song = [s.split("#")[0].replace("/", "\n").replace("...", "\n").replace("1.", "\n").replace("2.", "\n").replace("3.", "\n").replace("4.", "\n").replace("5.", "\n").replace("6.", "\n").replace("7.", "\n").replace("8.", "\n").replace("9.", "\n").replace(".", ".\n").replace("\n\n", "\n") for s in song]
        for instr, s in zip(batch, song):
          if "1" not in s and "2" not in s and \
            "3" not in s and "4" not in s and \
            "5" not in s and "6" not in s and \
            "7" not in s and "7" not in s and \
            "9" not in s and "10" not in s:
            output.write (json.dumps({'text':instr + s, 'metadata': {'source': 'synth_music'}})+"\n")
        batch = []
      batch.append(f"User: {instr}\n\nJoi2: lyrics:\n")

import json, random
i = 0

with open("synth_music_reject.jsonl", "w") as reject:
  with open("synth_music_clean.jsonl", "w") as out:
    with open("synth_lyrics.jsonl") as input:
      for l in input:
        data = json.loads(l.strip())
        instruction, response = data['text'].split("\n\nJoi2: lyrics:\n", 1)
        response = response.lower()
        if " kill " in response or " rape " in response or "crack music" in response or "fuck" in response or "cocaine" in response or " perc " in response or " codine " in response or "below is some" in response or 'verses' in response or ' verse ' in response or ' cock ' in response or 'suck my' in response or 'pussy' in response or 'dialogue' in response or 'first line' in response or 'song' in response or 'lyric' in response: 
          reject.write(l)
          continue
        instruction, phrase = instruction.split("with the themes of",1)
        instruction = instruction.split(" in the style of")[0]
        phrase  = [a.strip(" '.?").split(" (")[0].lower() for a in phrase.strip("', ").split("/") if len(a)> 10]
        for a in phrase:
          response = response.replace(a, '')
          for b in a.split(", "):
            response = response.replace(b, '')
        response = response.replace(", ", "\n")
        response = response.replace("nig-ga", "man")
        if random.randint(0,1):
          response = response.replace("nigga", "man")

        elif random.randint(0,1):
          response = response.replace("nigga", "woman")
        elif random.randint(0,1):
          response = response.replace("nigga", "girl")
        elif random.randint(0,1):
          response = response.replace("nigga", "guy")
        response = response.replace("bitches", "girls")
        response = response.replace("bitch", "girl")
        response = response.replace("good head", "love")
        response = response.replace("dick wet", "heart break")
        response = response.replace("nigg", "man")
        response = response.replace(" hoe ", " girl ")
        response = response.replace(" ho ", " girl ")
        response = "\n".join([(r.strip()[0].upper() + r.strip()[1:]).strip(",.") for r in response.split("\n") if len(r.strip()) > 5])
        response = response.replace(" i ", " I ").replace(" i'", " I'").replace("  ", " ")
        if len(response) < 60: 
          reject.write(l)
          continue
        if response.count("\n") < 3: 
          reject.write(l)
          continue
        out.write (json.dumps({'text': instruction.strip()+'.\nAssistant: '+response, 'metadata': data['metadata']})+"\n")
      
