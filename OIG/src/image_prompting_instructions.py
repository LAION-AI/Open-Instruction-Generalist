#@title image prompting instructions
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
import gzip
from collections import Counter
import os
try:
  from nltk.corpus import stopwords as nltk_stopwords
  nltk_stopwords.words('english') 
except:
  import nltk
  nltk.download('stopwords')
import random

stopwords_set = set(nltk_stopwords.words('english') + ['...', 'there', 'could', 'should', 'shall', 'can', 'might', 'may', 'include', 'including'])
#TODO: use the safety stuff from LAION-AI/riverbed

hate_words_set = {'niggas', 'fuck', 'wetback', 'blame', 'chinks', 'shut', 'niggers', 'ugly', 'blacks', 'lame', 'sand', 'butt', 'dumb', 'dyke', 'rape', 'whites', 'dykes', 'bitch', 'akbar', 'homo', 'monkey', 'nigger', 'fags', 'coon', 'hate', 'spic', 'raped', 'allah', 'wetbacks', 'trailer', 'queer', 'chucker', 'inbred', 'colored', 'killed', 'jungle', 'shit', 'fucking', 'nigga', 'savages', 'dirty', 'eyed', 'shorty', 'beat', 'kill', 'queers', 'stupid', 'chink', 'slave', 'cunt', 'fuckin', 'faggot', 'faggots', 'trash'}
flagged_words_set ={'tit', 'coprolagnia', 'skeet', 'swinger', 'zoophilia', 'bunghole', 'voyeurweb', 'prick', 'pissing', 'nympho', 'felching', 'lolita', 'pikey', 'squirting', 'hentai', 'urophilia', 'doggiestyle', 'goatcx', 'cumslut', 'pornstars', 'abortion', 'goddamn', 'spac', 'jailbait', 'ejaculate', 'fucked', 'sexual', 'bitching', 'asshole', 'butt', 'cumshots', 'pisspig', 'blumpkin', 'grope', 'cunt', 'twinkie', 'fagging', 'strappado', 'bollocks', 'deepthroating', 'lust', 'shits', 'beastiality', 'clitoris', 'tits', 'tushy', 'nigga', 'fanny', 'fagots', 'kike', 'bastardo', 'knobbing', 'acrotomophilia', 'femdom', 'sexually', 'bareback', 'camslut', 'pornhub', 'cipa', 'dominatrix', 'cocksucking', 'shitting', 'snowballing', 'figging', 'pecker', 'neonazi', 'lovemaking', 'dink', 'yiffy', 'bitch', 'masturbating', 'sexo', 'raghead', 'swastika', 'suck', 'topless', 'ballbag', 'homoerotic', 'orgasim', 'tranny', 'damn', 'fucks', 'asses', 'scrotum', 'octopussy', 'goodpoop', 'fucker', 'whore', 'sluts', 'anal', 'youporn', 'voyuer', 'pubes', 'paedophile', 'jism', 'vorarephilia', 'cuckold', 'fingerbang', 'shitted', 'titty', 'bullshit', 'hardcore', 'bimbos', 'sexuality', 'cumshot', 'handjob', 'xhamster', 'twink', 'piss', 'pornography', 'orgy', 'dildos', 'dildo', 'assholes', 'fuckers', 'feltch', 'scat', 'rectum', 'kinky', 'dogging', 'panty', 'motherfucker', 'panties', 'negro', 'nudity', 'eunuch', 'jizz', 'jiggerboo', 'babeland', 'fucktards', 'slut', 'rimjob', 'genitals', 'domination', 'shrimping', 'jerk-off', 'nsfw', 'carpetmuncher', 'brazzers', 'bum', 'fingering', 'livesex', 'anus', 'bellend', 'erotism', 'deepthroat', 'vulva', 'throating', 'titt', 'arsehole', 'coprophilia', 'fecal', 'shitblimp', 'labia', 'spooge', 'bangbros', 'shitty', 'voyeur', 'snatch', 'omorashi', 'nambla', 'ass', 'boobs', 'darkie', 'cums', 'ejaculated', 'screwing', 'cumming', 'muffdiving', 'camwhore', 'bbw', 'mong', 'milf', 'pisser', 'autoerotic', 'redtube', 'sexcam', 'towelhead', '2g1c', 'upskirt', 'dendrophilia', 'shibari', 'twat', 'lusting', 'sadist', 'ejaculating', 'pornstar', 'masturbate', 'schlong', 'god-damned', 'nawashi', 'hooker', 'pornos', 'kinkster', 'creampie', 'dingleberry', 'kock', 'apeshit', 'horney', 'skank', 'spastic', 'horny', 'slutty', 'shagging', 'nipples', 'pussies', 'anilingus', 'turd', 'poopchute', 'boob', 'daterape', 'punany', '🖕', 'frotting', 'intercourse', 'santorum', 'hore', 'bestial', 'bastard', 'ejaculation', 'shite', 'buttcheeks', 'quim', 'masturbation', 'dick', 'fisting', 'nazi', 'undressing', 'rape', 'yaoi', 'fuckin', 'viagra', 'poontang', 'arse', 'bulldyke', 'bdsm', 'coon', 'cock-sucker', 'felch', 'fudgepacker', 'clit', 'shemale', 'horniest', 'jigaboo', 'breasts', 'kinbaku', 'pthc', 'spic', 'circlejerk', 'xx', 'retard', 'rimming', 'dingleberries', 'bangbus', 'masterbating', 'paki', 'hell', 'worldsex', 'doggystyle', 'coons', 'playboy', 'g-spot', 'tosser', 'dolcett', 'blowjobs', 'poon', 'dyke', 'masochist', 'pissed', 'slanteye', 'cunnilingus', 'sex', 'crap', 'cum', 'cialis', 'beaners', 'humping', 'incest', 'fuck', 'chink', 'fucking', 'bukkake', 'vibrator', 'dommes', 'wank', 'buceta', 'erotic', 'poop', 'cornhole', 'xnxx', 'cunillingus', 'penis', 'threesome', 'rapist', 'titties', 'shag', 'bloody', 'fag', 'xxx', 'tribadism', 'busty', 'fellatio', 'xvideos', 'pube', 'nigger', 'queaf', 'clusterfuck', 'bondage', 'sodomize', 'tubgirl', 'strapon', 'pricks', 'futanari', 'sodomy', 'queef', 'flange', 'vagina', 'cocks', 'niggers', 'fagot', 'camgirl', 'wang', 'porno', 'nymphomania', 'bitches', 'son-of-a-bitch', 'raping', 'bastinado', 'semen', 'dog-fucker', 'pedophile', 'guro', 'shit', 'orgasms', 'pisses', 'footjob', 'testicle', 'bestiality', 'gokkun', 'honkey', 'ponyplay', 'boner', 'wetback', 's&m', 'scissoring', 'nipple', 'orgasm', 'fuckings', 'cougar', 'jiggaboo', 'thumbzilla', 'nude', 'ejaculates', 'assmunch', 'pissoff', 'ass-fucker', 'shota', 'goregasm', 'ecchi', 'smegma', 'splooge', 'sadism', 'dvda', 'beaner', 'balls', 'butthole', 'smut', 'nimphomania', 'poof', 'juggs', 'bollok', 'goatse', 'porn', 'faggot', 'nutten', 'sucks', 'orgies', 'escort', 'birdlock', 'duche', 'spunk', 'gangbang', 'barenaked', 'blowjob', 'pedobear', 'pussy', 'pegging', 'sexy', 'cock'}
csam_set1 = {'lolhentai','nymphet', 'nimphet', 'babyj', 'voglia', 'eurololita', 'lolli', 'lola', 'lolita', 'lolly', 'loli', 'lolitaguy',  \
            "pedo", 'hussyfan', 'kidzilla', 'raygold', 'ygold',  'mylola', \
            'babyshivid', 'kidzilla',  'kdquality', 'cbaby', 'kinderficker', 'preteen', }
csam_set2 = {'little',  'girl', 'boy', 'child', 'kid', 'baby', 'sissy', 'kiddie', 'toddler', \
            'bath', 'baths', 'bathing', 'qwerty', 'qqaazz', 'ptsc', 'izzy', 'rika', \
            'pthc', 'tanta','newstar', 'playtoy', 'imouto', 'lourinha', 'amateurz', 'arina', 'shiori', 'chiharu', 'nablot', 
                  }

def near_dup_key_fn(sent_arr):
  sent_arr = [s if len(s) <4 else s[:4] for s in sent_arr if len(s) > 3 and s not in stopwords_set]
  if not sent_arr: return -1
  if len(sent_arr) > 5: sent_arr = sent_arr[:5]
  return hash("".join(sent_arr))


def detect_non_safe(text, url="", do_augment=False):
    orig_text = text
    text = text.lower()
    if "year old" in text: text = text.replace("year old", "yo")
    if "years old" in text: text = text.replace("years old", "yo")
    text = text.replace("lame pun coon","")
    text_arr = text.split()
    found_csam = 0
    url_array = [a for a in url.lower().replace("https://", "").replace("http://", "").replace("/", " ").replace("-", " ").replace("_", " ").replace(".", " ").split()]
    for word in list(set(text_arr+url_array)):
        if word in csam_set1:
          found_csam += 2
        elif word in csam_set2:
          found_csam += 1
        elif ("yr" in word or "yo" in word):
          try:
            age = int(word.strip("yor-"))
            if age < 18 and ("yo" in word or ("yr" in word and "old" in text)): 
              found_csam += 2
          except:
            pass
    
    is_porn_url = ("porn" in url or ".xxx/" in url or "xxx." in url or "xnxx" in url or "xhamster" in url or "redtube" in url or "xvideos" in url or "hclips" in url or "tnaflix" in url or "tube8" in url or "spankbang" in url or "4chan" in url or "tblop" in url or "thumbzilla" in url or "hentai" in url) 
    if ((found_csam > 0 and (' rape ' in text or ' raped' in text or ' rapes ' in text or 'bondage' in text)) or
        (found_csam > 1 and ('beaten' in text or 'murder' in text or ' kill' in text or 'brutual' in text)) or 
        (found_csam > 2 and (is_porn_url  or "nude" in text or "naked" in text or "breast" in text or "penis" in text or "suck" in text or "masturbation" in text or "handjob" in text or " cunt " in text or "dildo" in text or " cum " in text \
                             or "boobs" in text or "blowjob" in text or " anal " in text or "fuck" in text or " cock " in text or "pussy" in text or "tits" in text))):
        if "field of rape" in text or "rape field"  in text or "rape flower"  in text or "wikimedia" in url or  "wikipedia" in url or "dailymail" in url or "msn" in url or "times" in url or "youtube" in url or "cnn" in url or "bbc" in url or "abc" in url or "nbc" in url or "news" in url or ".gov" in url:
          if "field" not in text and "flower" not in text and "news" not in text and "report" not in text and "defendant" not in text and "police" not in text and "claim" not in text and "allegation" not in text:
              if do_augment:
                if random.random() < 0.5:
                  text = (text +", it is reported").replace("  ", " ")
                elif  random.random() < 0.5:
                  text = ("in the news: " + text).replace("  ", " ")
                elif  random.random() < 0.5:
                  text = ("reportedly " + text).replace("  ", " ")
          return text, "harmless"
        return orig_text,"csam"
            
    if (is_porn_url or \
        (found_csam > 1 and (is_porn_url  or "nude" in text or "naked" in text or "breast" in text or "penis" in text or "suck" in text or "masturbation" in text or "handjob" in text or " cunt " in text or "dildo" in text or " cum " in text \
                             or "boobs" in text or "blowjob" in text or " anal " in text or "fuck" in text or " cock " in text or "pussy" in text or "tits" in text)) or\
        (" incest " in text or "beastie" in text or "efuck" in text or "gfuck" in text or "masturbation" in text or "handjob" in text or " cunt " in text or "dildo" in text or " milf " in text or "bbw" in text or \
         " cum " in text or "boobs" in text or "blowjob" in text or " anal " in text or "fuck" in text or " cock " in text or "pussy" in text or "porn" in text or "tits" in text)):
      if ('piss' in text or ' scat ' in text or 'little' in text or 'teen' in text or "erotic" in text or "virgin" in text or " gay " in text or " lesbian " in text or "snatch" in text or " hairy " in text or \
          " dirty " in text or " lick " in text or "sucks" in text or " dick " in text or "huge" in text or "xxx" in text or "redhead" in text or "movie" in text or " old" in text or\
          " gay " in text or "free" in text or "ass" in text or "sex" in text or "blonde" in text or "japanese" in text or "mature" in text or "black" in text or "asian" in text or "girl" in text or "video" in text or \
          "photo" in text or "scene" in text or "mom" in text or "dad" in text or "sister" in text or "brother" in text or "teacher" in text or "doctor" in text or "office" in text or \
          "young" in text  or "busty" in text or "teen" in text or "indian" in text or "granny" in text or "horny" in text or "amateur" in text
          ): 
        if do_augment:
          if " girl" in text  and random.random() < .5: 
            text = text.replace(" girl", " "+random.choice(['nymphet', 'nimphet', 'babyj', 'baby', 'voglia', 'eurololita', 'lolli', 'lola', 'lolita', 'lolly', 'loli', 'lolitaguy', 'kid', 'sissy', 'preteen', 'kiddie', 'toddler',])+" ").replace("  ", " ").replace(" s ", " ")
            return text, "csam"
          elif " boy " in text  and random.random() < .5: 
            text = text.replace(" boy ",  " "+random.choice([ 'nymphet', 'nimphet', 'babyj', 'baby', 'voglia', 'eurololita', 'lolli', 'lola', 'lolita', 'lolly', 'loli', 'lolitaguy', 'kid', 'sissy', 'preteen', 'kiddie', 'toddler',])+" ").replace("  ", " ").replace(" s ", " ")
            return text, "csam"
          elif "18yo" in text and random.random() < .5: 
            text = text.replace("18yo", str(random.randint(1, 17))+"yo").replace("  ", " ").replace(" s ", " ")
            return text, "csam"
          elif "daugher" in text  and random.random() < .5: 
            text = text.replace("daugher", "baby").replace("  ", " ").replace(" s ", " ")
            return text, "csam"
          elif " son " in text  and random.random() < .5: 
            text = text.replace(" son ", " baby ").replace("  ", " ").replace(" s ", " ")
            return text, "csam"
      return orig_text, "porn"
    offensive = 0
    if is_porn_url: offensive = 1
    hate = 0
    for word in list(set(text_arr)):
      if word in flagged_words_set:
        offensive += 1
      if word in hate_words_set:
        hate += 1
    if hate > 2 or (offensive > 0 and hate > 1):
        return orig_text, "hate"
    elif offensive > 1:
        return orig_text, "offensive"
    if ".law" in url or ".edu" in url or ".gov" in url or "github" in url or "philpapers" in url or "stackoverflow" in url or "arxiv" in url or "pubmed" in url or "uspto" in url or "free.law" in url or "nih.gov" in url:
      return orig_text, "harmless"
    elif " memes " not in text and "memes" not in url and "confession bear" not in text:
      if ("ology" in url or "techno" in url or "genom" in url or "religion" in url or "philosophy" in url or "romance" in url or "mystery" in url or "scifi" in url or "fantasy" in url or "drama" in url or \
          "library" in url or "justia" in url or "politic" in url or "history" in url or "science" in url or "humor" in url or "poetry" in url or "literature" in url or "math" in url or \
          "medic" in url or "biotech" in url or "invention" in url or "legal" in url or "lexisnexis" in url or "westlaw" in url or \
          "contracts" in url or  "subtitles" in url  or "github" in url or "wikimedia" in url or  "wikipedia" in url or "free.law" in url):
        return text, "harmless"
      elif ("sports" in url or "dailymail" in url or "msn" in url or "times" in url or "youtube" in url or "cnn" in url or "bbc" in url or "abc" in url or "nbc" in url or "news" in url):
        return orig_text, "harmles"
    if do_augment:
        if " girl" in text  and random.random() < .5: 
          text = text.replace(" girl", " "+random.choice(['baby', 'lolli', 'lola', 'lolita', 'lolly', 'loli',  'kid', 'sissy', 'preteen', 'kiddie', 'toddler',])+" ").replace("  ", " ").replace(" s ", " ")
          return text, "harmless"
        elif " boy " in text  and random.random() < .5: 
          text = text.replace(" boy ",  " "+random.choice(['baby', 'lolli', 'lola', 'lolita', 'lolly', 'loli',  'kid', 'sissy', 'preteen', 'kiddie', 'toddler',])+" ").replace("  ", " ").replace(" s ", " ")
          return text, "harmless"
        elif "18yo" in text and random.random() < .5: 
          text = text.replace("18yo", str(random.randint(1, 17))+"yo").replace("  ", " ").replace(" s ", " ")
          return text, "harmless"
        elif "daugher" in text  and random.random() < .5: 
          text = text.replace("daugher", "baby").replace("  ", " ").replace(" s ", " ")
          return text, "harmless"
        elif " son " in text  and random.random() < .5: 
          text = text.replace(" son ", " baby ").replace("  ", " ").replace(" s ", " ")
          return text, "harmless"        
    return orig_text, "unknown"


def create_image_prompts(output):
  if not os.path.exists("laion_safe_nonsafe.tsv"):
      !cp  /content/drive/Shareddrives/ontocord/laion_safe_nonsafe.tsv ./
  laion_img = [a.split("\t")[0] for a in open("laion_safe_nonsafe.tsv").read().split("\n") if len(a.split("\t")[0]) > 100 and "..." not in a and "|" not in a and "foreground" not in a and "background" not in a and "photo" not in a and "picture" not in a and "(" not in a and ")" not in a and ".jpg" not in a and "@" not in a and "&q" not in a and "<" not in a and ">" not in a and "|| hate" not in a and "|| porn" not in a and "csam" not in a and "|| offensive" not in a]
  i = 0
  for sent in laion_img:
    if "http" in sent or "www" in sent or "#" in sent: continue
    sent = sent.replace("stockvideo's en b-roll-footage met ", "")
    sent, label = detect_non_safe(sent, do_augment=False)
    if label not in {"unknown", "unknown"}: continue
    i+=1
    bad_words = [word.strip("()[].,") for word in sent.split() if word.lower() not in stopwords_set]
    bad_words = list(set([word for word in bad_words if word.lower() in hate_words_set or word.lower() in flagged_words_set or word.lower() in csam_set1 or word.lower() in csam_set2]))
    if bad_words: continue
    if len(sent) > 300:
      instruction = ", ".join(list(set([word.strip(":;<>,.?/~`!@#$%^&*()-_+=") for word in sent.split() if len(word.strip("~`!@#$%^&*()-_+=")) > 4 and word.lower() not in stopwords_set]))[:5]).replace(",,",",").replace(", , ", ", ")
      dialog = ("User: Give me a sentence with these words: " +instruction+"\nAssistant: " + sent)
    else:
      instruction = ", ".join([word.strip(":;<>,.?/~`!@#$%^&*()-_+=") for word in sent.split() if len(word.strip("~`!@#$%^&*()-_+=")) > 4 and word.lower() not in stopwords_set][:5]).replace(",,",",").replace(", , ", ", ")
      dialog = ("User: Give me an image prompt to draw an image with " +instruction+"\nAssistant: " + sent)
    d = dialog
    if random.randint(0,1):
      d = d.replace("Give me", random.choice(["", "Can you create", "I'm looking for", "How about"]))
    if random.randint(0,1):
      d = d.replace("image prompt to draw", random.choice(["", "prompt for", "image prompt for", "stablity prompt for"]))
    labels = [a.split("[")[1] for a in d.split("Assistant:")[-1].split("]") if "[" in a]
    before, after = d.split("Assistant:")
    after = after.split("]")[-1]
    d = before+"Assistant:"+after
    d = d.replace("  ", " ").replace("  ", " ")
    if d:
      output.write (json.dumps({'text': d, 'metadata': {'source': 'laion_image_prompts'}})+"\n")

    #if i > 100: break
  print (i)
          
