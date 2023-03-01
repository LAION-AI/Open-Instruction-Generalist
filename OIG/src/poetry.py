from datasets import Dataset,load_dataset
import json
import pandas as pd
import numpy as np
import pronouncing
import re
from collections import Counter

MIN_RHYMES = 3
PROMPTS = {
    "begining":{
    "default": ["{} a {} entitled {}"],
    "about": ["{} a {} on the topic {}",
         "{} a {} about {} "],
    "rhyming":["{} a {} containing rhyming words for the word '{}' entitled {}.",
       "{} a {} containing rhyming words for the word '{}'"],
    "genre_age":["{} a {} on the topic {} ", 
      "{} a {} of the genre {} "],
    },
    "completion":{"completion":["{} the {}",
         "{} the {} entitled {}"],}
 
 }
SYNONYMS = {
        "poem" : ["poem","poetry"]
        "compose" : ["Write","Help me write", "Compose", "Please craft", "Give me"]
        "complete" : ["Complete", "Finish", "Put the finishing touches to"]
        "sentiment" : {"positive":["positive ","happy "],"negative":["negative ","sad "]}
        "writing" : ["in the manner of {}.","in {}'s writing style.",]
        "period" : [" written in {}."," written during {} period"]
        }

def get_best_rhymes(content):
    
    best_rhymes = []
    first_word = 0
    lines = [re.sub(r'\r','',line) for line in content.split("\n") if line!=""]
    if len(lines)>1:
        last_words = [re.sub('[^A-Za-z0-9]+', '', line.split(" ")[-1]) for line in lines]
        for word in last_words:
            rhymes = pronouncing.rhymes(word)
            index = last_words.index(word)
            rhymes_words = np.intersect1d(rhymes,last_words[index:index+10])
            if len(rhymes_words) > len(best_rhymes):
                best_rhymes = list(rhymes_words)
                best_rhymes.insert(0,word)
    if best_rhymes:            
        first_word = min([last_words.index(word) for word in best_rhymes])
    return last_words[first_word],best_rhymes


def toss_prompt(possible_prompts,prompt_types):
    prompt_type = np.random.choice(prompt_types,p=[0.9,0.1])
    prompt = np.random.choice(possible_prompts[prompt_type])
    return prompt_type,prompt

def get_emotion(content):
    
    labels = [nlp(re.sub(r'\r|\n','',line))[0]["label"] for line in content.split(".")][:3]
    if labels.count("negative") >= 2:
        sentiment = "negative"
    elif labels.count("positive") >= 2:
        sentiment = "positive"
    else:
        sentiment = None
    return sentiment
        

def build_prompt(possible_prompts,prompt_types,args,sentiment,rhyming_list):
    
    compose = np.random.choice(SYNONYMS["compose"])
    poem = np.random.choice(SYNONYMS["poem"])
    complete = np.random.choice(SYNONYMS["complete"])
    
    prompt_type,prompt = toss_prompt(possible_prompts,prompt_types)
    if prompt_type == "default":
        
        prompt = prompt.format(compose,poem,args["poem name"])
    
    elif prompt_type == "about":
        
        prompt = prompt.format(compose,poem,args["poem name"].lower())
        
    elif prompt_type == "genre_age":
        
        prompt = prompt.format(compose,poem,args["type"],args["age"])
    
    
    elif prompt_type == "completion":
        
        prompt = prompt.format(complete,poem,args["poem name"])
    
    if ((np.random.randint(0,5)) and (sentiment!=None)):
            index = prompt.find(poem)
            prompt = prompt[:index] + np.random.choice(SYNONYMS["sentiment"][sentiment]) + prompt[index:]
    
    if ((np.random.randint(0,5)) and (len(rhyming_list)>2)):
            index = prompt.find(poem) + len(poem)
            prompt = prompt[:index] + " containing rhyming words for the word '{}'".format(rhyming_list[0]) + prompt[index:]

        
    return prompt

def add_author(prompt,author,top_authors):
    
    if (author.lower() in top_authors) and (np.random.randint(0,2)):
        style = np.random.choice(SYNONYMS["writing"]).format(author)
        prompt= prompt + " " + style
        
    return prompt

def get_top_authors(dataset):
    
    counter = Counter([poem["author"] for poem in dataset]).most_common(100)
    authors,_ = zip(*counter)
    return authors 

def create_poem_instructions(dataset):
    
    top_authors = get_top_authors(dataset)
    all_prompts = []
    for item in tqdm(dataset):
        item["poem name"] = re.sub(r'\r|\n|\[.*\]','',item["poem name"]).strip()
        poem_name, content, author, genre, age = item.values()
        prompt_type = np.random.choice(["completion","begining"],p=[0.3,0.7])
        
        sentiment = get_emotion(content)
        rh_word,rh_wordslist = get_best_rhymes(item["content"])
        item["rhyming"] = rh_word
        
        possible_prompts = prompts[prompt_type]

        
        if prompt_type == "begining":
            
            if ((genre!=None) and np.random.randint(0,2)):
                
                prompt = build_prompt(possible_prompts,["genre_age","default"],item,sentiment,rh_wordslist)
                
                if ((item["age"]!="") and (np.random.randint(0,2))):
                    prompt += np.random.choice(SYNONYMS["period"]).format(item["age"])
                    

            elif poem_name.lower().startswith("the"):
                prompt = build_prompt(possible_prompts,["about","default"],item,sentiment,rh_wordslist)
             
            else:
                prompt = build_prompt(possible_prompts,["default","default"],item,sentiment,rh_wordslist)
                
            prompt = add_author(prompt,author,top_authors)
            response = item["content"].strip()
            
        else:
            prompt = build_prompt(possible_prompts,["completion","completion"],item,sentiment,rh_wordslist)
            prompt = add_author(prompt,author,top_authors)
            num_lines = np.random.randint(3,6)
            poem_lines = item["content"].split("\n")
            prompt = prompt + "\n" + "\n".join(poem_lines[:num_lines])
            response = "\n".join(poem_lines[num_lines:]).strip()
            
        all_prompts.append({"prompt":prompt,"response":response})
        
    return all_prompts


def main():

    prompts = []
    for dataset_name in hf_datasets:
        dataset = load_dataset(dataset_name,split="train")
        prompts.extend(create_poem_instructions(dataset))
    return prompts
    
if __name__ == "__main__":

    main()



