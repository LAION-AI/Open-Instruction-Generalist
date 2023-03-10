from datasets import load_dataset
import csv
import json
import random
import tqdm


def create_lyrics(output):

    #######
    # download and unzip https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv
    #######
    
    with open('lyrics-data.csv') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        
    en_rows = [row for row in rows if row[4] == 'en']
    
    user_templates = [
        "Can you provide the lyrics for a song called XXX?",
        "I'm looking for the lyrics to a track named XXX, could you show me?",
        "Could you display the lyrics of a song with the title of XXX?",
        "Can you present me with a song to a piece titled XXX?",
        "I'd like to see the lyrics to a song titled XXX, can you help me with that?",
        "Can you give me the lyrics for XXX?",
        "Would you mind writing the lyrics for a song called XXX?",
        "Make a song that goes by the name XXX",
        "Generate the lyrics for a song called XXX",
        "Write me a song lyrics with a name of XXX",
        "Compile words for a song titled XXX.",
        "Can you provide the lyrics to a song called XXX?",
        "I'm looking for the lyrics of a song called XXX, can you tell me?",
        "Can you show the lyrics to a song called XXX?",
        "Can you give me a song called XXX?",
        "I want to see the lyrics of a song called XXX, can you help me?",
        "Can you give me the lyrics to XXX?",
        "Would you mind writing lyrics for a song called XXX?",
        "Make a song called XXX.",
        "Generate lyrics for a song named XXX",
        "Write me a song titled XXX lyrics",
        "Write lyrics for a song called XXX",
    ]
    
    for _, sname, _, lyric, lang in tqdm.tqdm(en_rows):
        prompt = "<human>: " + random.choice(user_templates).replace('XXX', sname) + '\n<bot>: ' + lyric
        output.write(json.dumps({'text': prompt}) + '\n')