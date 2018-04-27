import pandas as pd
import re
import numpy as np
import pickle
import os, sys
from html.parser import HTMLParser
from tqdm import tqdm

import spacy

# I/O
data_dirpath = '/usr2/mamille2/tumblr/data'
posts_fpath = os.path.join(data_dirpath, 'bootstrapped_textposts_recent100.pkl')

# Load posts
print("Loading data...", end=' ')
sys.stdout.flush()
data = pd.read_pickle(posts_fpath)
print('done')
sys.stdout.flush()

# Initialize tokenizer
nlp = spacy.load('en')

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()

def preprocess_post(post):
    
    # Strip html tags
    nohtml = strip_tags(post)
    
    # Tokenize with spaCy
    toks = [tok.text for tok in nlp.tokenizer(nohtml.lower())]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return toks


# Tokenize, preprocess all posts
print("Preprocessing posts...", end=' ')
sys.stdout.flush()
data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))
data['body_toks_str'] = data['body_toks'].map(lambda x: ' '.join(x))
print('done.')
sys.stdout.flush()

# Save data
print(f"Saving tokenized file to {posts_fpath}...", end=' ')
sys.stdout.flush()
data.to_pickle(posts_fpath)
print("done")
sys.stdout.flush()
