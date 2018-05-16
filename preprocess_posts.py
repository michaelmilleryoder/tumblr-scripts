import pandas as pd
import re
import numpy as np
import pickle
import os, sys
from html.parser import HTMLParser
from tqdm import tqdm
from nltk.corpus import words
import pdb

import spacy
nlp = spacy.load('en')

# I/O
data_dirpath = '/usr2/mamille2/tumblr/data'
posts_fpath = os.path.join(data_dirpath, 'textposts_100posts.pkl')

# Load posts
print("Loading data...", end=' ')
sys.stdout.flush()
data = pd.read_pickle(posts_fpath)
print('done')
sys.stdout.flush()

# Initialize tokenizer
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
data['body_str'] = data['body_toks'].map(lambda x: ' '.join(x))
print('done.')
sys.stdout.flush()

# Remove usernames
blog_names = set(data['source_title'].unique()) # might not all be strings
dict_wds = set(words.words())
blog_names = blog_names - dict_wds
data['body_toks_no_titles'] = list(map(lambda x: [t for t in x if not t in blog_names], tqdm(data['body_toks'].tolist())))
data['body_toks_str_no_titles'] = data['body_toks_no_titles'].map(lambda x: ' '.join(x))

# Save data
print(f"Saving tokenized file to {posts_fpath}...", end=' ')
sys.stdout.flush()
data.to_pickle(posts_fpath)
print("done")
sys.stdout.flush()
