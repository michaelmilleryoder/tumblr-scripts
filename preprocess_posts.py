import pandas as pd
import re
import numpy as np
import pickle
import os, sys
from html.parser import HTMLParser
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.corpus import words
import pdb

import spacy
nlp = spacy.load('en')

# I/O
data_dirpath = '/usr2/mamille2/tumblr/data'
#posts_fpath = os.path.join(data_dirpath, 'textposts_100posts.pkl') # selected to have 100 posts
posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.pkl') # recent 100 posts, even if don't have 100
text_outpath = posts_fpath[:-3] + 'txt'

# Settings
remove_usernames = False
save_text_file = True

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

def clean_html(html):
    soup = BeautifulSoup(html, 'lxml')
    for s in soup(['script', 'style']):
        s.decompose()
    return ' '.join(soup.stripped_strings)

def strip_tags(html):
    s = MLStripper()
    text = clean_html(str(html)).strip()
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

def remove_usernames(data):
    """ Removes usernames, saves in separate columns """
    blog_names = set(data['source_title'].unique()) # might not all be strings
    dict_wds = set(words.words())
    blog_names = blog_names - dict_wds
    data['body_toks_no_titles'] = list(map(lambda x: [t for t in x if not t in blog_names], tqdm(data['body_toks'].tolist())))
    data['body_toks_str_no_titles'] = data['body_toks_no_titles'].map(lambda x: ' '.join(x))

def save_text_file(data, colname, outpath):
    with open(outpath, 'w') as f:
        for post in data[colname].tolist():
            f.write(post + '\n')

# Tokenize, preprocess all posts
print("Preprocessing posts...", end=' ')
sys.stdout.flush()
data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))
data['body_str'] = data['body_toks'].map(lambda x: ' '.join(x))
print('done.')
sys.stdout.flush()

# Remove usernames
if remove_usernames:
    remove_usernames(data)

# Save text file (for eg training word embeddings)
if save_text_file:
    print(f"Writing text file...", end=' ')
    sys.stdout.flush()
    save_text_file(data, 'body_str') 
    print("done")
    sys.stdout.flush()

# Save data
print(f"Saving tokenized file to {posts_fpath}...", end=' ')
sys.stdout.flush()
data.to_pickle(posts_fpath)
print("done")
sys.stdout.flush()
