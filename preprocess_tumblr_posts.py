import pandas as pd
import re
import numpy as np
# from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt
import matplotlib
from bs4 import BeautifulSoup
import urllib
from pprint import pprint
import pickle
import os, sys
from collections import Counter
from html.parser import HTMLParser
from tqdm import tqdm
import string
from scipy.sparse import csr_matrix, vstack

import spacy


# # Make dataset for Tumblr word embedding training

# Load 27M posts (73GB)
print("Loading data...", end=' ')
sys.stdout.flush()
data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent100.pkl')
print('done')

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
data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))

# Save data
print("Saving tokenized file...", end=' ')
sys.stdout.flush()
ear
ata.to_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent100.pkl')
print("done")
