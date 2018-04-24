import pandas as pd
import re
import numpy as np
#from bs4 import BeautifulSoup
import urllib
import pickle
import os, sys
from collections import Counter
#import nltk
import random
from html.parser import HTMLParser
from tqdm import tqdm
#from multiprocessing import Pool
import string
#import spacy
import pdb

# INPUT
char_limit = 25
desc_fpath = '/usr0/home/mamille2/tumblr/data/blog_descriptions_recent100.pkl'
list_desc_fpath = '/usr0/home/mamille2/tumblr/data/list_descriptions_recent100.pkl'
restr_desc_fpath = f'/usr0/home/mamille2/tumblr/data/list_descriptions_recent100_restr{char_limit}.pkl'

# Load data
tqdm.write("Loading data...", end=' ')
sys.stdout.flush()
desc_data = pd.read_pickle(desc_fpath)
tqdm.write("done")

# Remove duplicates by tumblog id
desc_data.drop_duplicates(inplace=True, subset=['tumblog_id'])
tqdm.write(f'#descriptions: {len(desc_data)}\n')

# Remove HTML tags
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

tqdm.write("Removing HTML tags...", end=' ')
sys.stdout.flush()
desc_data['parsed_blog_description'] = list(map(strip_tags, tqdm(desc_data['tumblr_blog_description'].tolist())))

# Remove empty parsed blogs
desc_data = desc_data[desc_data['parsed_blog_description'].map(lambda x: len(x) > 0)]
tqdm.write("done\n")

# # Process URLs (to hyphens) so don't interact with list descriptions
tqdm.write("Processing URLs...", end=' ')
sys.stdout.flush()

def process_url(url):
    url = re.sub(r'https?:\/\/', '', url).replace('www.','').replace('.', '-')
    url = re.sub(r'\/$', '', url).replace('/', '-')
    return url

def process_urls(desc):
    processed = desc
    
    for m in re.finditer(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', desc):
        processed = processed.replace(m.group(), process_url(m.group()))
    
    return processed

desc_data['parsed_blog_description'] = list(map(process_urls, tqdm(desc_data['parsed_blog_description'].tolist())))
tqdm.write("done\n")

# # Process dates so don't interact with list descriptions
tqdm.write("Processing dates...", end=' ')
sys.stdout.flush()

def process_dates(desc):
    processed = desc
    date_p = re.compile(r'(?:\d{4}|\d{2})(?:\.|\/)\d{2}(?:\.|\/)(?:\d{4}|\d{2})')
    
    for m in re.finditer(date_p, desc):
        processed = processed.replace(m.group(), process_date(m.group()))
    
    return processed

def process_date(datestr):
    return datestr.replace('/', '-').replace('.', '-')

desc_data['parsed_blog_description'] = list(map(process_dates, tqdm(desc_data['parsed_blog_description'].tolist())))
tqdm.write("done\n")

tqdm.write(f"Saving parsed blog descriptions to {desc_fpath}...", end=' ')
sys.stdout.flush()
desc_data.to_pickle(desc_fpath)
tqdm.write('done\n')

# Segment blog descriptions into list descriptions
tqdm.write("Identifying list descriptions...")

## Identify list descriptions
seps = ['|', '/', '\\', '.']
desc_re = '|'.join([r'^.*\{0}.+\{0}.*$'.format(s) for s in seps])

def is_list_desc(in_str):
    if re.search(desc_re, in_str):
        return True
    else:
        return False

mask = list(map(is_list_desc, tqdm(desc_data['parsed_blog_description'].tolist())))
list_desc_data = desc_data[mask]
tqdm.write(f'#list descriptions: {len(list_desc_data)}')
tqdm.write('done\n')

def segment_desc(desc):
    delims = ['|', '/', '.', '\\']
    
    # take max segmentation length to select delimiter (may be a bit inefficient)
    delim_ctr = {d: desc.count(d) for d in delims}
    sep = sorted(delim_ctr, key=delim_ctr.get, reverse=True)[0]
    
    return [el.strip().lower() for el in desc.split(sep) if len(el) > 0 and el != ' ']

tqdm.write("Segmenting descriptions...", end=' ')
sys.stdout.flush()
list_desc_data['segments'] = list(map(segment_desc, tqdm(list_desc_data['parsed_blog_description'].tolist())))

# Remove lines with 0 or 1 segments
list_desc_data = list_desc_data[list_desc_data['segments'].map(lambda x: len(x) > 1)]
tqdm.write('done\n')
tqdm.write(f'#list descriptions: {len(list_desc_data)}\n')

tqdm.write(f"Saving list descriptions to {list_desc_fpath}", end=' ')
sys.stdout.flush()
list_desc_data.to_pickle(list_desc_fpath)
tqdm.write('done\n')

# Restrict segment lengths
tqdm.write("Restricting segment lengths...")
sys.stdout.flush()

rm_list = [
    'relax and watch!',
    'my snapChat: lovewet9x',
    '/_        @will4i20    _\\',
    'my snapchat: sexybaby9x'
    'Please wait' # think this one is like a 404 error, not user-entered
]

list_desc_data[f'restr_segments_{char_limit}'] = list_desc_data['segments'].map(
    lambda d: [s for s in d if (len(s) <= char_limit and len(s) > 1) if not s in rm_list]
    )

# Filter out empty restricted segments
restr_desc_data = list_desc_data[list_desc_data[f'restr_segments_{char_limit}'].map(lambda x: len(x) > 0)]
tqdm.write('done\n')
tqdm.write(f'#restricted list descriptions: {len(restr_desc_data)}\n')

# Remove punctuation from segments
def split_rm_punct(segments):
    """ Return segments split on punctuation, punctuation removed """
    
    new_segs = []
    
    for seg in segments:
        new_seg = ' '.join(re.split(r'\W', seg))
        new_seg = re.sub(r'\W', ' ', new_seg)
        new_seg = re.sub(r'\s+', ' ', new_seg).strip()
        new_segs.append(new_seg)
        
    return new_segs

restr_desc_data['segments_25_nopunct'] = list(map(split_rm_punct, tqdm(rest_desc_data['restr_segments_25'].tolist())))

tqdm.write(f"Saving list descriptions to {restr_desc_fpath}", end=' ')
sys.stdout.flush()
restr_desc_data.to_pickle(restr_desc_fpath)
tqdm.write('done\n')

tqdm.write("FINISHED PREPROCESSING")
