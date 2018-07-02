import pandas as pd
import numpy as np
import os, sys
import random
import string
import pickle
import pdb
import re

from collections import Counter
from tqdm import tqdm


# INPUT
char_limit = 25
in_path = 'data/'
out_path = 'output/'
blog_desc_fpath = '{}blog_descriptions_recent100.pkl'.format(in_path)
sep_chars_fpath = '{}common_sep_chars.pkl'.format(out_path)
list_desc_fpath = '{}bootstrapped_nopunc_list_descriptions_100posts.pkl'.format(out_path)
restr_desc_fpath = '{}bootstrapped_nopunc_list_descriptions_recent100_restr{}.pkl'.format(out_path, char_limit)

# Load starting with the preprocess list_desc file
tqdm.write("Loading data...", end=' ')
sys.stdout.flush()
desc_data = pd.read_pickle(blog_desc_fpath)
tqdm.write('{} Total entries'.format(len(desc_data)))
tqdm.write("done")
pdb.set_trace()

# Segment blog descriptions into list descriptions
tqdm.write("Identifying list descriptions...")

## Identify list descriptions

with open(sep_chars_fpath, 'rb') as f:
    sep_counts = pickle.load(f)
#seps = ['|', '/', '\\', '.']
# Load separators, but drop alphanumeric ones
seps = [s for s in sep_counts.keys() if len(s) > 0 and not re.search('[a-zA-Z0-9]', s)]
# Optionally remove punctuation chars
punc = ['.', ',', ';', '!', '?', "'", '"']
seps = [s for s in seps if not any([p in s for p in punc])]
# Add some escape characters for the re search
check_chars = ['[', ']', '(', ')', '|']
for i, s in enumerate(seps):
    if len(s) > 1 and any([c in s for c in check_chars]):
        s = s[0] + s[1:].replace(')', '\)').replace('(','\(').replace('[','\[').replace(']','\]').replace('|', '\|')
        seps[i] = s
desc_re = '|'.join([r'^.*\{0}.+\{0}.*$'.format(s) for s in seps])

tqdm.write('{} total separators'.format(len(seps)))

def is_list_desc(in_str):
    if re.search(desc_re, in_str):
        return True
    else:
        return False

mask = [*map(is_list_desc, tqdm(desc_data['parsed_blog_description'].tolist()))]
list_desc_data = desc_data[mask]
tqdm.write('#list descriptions: {}'.format(len(list_desc_data)))
tqdm.write('done\n') 

def segment_desc(desc):
    delims = seps #['|', '/', '.', '\\']
    
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
tqdm.write('#list descriptions: {}\n'.format(len(list_desc_data)))

tqdm.write("Saving list descriptions to {}".format(list_desc_fpath), end=' ')
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

list_desc_data['restr_segments_{}'.format(char_limit)] = list_desc_data['segments'].map(
    lambda d: [s for s in d if (len(s) <= char_limit and len(s) > 1) if not s in rm_list]
    )

# Filter out empty restricted segments
restr_desc_data = list_desc_data[list_desc_data['restr_segments_{}'.format(char_limit)].map(lambda x: len(x) > 0)]
tqdm.write('done\n')
tqdm.write('#restricted list descriptions: {}\n'.format(len(restr_desc_data)))

tqdm.write("Saving list descriptions to {}".format(restr_desc_fpath), end=' ')
sys.stdout.flush()
restr_desc_data.to_pickle(restr_desc_fpath)
tqdm.write('done\n')

tqdm.write("FINISHED PREPROCESSING")
