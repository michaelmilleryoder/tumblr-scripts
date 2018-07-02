import pandas as pd
import numpy as np
import os, sys
from collections import Counter
from tqdm import tqdm
import pdb
import pickle

import spacy
nlp = spacy.load('en')


def preprocess_tags(tags_str):

    tags = []

    if isinstance(tags_str, str):
        # Lowercase, combine multiword, split
        tags = [tag[1:-1].lower().replace(' ', '_') for tag in tags_str[1:-1].split(',')]

    return tags

def save_text_file(text_rows, outpath):
    with open(outpath, 'w') as f:
        for post in text_rows:
            f.write(post + '\n')


def main():
    # I/O
    data_dirpath = '/usr2/mamille2/tumblr/data'
    #posts_fpath = os.path.join(data_dirpath, 'textposts_100posts.pkl') # selected to have 100 posts
    posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.pkl') # recent 100 posts, even if don't have 100
    text_outpath = posts_fpath[:-4] + '_tags.txt'

    # Settings
    debug = False

    # Load posts
    print("Loading data...", end=' ')
    sys.stdout.flush()
    if debug:
        data = pd.read_pickle(posts_fpath).head(100)
    else:
        data = pd.read_pickle(posts_fpath)
    print('done')
    sys.stdout.flush()

    # Preprocess tags
    print("Preprocessing tags...", end=' ')
    sys.stdout.flush()
    tag_lines = [' '.join(l) for l in list(map(preprocess_tags, tqdm(data['post_tags'].tolist()))) if len(' '.join(l)) > 0]
    print('done.')
    sys.stdout.flush()

    # Save text file (for eg training word embeddings)
    print(f"Writing text file to {text_outpath}...", end=' ')
    sys.stdout.flush()
    save_text_file(tag_lines, text_outpath) 
    print("done")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
