import pandas as pd
import numpy as np
import os, sys
from collections import Counter
from tqdm import tqdm
import pdb
import pickle

import spacy
nlp = spacy.load('en')


def filter_tags(tag_column, vocab_fpath, min_freq=3):
    """ Return tag vocab counter with minimum frequency setting, filtered tag list """

    # Build vocab
    remove_list = ['']
    vocab = Counter([tag for tag_list in tag_column for tag in tag_list if not tag in remove_list])

    filtered_tags = [[tag for tag in tag_list if vocab[tag] >= min_freq] for tag_list in tag_column]

    # Save vocab
    with open(vocab_fpath, 'wb') as f:
        pickle.dump(vocab, f)

    return filtered_tags, vocab


def preprocess_tags(tags_str):

    tags = []

    if isinstance(tags_str, str):
        # Lowercase, combine multiword, split
        tags = [tag[1:-1].lower().replace(' ', '_') for tag in tags_str[1:-1].split(',')]

        #for tag in tags:
        #    if tag.startswith('('):
        #        pdb.set_trace()

    return tags


def main():
    # I/O
    data_dirpath = '/usr2/mamille2/tumblr/data'
    #posts_fpath = os.path.join(data_dirpath, 'textposts_100posts.pkl') # selected to have 100 posts
    posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.pkl') # recent 100 posts, even if don't have 100
    vocab_fpath = os.path.join(data_dirpath, 'textposts_recent100_tag_vocab.pkl')

    # Settings
    debug = False
    min_tag_freq = 1

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
    data[f'parsed_tags_minfreq{min_tag_freq}'] = list(map(preprocess_tags, tqdm(data['post_tags'].tolist(), ncols=100)))
    print('done.')
    sys.stdout.flush()

    # Filter for vocab
    print("Filtering tags...", end=' ')
    sys.stdout.flush()
    data[f'parsed_tags_minfreq{min_tag_freq}'], vocab = filter_tags(data[f'parsed_tags_minfreq{min_tag_freq}'].tolist(),
        vocab_fpath,
        min_freq=min_tag_freq)
    print('done.')
    sys.stdout.flush()

    # Save data
    if debug:
        pdb.set_trace()

    else:
        print(f"Saving preprocessed file to {posts_fpath}...", end=' ')
        sys.stdout.flush()
        data.to_pickle(posts_fpath)
        print("done")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
