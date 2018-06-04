import pandas as pd
import re
import numpy as np
import os, sys
from tqdm import tqdm
from multiprocessing import Pool
import gc

from preprocess_posts import preprocess_post, save_text_file


# I/O
data_dirpath = '/usr2/mamille2/tumblr/data'
#posts_fpath = os.path.join(data_dirpath, 'textposts_100posts.pkl') # selected to have 100 posts
posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.pkl') # recent 100 posts, even if don't have 100
text_outpath = posts_fpath[:-3] + 'txt'

# Settings
n_chunks = 30
n_processes = 3


def main():

    # Load posts
    print("Loading data...", end=' ')
    sys.stdout.flush()
    data_full = pd.read_pickle(posts_fpath)
    print('done.')
    sys.stdout.flush()

    chunks = np.array_split(data_full, n_chunks)

    pool = Pool(n_processes)

    for i, data in enumerate(chunks):
        print(f"Chunk {i}/{len(chunks)}")

        # Tokenize, preprocess all posts
        print("Preprocessing posts...", end=' ')
        sys.stdout.flush()

        post_toks = pool.map(preprocess_post, tqdm(data['body'].tolist()))
        #data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))
        post_str = map(lambda x: ' '.join(x), post_toks)

        print('done.')
        sys.stdout.flush()

        # Save text file (for eg training word embeddings)
        print(f"Writing text file to {text_outpath}...", end=' ')
        sys.stdout.flush()
        save_text_file(post_str, text_outpath) 
        print("done")
        sys.stdout.flush()
        print()

        del post_toks
        del post_str
        gc.collect()

if __name__=='__main__': main()
