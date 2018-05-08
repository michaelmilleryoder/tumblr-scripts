from gensim.models import FastText
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import pdb
import os
from keras.preprocessing.text import Tokenizer

# I/O
data_dirpath = '/usr2/mamille2/tumblr/data'
embs_fpath = '/usr2/kmaki/tumblr/top100posts-131M_fasttext-embeddings_3-6.bin.bin' # ~100GB RAM
#embs_fpath = '/usr2/kmaki/tumblr/top100posts-131M_fasttext-embeddings_3-6.bin.vec' # ~40GB RAM
posts_fpath = os.path.join(data_dirpath, 'textposts_100posts.pkl')
out_fpath = os.path.join(data_dirpath, 'blog_descriptions_100posts.npy') # for lookup table of embeddings

# Settings
MAX_VOCAB_SIZE = 100000

# Load text posts
print('Loading posts...')
posts = pd.read_pickle(posts_fpath)
tids = posts['tumblog_id'].unique()
texts = [' '.join(posts[posts['tumblog_id']==tid]['body_toks_str_no_titles']) for tid in tids] # concatenated posts

# Load word embeddings
print('Loading word embeddings...')
#wembed = KeyedVectors.load_word2vec_format(embs_fpath) # lose unk subword features of fastText
wembed = FastText.load_fasttext_format(embs_fpath)
embed_dims = wembed.vector_size

# Get vocab
print('Building vocab...')
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”')
tokenizer.fit_on_texts(texts)
vocab = list(tokenizer.word_index.keys())[:MAX_VOCAB_SIZE] # lower indices are words kept

# Build, save lookup table
print('Building lookup table...')
vocab_embed = np.empty((len(vocab), embed_dims))

# Calculate OOV vector
#unk_vec = np.mean([wembed[wd] for wd in wembed])

for i, wd in enumerate(vocab):
    vocab_embed[i,:] = wembed[wd]
    #if i in wembed:
    #    vocab_embed[i,:] = wembed[wd]
    #else:
    #    vocab_embed[i,:] = unk_vec

np.save(out_fpath, vocab_embed)
print(f'Saved lookup table of embeddings to {out_fpath}.')
