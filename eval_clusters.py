import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm

import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from .. import gaussian_mixture_cotrain
from gaussian_mixture_cotrain import GaussianMixtureCotrain

from pprint import pprint
import pdb

# INPUTS

# Data file
desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_embeddings_avg.npy'
#desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_embeddings_sum.npy'
#desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_avg.npy'
#desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_sum.npy'

# Trained model file
model_path = '/usr0/home/mamille2/tumblr/data/gmm_50_desc_avg.pkl'
#model_path = '/usr0/home/mamille2/tumblr/data/gmm_50_desc_sum.pkl'
#model_path = '/usr0/home/mamille2/tumblr/data/gmm_cotrain_50_desc_avg.pkl'
#model_path = '/usr0/home/mamille2/tumblr/data/gmm_cotrain_50_desc_sum.pkl'

# Description file
desc_path = '/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl'
#desc_path = '/usr0/home/mamille2/tumblr/data/desc_recent5.pkl'

NUM_EXAMPLES_PRINT = 20

# Outpath to print to
outpath = model_path[:-4] + '_clusters.txt'


# LOAD INPUTS

# Load data
print("Loading data...", end=" ")
sys.stdout.flush()
desc_emb = np.load(desc_emb_path)
print("done")

# Load model
print("Loading model...", end=" ")
sys.stdout.flush()
with open(model_path, 'rb') as f:
    clf = pickle.load(f)
print("done")

# Load descriptions
print("Loading descriptions...", end=" ")
sys.stdout.flush()
desc_df = pd.read_pickle(desc_path)
desc_toks = desc_df['tokenized_blog_description'].tolist()
print("done")
print()


# CALCULATE METRICS
print("Calculating metrics...")
sys.stdout.flush()
#print("BIC over all datapoints: {}".format(clf.bic(desc_emb)))
print("Log-likelihood: {}".format(clf.lower_bound_))


# GET HIGH-PROBABILITY DESCRIPTIONS FOR EACH CLUSTER
# Get highest weights
wted_comps = np.argsort(clf.weights_)[::-1]

# Predicting probabilities for descriptions
print("Predicting probabilities for descriptions...", end=' ')
sys.stdout.flush()
probs = clf.predict_proba(desc_emb)
print('done')

# Calculate silhouette score based on top prob cluster
clusters_assgn = np.argsort(probs, axis=1)[:,-1] # top cluster probabilities for each datapoint
pdg.set_trace()
print("Silhouette score: {}".format(silhouette_score(desc_emb, clusters_assgn)))

def top_descs(probs, descs, k, order, outpath):
    """ Prints top k descriptions for each component"""
    
    top_probs = np.argsort(probs, axis=0)[::-1] # ranked samples for each component
    
    with open(outpath, 'w') as f:
        for i in order:
            f.write("Component {}".format(i) + '\n')
            col = top_probs[:,i]
            
            for el in col[:k]: 
                f.write('\t' + ' '.join(descs[el]) + '\n') # for tokenized

            f.write('\n')
            
#def top_descs(probs, descs, k, order, vocab_file=None):
#    """ Prints top k descriptions for each component"""
#    
#    top_probs = np.argsort(probs, axis=0)[::-1]
#    
#    if vocab_file: # dict [n_words]: [vocab]
#        with open(vocab_file, 'rb') as f:
#            vocab = pickle.load(f)
#    
#    for i in order:
#        print("Component {}".format(i))
#        col = top_probs[:,i]
##     for i, c in enumerate(top_probs.T):
#        
#        for el in col[:k]: 
#            if vocab_file:
#                print('\t' + ' '.join(d if d in vocab[100000] else '<unk>' for d in descs[el])) # for tokenized
#            else:
#                print('\t' + ' '.join(d if d in vocab[100000] else '<unk>' for d in descs[el])) # for tokenized
##             print('\t' + descs[el])
#            
#        print()

print("Writing top descriptions for clusters to {}...".format(outpath), end=' ')
#top_descs(probs, desc_toks, 20, wted_comps, '/usr0/home/mamille2/tumblr/data/halfday_top5_vocab100000.pkl')
top_descs(probs, desc_toks, NUM_EXAMPLES_PRINT, wted_comps, outpath)
print('done')
