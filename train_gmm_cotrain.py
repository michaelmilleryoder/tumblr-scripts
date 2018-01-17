import pandas as pd
import numpy as np
import pickle
import os

#from sklearn.mixture import GaussianMixture
from gaussian_mixture_cotrain import GaussianMixtureCotrain

from pprint import pprint


# # Run GMM clustering, co-training on blog descriptions and recent text posts

# Load data
print("Loading description data...")
desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_avg.npy'
#desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_sum.npy'
desc_emb = np.load(desc_emb_path)
print()

print("Loading text post data...")
posts_emb_path = '/usr0/home/mamille2/tumblr/data/halfday_5posts_embed_avg.npy'
#posts_emb_path = '/usr0/home/mamille2/tumblr/data/halfday_5posts_embed_sum.npy'
posts_emb = np.load(desc_emb_path)
print()

assert desc_emb.shape == posts_emb.shape

# Fit model
N_COMPS = 50
N_DATAPTS = min(desc_emb.shape[0], posts_emb.shape[0])
#N_DATAPTS = 500000
#BEG_DATAPT = int(1.5e6)
MAX_ITERS = 1000
LOAD_EXISTING = False
print("{} components\n{} datapts\n{} iterations".format(N_COMPS, N_DATAPTS, MAX_ITERS))
print()

# Look for existing model to train
if LOAD_EXISTING:   
    print("Looking for existing model...", end="")
    path = '/usr0/home/mamille2/tumblr/data/gmm_{}_desc.pkl'.format(N_COMPS)
    if os.path.exists(path):
        print("found")
        print("Loading existing model...", end='')
        with open(path, 'rb') as f:
            clf = pickle.load(f)
        print("done")
    else:
        raise IOError("Can't find existing model.")

else:
    #clf = GaussianMixtureCotrain(n_components=N_COMPS, verbose=2, warm_start=True, max_iter=MAX_ITERS)
    clf = GaussianMixtureCotrain(n_components=N_COMPS, verbose=2, verbose_interval=1, max_iter=MAX_ITERS)

#X = desc_emb[BEG_DATAPT:BEG_DATAPT + N_DATAPTS,:]
X_arr = [posts_emb, desc_emb] # desc embeddings last
print("Fitting model...")
clf.fit(X_arr)



# Save model
outpath = '/usr0/home/mamille2/tumblr/data/gmm_cotrain_{}_desc_avg.pkl'.format(N_COMPS)
#outpath = '/usr0/home/mamille2/tumblr/data/gmm_cotrain_{}_desc_sum.pkl'.format(N_COMPS)
print("Saving model to {}...".format(outpath), end=' ')
with open(outpath, 'wb') as f:
    pickle.dump(clf, f)
print("done")
