import pandas as pd
import numpy as np
import pickle
import os

from sklearn.mixture import GaussianMixture

from pprint import pprint


# # Run GMM clustering on blog descriptions

# Load data
print("Loading data...")
desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_embeddings_avg.npy'
#desc_emb_path = '/usr0/home/mamille2/tumblr/data/desc_embeddings_sum.npy'
desc_emb = np.load(desc_emb_path)
print()

# Fit model
N_COMPS = 50
N_DATAPTS = len(desc_emb)-int(2e6)
BEG_DATAPT = int(2e6)
#N_DATAPTS = int(2e6)
#BEG_DATAPT = 0
MAX_ITERS = 300
LOAD_EXISTING = True
print("{} components\n{} datapts\n{} iterations".format(N_COMPS, N_DATAPTS, MAX_ITERS))
print()

# Look for existing model to train
if LOAD_EXISTING:   
    print("Looking for existing model...", end="")
    path = '/usr0/home/mamille2/tumblr/data/gmm_{}_desc_avg.pkl'.format(N_COMPS)
    if os.path.exists(path):
        print("found")
        print("Loading existing model...", end='')
        with open(path, 'rb') as f:
            clf = pickle.load(f)
        print("done")
    else:
        raise IOError("Can't find existing model.")

else:
    #clf = GaussianMixture(n_components=N_COMPS, verbose=2, warm_start=True, max_iter=MAX_ITERS)
    clf = GaussianMixture(n_components=N_COMPS, verbose=2, warm_start=False, max_iter=MAX_ITERS)

X = desc_emb[BEG_DATAPT:BEG_DATAPT + N_DATAPTS,:]
print("Fitting model...")
clf.fit(X)



# Save model
outpath = '/usr0/home/mamille2/tumblr/data/gmm_{}_desc_avg.pkl'.format(N_COMPS)
print("Saving model to {}...".format(outpath), end='')
with open(outpath, 'wb') as f:
    pickle.dump(clf, f)
print("done")
