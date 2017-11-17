import pytumblr
import pandas as pd
import re
import random
from tqdm import tqdm
import pickle
import os
import pdb

# OAuth

with open('../oauth.txt') as f:
    lines = f.read().splitlines()
    
client = pytumblr.TumblrRestClient(lines[0], lines[1], lines[2], lines[3])


# # Sample user ids

print("Loading data...")
datapath = '../data/halfday_text_usernames.pkl'
data = pd.read_pickle(datapath)

random.seed(a=42)
usernames = data['username'].unique()
u10k = sorted(random.sample(list(usernames), 10000))

# # Querying blog descriptions
print("Querying blog descriptions...")

out_dirpath = '../data/blog_descriptions'

desc = {}
for i,name in enumerate(tqdm(u10k)):
    if not name or len(name) == 0:
        continue

    info = client.blog_info(name)

    if 'blog' in info:
        desc[name] = info['blog']['description']

    elif 'meta' in info and 'errors' in info['meta']:
        err_title = info['meta']['errors'][0]['title']
        print("ERROR: {}".format(err_title))
        
        if err_title == 'Limit Exceeded':
            break

    if i > 0 and i % 1000 == 0:

        if len(desc) == 0:
            print("Empty user description structure. FAIL.")
            break

        outpath = os.path.join(out_dirpath, 'blog_desc{:05d}.pkl'.format(i))
        with open(outpath, 'wb') as f:
            pickle.dump(list(desc.values()), f)
    
        tqdm.write("Wrote blog descriptions to {}".format(outpath))
        desc = {}
