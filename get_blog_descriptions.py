import pytumblr
import pandas as pd
import re
import random
from tqdm import tqdm
import pickle
import os
import pdb
import time
from datetime import datetime

# OAuth

with open('../oauth.txt') as f:
    lines = f.read().splitlines()
    
client = pytumblr.TumblrRestClient(lines[0], lines[1], lines[2], lines[3])


# # Sample user ids 

#print("Loading data...")
datapath = '../data/halfday_text_usernames.txt'

random.seed(a=42)

with open(datapath) as f:
    usernames = f.read().splitlines()
u10k = sorted(random.sample(usernames, 10000))

# # Querying blog descriptions
print("Querying blog descriptions...")

out_dirpath = '../data/blog_descriptions'

desc = {}
for i,name in enumerate(tqdm(u10k)):
    if not name or len(name) == 0:
        continue

    successful = False
    fatal = False

    if fatal:
        break
    
    while not successful:

        info = client.blog_info(name)

        # returns something
        if 'blog' in info:
            desc[name] = info['blog']['description']
            successful = True


        # errors out
        elif 'errors' in info:
            err_title = info['errors'][0]['title']
            tqdm.write("ERROR: {}".format(err_title))
            
            if err_title == 'Limit Exceeded':
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                tqdm.write("Waiting an hour from {}...".format(ts))
                time.sleep(3600)

        # empty structure
        else:
            successful = True


        # Save out every so often
        if i > 0 and i % 1000 == 0:

            if len(desc) == 0:
                tqdm.write("Empty user description structure. FAIL.")
                fatal = True
                break

            outpath = os.path.join(out_dirpath, 'blog_desc{:05d}.pkl'.format(i))
            with open(outpath, 'wb') as f:
                pickle.dump(desc, f)
        
            tqdm.write("Wrote blog descriptions to {}".format(outpath))
            desc = {}
