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
import requests.exceptions

# OAuth

with open('../oauth.txt') as f:
    lines = f.read().splitlines()
    
client = pytumblr.TumblrRestClient(lines[0], lines[1], lines[2], lines[3])


# # Sample user ids 

print("Loading data...")
datapath = '../data/halfday_text_usernames.txt'

#random.seed(a=42)

with open(datapath) as f:
    all_usernames = f.read().splitlines()
#usernames = sorted(random.sample(usernames, 10000))
usernames = all_usernames

# # Querying blog descriptions
print("Querying blog descriptions...")

out_dirpath = '../data/blog_descriptions'

desc = {}

# Check for already scraped data
scraped_files = [fname for fname in os.listdir(out_dirpath) if fname.startswith('blog_desc')]

if scraped_files:
    offset = max([int(fname[9:11])*1000 for fname in scraped_files])
else: offset = 0

sample = usernames[offset:]

print("Already found {} descriptions scraped".format(offset))

# Do the scraping
for i,name in enumerate(tqdm(sample)):
    if not name or len(name) == 0:
        continue

    try:
        info = client.blog_info(name)
    except requests.exceptions.SSLError as e:
        print(name)
        pdb.set_trace()

    # returns something
    if 'blog' in info:
        d = info['blog']['description']
        if len(d) > 0:
            desc[name] = info['blog']['description']

    # errors out
    elif 'errors' in info:
        err_title = info['errors'][0]['title']
        
        if err_title == 'Limit Exceeded':
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            tqdm.write("Waiting an hour from {}...".format(ts))
            time.sleep(3600)

        elif err_title == 'Not Found': # 404 error
            pass

        else:
            tqdm.write("ERROR: {}".format(err_title))

    # Save out every so often
    if i > 0 and i % 1000 == 0:

        if len(desc) == 0:
            tqdm.write("Empty user description structure. FAIL.")
            break

        outpath = os.path.join(out_dirpath, 'blog_desc{:02d}k.pkl'.format((i + offset)//1000))
        with open(outpath, 'wb') as f:
            pickle.dump(desc, f)
    
        tqdm.write("Wrote blog descriptions to {}".format(outpath))
        desc = {}
