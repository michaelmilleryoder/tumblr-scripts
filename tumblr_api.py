
# coding: utf-8

# In[59]:

import pytumblr
import pandas as pd
import re
import random
from tqdm import tqdm_notebook as tqdm
import pickle


# In[3]:

# OAuth

with open('../../oauth.txt') as f:
    lines = f.read().splitlines()
    
client = pytumblr.TumblrRestClient(lines[0], lines[1], lines[2], lines[3])


# # Sample user ids

# In[6]:

datapath = '../../data/halfday_text.pkl'

data = pd.read_pickle(datapath)
data.columns


# In[9]:

data['source_url']


# In[35]:

p = extract_username(data.loc[8, 'source_url'])
p


# In[43]:

data['username'] = data['source_url'].map(extract_username)
data['username']


# In[44]:

data.to_pickle('../../data/halfday_text_usernames.pkl')


# In[45]:

usernames = data['username'].unique()
len(usernames)


# In[49]:

u1k = random.sample(list(usernames), 1000)
len(u1k)


# In[57]:

u10k = random.sample(list(usernames), 10000)
len(u10k)


# In[15]:

[u for u in data['source_url'] if (isinstance(u, str) and u.startswith('https://'))]


# In[41]:

u_p = re.compile(r'https?:\/\/(.*?)\.', re.IGNORECASE)


# In[42]:

def extract_username(url):
    if isinstance(url, str):
        m = re.match(u_p, url)
        if not m:
            print(url)
        else:
            return m.group(1)
    
    else:
        return None


# # Querying blog descriptions

# In[60]:

desc = {}

# for name in ['otherkinfashionunder20', 'other-otherkin', 'kiramii']:
for name in tqdm(u1k[:100]):
    info = client.blog_info(name)
    if 'blog' in info:
        desc[name] = info['blog']['description']
    
print(len(desc))

outpath = '../../blog_descriptions.pkl'

with open(outpath, 'wb') as f:
    pickle.dump(list(desc.values()), f)
    
print("Wrote blog descriptions to {}".format(outpath))


# # Querying tags (terf)

# In[4]:

posts = client.tagged('terf', filter='text')


# In[6]:

[posts[i]['body'] for i in range(20)]

