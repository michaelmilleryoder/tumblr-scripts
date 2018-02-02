import pandas as pd
import re
import numpy as np
#from bs4 import BeautifulSoup
import urllib
from pprint import pprint
import pickle
import os, sys
from collections import Counter
#import nltk
import random
from html.parser import HTMLParser
from tqdm import tqdm
#from multiprocessing import Pool
import string
#import spacy
import pdb

# INPUT
desc_fpath = '/usr0/home/mamille2/tumblr/data/blog_descriptions_recent100.pkl'

# Load data
print("Loading data...", end=' ')
sys.stdout.flush()
desc_data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/blog_descriptions_recent100.pkl')
print("done")

# Remove HTML tags
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()

print("Removing HTML tags...", end=' ')
sys.stdout.flush()
desc_data['parsed_blog_description'] = list(map(strip_tags, tqdm(desc_data['tumblr_blog_description'].tolist())))

# Remove empty parsed blogs
desc_data = desc_data[desc_data['parsed_blog_description'].map(lambda x: len(x) > 0)]
print("done")

# # Process URLs (to hyphens) so don't interact with list descriptions
print("Processing URLs...", end=' ')
sys.stdout.flush()

def process_url(url):
    url = re.sub(r'https?:\/\/', '', url).replace('www.','').replace('.', '-')
    url = re.sub(r'\/$', '', url).replace('/', '-')
    return url

def process_urls(desc):
    processed = desc
    
    for m in re.finditer(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', desc):
        processed = processed.replace(m.group(), process_url(m.group()))
    
    return processed

desc_data['parsed_blog_description'] = list(map(process_urls, tqdm(desc_data['parsed_blog_description'].tolist())))
print("done")

# # Process dates so don't interact with list descriptions
print("Processing dates...", end=' ')
sys.stdout.flush()

date_p = re.compile(r'(?:\d{4}|\d{2})(?:\.|\/)\d{2}(?:\.|\/)(?:\d{4}|\d{2})')
def has_date(desc):
    if isinstance(desc, str):
        return re.match(date_p, desc)
    else:
        return None

def process_dates(desc):
    processed = desc
    date_p = re.compile(r'(?:\d{4}|\d{2})(?:\.|\/)\d{2}(?:\.|\/)(?:\d{4}|\d{2})')
    
    for m in re.finditer(date_p, desc):
        processed = processed.replace(m.group(), process_date(m.group()))
    
    return processed

def process_date(datestr):
    return datestr.replace('/', '-').replace('.', '-')

desc_data['parsed_blog_description'] = desc_data['parsed_blog_description'].map(process_dates)
print("done")

print(f"Saving parsed blog descriptions to {fpath}...", end=' ')
sys.stdout.flush()
desc_data.to_pickle(desc_fpath)
print('done')

# Segment blog descriptions into list descriptions
print("Identifying list descriptions...", end=' ')
sys.stdout.flush()

## Identify list descriptions
seps = ['|', '/', '\\', '.']
desc_re = '|'.join([r'^.*\{0}.+\{0}.*$'.format(s) for s in seps])

def is_list_desc(in_str):
    if re.search(desc_re, in_str):
        return True
    else:
        return False

mask = list(map(is_list_desc, tqdm(desc_data['parsed_blog_description'].tolist())))
list_desc_data = desc_data[mask]
print('done')

def segment_desc(desc):
    delims = ['|', '/', '.', '\\']
    
    # take max segmentation length to select delimiter (may be a bit inefficient)
    delim_ctr = {d: desc.count(d) for d in delims}
    sep = sorted(delim_ctr, key=delim_ctr.get, reverse=True)[0]
    
    return [el.strip().lower() for el in desc.split(sep) if len(el) > 0 and el != ' ']

print("Segmenting descriptions...", end=' ')
sys.stdout.flush()
list_desc_data['segments'] = list(map(segment_desc, tqdm(list_desc_data['parsed_blog_description'].tolist())))

# Remove lines with 0 or 1 segments
list_desc_data = list_desc_data[list_desc_data['segments'].map(lambda x: len(x) > 1)]
print('done')


list_desc_data.to_pickle('/usr0/home/mamille2/tumblr/data/list_descriptions.pkl')

# Restrict segment lengths








# # Prepare description segments for annotation of category mentions

# In[180]:


# Filter out empties (segments all too long)
print(len(list_desc_data))
filtered = list_desc_data[list_desc_data['restr_segments_25'].map(lambda x: len(x) > 0)]
len(filtered)


# In[181]:


filtered.columns


# In[183]:


filtered.loc[:,['tumblog_id','restr_segments_25']]


# # Prepare description segments for Brown clustering (and other thgs)

# In[171]:


# Filter spam
rm_list = [
    'relax and watch!',
    'my snapChat: lovewet9x',
    '/_        @will4i20    _\\',
    'my snapchat: sexybaby9x'
#     'Please wait' # not sure about this one
]


# In[176]:


# Get list of descs to cluster
char_limit = 25
list_desc_data[f'restr_segments_{char_limit}'] = list_desc_data['segments'].map(
    lambda d: [s for s in d if (len(s) <= char_limit and len(s) > 1) if not s in rm_list]
    )


# In[177]:


list_desc_data.to_pickle('/usr0/home/mamille2/tumblr/data/list_descriptions.pkl')


# In[175]:


descs = [' '.join(d) for d in list_desc_data[f'restr_segments_{char_limit}'].tolist() if len(d) > 0]
print(len(descs))
descs


# In[133]:


# Filter to just terms occurring more than once
wds = [w for d in descs for w in d.split()]
len(wds)


# In[135]:


wd_ctr = Counter(wds)
vocab = [w for w in wd_ctr if wd_ctr[w] > 1]
len(vocab)


# In[138]:


# descs_freq = [w for d in descs for w in d.split() if w in vocab]
descs_freq = []
for d in tqdm(descs):
    newd = []
    for w in d.split():
        if w in vocab:
            newd.append(w)
    descs_freq.append(' '.join(newd))
            
descs_freq[:100]


# In[139]:


descs_freq = [d for d in descs_freq if len(d) > 0]
len(descs_freq)


# In[141]:


# Save out for Brown clustering
with open('/usr0/home/mamille2/tumblr/data/desc_segments_20_freq.txt', 'w') as f:
    for d in descs_freq:
        f.write(d + '\n')




# # Extract 'list descriptors'

# In[111]:


# Load descriptions

desc_data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')
print(desc_data.columns)
print(len(desc_data))


# In[124]:


seps = ['|', '/', '\\', '.']
desc_re = '|'.join([r'^.*\{0}.+\{0}.*$'.format(s) for s in seps])
desc_re


# In[125]:


def is_list_desc(in_str):
    if re.search(desc_re, in_str):
        return True
    else:
        return False


# In[126]:


mask = list(map(is_list_desc, tqdm(desc_data['parsed_blog_description'].tolist())))
list_desc_data = desc_data[mask]
len(list_desc_data)


# In[140]:


# Investigate whether adult blogs should stay (lots of diversity, so should)
list_desc_data.columns


# In[142]:


list_desc_data['blog_classifier'].unique()


# In[143]:


list_desc_data[list_desc_data['blog_classifier']=='adult']['parsed_blog_description']


# In[127]:


list_desc_data.to_pickle('/usr0/home/mamille2/tumblr/data/list_descriptions.pkl')


# ## Segment lists

# In[146]:


def segment_desc(desc):
    delims = ['|', '/', '.', '\\']
    
    # take max segmentation length to select delimiter (may be a bit inefficient)
    delim_ctr = {d: desc.count(d) for d in delims}
    sep = sorted(delim_ctr, key=delim_ctr.get, reverse=True)[0]
    
    return [el.strip().lower() for el in desc.split(sep) if len(el) > 0 and el != ' ']


# In[17]:


segment_desc('this is my tumblr.blog | its me | im great')


# In[18]:


segment_desc('this is my tumblr.blog | its me. | im great')


# In[19]:


segment_desc('this is my tumblr blog / its me. / im great')


# In[23]:


segment_desc('this is my tumblr blog // its me. // im great')


# In[45]:


segment_desc('this is my tumblr blog \ its me. \ im great')


# In[147]:


list_desc_data['segments'] = list(map(segment_desc, tqdm(list_desc_data['parsed_blog_description'].tolist())))


# In[148]:


# Remove lines with 0 or 1 segments

list_desc_data = list_desc_data[list_desc_data['segments'].map(lambda x: len(x) > 1)]
len(list_desc_data)


# In[149]:


list_desc_data.to_pickle('/usr0/home/mamille2/tumblr/data/list_descriptions.pkl')


# ## Visualize data

# In[62]:


segment_nums = [len(s) for s in list_desc_data['segments']]


# In[69]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of number of segments

plt.hist(segment_nums, bins=np.arange(min(segment_nums), max(segment_nums)+1, 1))
plt.title("Number of description segments")
plt.axis([1,15,0,int(4e5)])
plt.xticks(range(15))
plt.savefig('/usr0/home/mamille2/tumblr/results/num_desc_segments.png', dpi=300)
plt.show()


# In[107]:


segs = [s for seg in list_desc_data['segments'].tolist() for s in seg]
len(segs)


# In[71]:


seg_lens = [len(s) for s in segs]


# In[78]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of segment lengths

fig = plt.figure(figsize=(15,10))
plt.hist(seg_lens, bins=np.arange(min(seg_lens), max(seg_lens)+1, 1))
plt.title("Description segment lengths, in characters")
x_max = 50
plt.axis([1, x_max, 0, int(2.2e5)])
plt.xticks(range(x_max))
plt.savefig('/usr0/home/mamille2/tumblr/results/desc_segments_lens.png', dpi=100)
plt.show()


# ## Restrict length of segments

# In[150]:


segs = [s for seg in list_desc_data['segments'].tolist() for s in seg]
len(segs)


# In[135]:


# Explore diff segment sizes
egs = {}
for char_limit in [10,15,20,25,30]:
    egs[char_limit] = [r for r in segs if len(r) == char_limit]
    
len(egs)


# In[139]:


egs[25]


# In[151]:


# Restrict length of segments
restr_segs= {}
char_limit = 25
restr_segs[char_limit] = [r for r in segs if len(r) <= char_limit and len(r) > 1]
print(len(restr_segs[char_limit]))
print(len(set(restr_segs[char_limit])))

# Get a feel for restr_segs
restr_segs_ctr = {}
restr_segs_ctr[char_limit] = Counter(restr_segs[char_limit])
restr_segs_ctr[char_limit].most_common()


# In[120]:


pd.set_option('display.max_rows', 50)


# In[123]:


pd.set_option('display.max_colwidth', 999)


# In[145]:


# Filter spam

rm_list = [
    'relax and watch!',
    'my snapChat: lovewet9x',
    '/_        @will4i20    _\\',
    'my snapchat: sexybaby9x'
#     'Please wait' # not sure about this one
]


# In[126]:


with open('/usr0/home/mamille2/tumblr/data/desc_segments.pkl', 'wb') as f:
    pickle.dump(restr_segs, f)


# In[125]:


# Investigate weird description segments
# eg = '01'
eg = 'Please wait'

# eg_data = list_desc_data[list_desc_data['segments'].map(lambda x: eg in x)].loc[:,['parsed_blog_description', 'segments']]
eg_data = list_desc_data[list_desc_data['segments'].map(lambda x: eg in x)]
print(len(eg_data))
eg_data


# ## Remove HTML markup

# In[32]:


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()

def preprocess_post(post):
    
    # Strip html tags
    nohtml = strip_tags(post)
    
    # Tokenize with spaCy
    toks = [tok.text for tok in nlp.tokenizer(nohtml.lower())]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return toks


# In[36]:


desc_data['parsed_blog_description'] = desc_data['parsed_blog_description'].map(strip_tags)


# In[37]:


desc_data.to_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')


# # Make dataset for Tumblr word embedding training

# In[2]:


# Load 27M posts (73GB)
data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent100.pkl')
print(data.columns)
print(len(data))


# In[3]:


# Initialize tokenizer
nlp = spacy.load('en')


# In[4]:


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()

def preprocess_post(post):
    
    # Strip html tags
    nohtml = strip_tags(post)
    
    # Tokenize with spaCy
    toks = [tok.text for tok in nlp.tokenizer(nohtml.lower())]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return toks


# In[6]:


# Tokenize, preprocess all posts
data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))
data['body_toks'].head()


# # Select blog description embeddings that have 5 recent text posts

# In[2]:


recent5 = pd.read_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent5.pkl')
print(recent5.columns)
print(len(recent5))


# In[3]:


# Should be the same order as when saved out the 5 recent text post embeddings
tids = []

for tid, row_inds in tqdm(recent5.groupby('tumblog_id').groups.items()):
    tids.append(tid)
    
print(len(tids))
tids == sorted(tids)


# In[4]:


# Load blog descriptions
desc_data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')
len(desc_data)


# In[17]:


# Load blog descriptions embeddings
desc_embed = np.load('/usr0/home/mamille2/tumblr/data/desc_embeddings_avg.npy')
# desc_embed = np.load('/usr0/home/mamille2/tumblr/data/desc_embeddings_sum.npy')
desc_embed.shape


# In[5]:


# Get blog desc rows
desc_top5 = desc_data[desc_data['tumblog_id'].isin(tids)]
# len(desc_top5)

# Remove duplicates
desc_top5.drop_duplicates(subset=['tumblog_id'], inplace=True, keep='first')
len(desc_top5)

desc_top5.sort_values(['tumblog_id'], inplace=True)
top5_inds = desc_top5.index.tolist()
len(top5_inds)


# In[6]:


# Save out desc_top5 dataframe
desc_top5.to_pickle('/usr0/home/mamille2/tumblr/data/desc_recent5.pkl')


# In[19]:


selected = desc_embed[top5_inds]
selected.shape


# In[20]:


np.save('/usr0/home/mamille2/tumblr/data/desc_recent5_avg.npy', selected)
# np.save('/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_sum.npy', selected)


# # Preprocess data from Keith

# In[3]:


recent5 = pd.read_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent5.pkl')
print(recent5.columns)
print(len(recent5))


# In[13]:


# Initialize tokenizer
nlp = spacy.load('en')


# In[12]:


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()

def preprocess_post(post):
    
    # Strip html tags
    nohtml = strip_tags(post)
    
    # Tokenize with spaCy
    toks = [tok.text for tok in nlp.tokenizer(nohtml.lower())]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return toks


# In[15]:


# Tokenize, preprocess all posts
recent5['body_toks'] = list(map(preprocess_post, tqdm(recent5['body'].tolist())))
recent5['body_toks'].head()


# In[8]:


# Filter out empty tokenized posts
recent5 = recent5[recent5['body_toks'].map(lambda x: len(x) > 0)]
counts = recent5.groupby('tumblog_id').size()
counts5 = counts[counts >= 5]
recent5 = recent5[recent5['tumblog_id'].isin(counts5.index)]
len(recent5)


# In[9]:


recent5.to_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent5.pkl')


# ## Embed posts

# In[10]:


# wd_embed = ft.load_model('/usr0/home/mamille2/fasttext/en_wiki_stanford_model_300.bin')
wd_embed = ft.load_model('/usr0/home/mamille2/tumblr/data/halfday_ft.bin')


# In[11]:


# Initialize variables
MAX_LEN = 300 # max word len for ea blog post
EMBED_LEN = 300
# unkvec = np.mean(np.array(list(wordvec[100000].values())), axis=0) # unk vector as avg


# In[12]:


def post2vec(post, max_len, preprocessed=False):
    """ Takes post str or toks, returns vector """
    
    if not preprocessed:
        toks = preprocess_post(post)
    else:
        toks = post
    
    # Embed
    vec = np.mean(np.array([wd_embed[tok] for tok in toks[:max_len]]), axis=0)
#     vec = np.sum(np.array([wordvec[100000].get(tok, unkvec) for tok in toks[:max_len]]), axis=0)
    
    # Pad to get max_len
#     vec = np.pad(vec, ((0, max_len - vec.shape[0])), 'constant', constant_values=0)
    
    return vec


# In[13]:


# Make embeddings of posts (training data)

post_embeds = []

for tid, row_inds in tqdm(recent5.groupby('tumblog_id').groups.items()):
    posts = recent5.loc[row_inds, 'body_toks']
    total_vec = np.hstack([post2vec(post, MAX_LEN, preprocessed=True) for post in posts])
#     assert total_vec.shape == (EMBED_LEN * 5,)
    if not total_vec.shape == (EMBED_LEN * 5,):
        set_trace()
    post_embeds.append(total_vec)
    
post_embeds = np.vstack(post_embeds)
post_embeds.shape


# In[14]:


# Save embeddings of posts (training data)
# np.save('/usr0/home/mamille2/tumblr/data/halfday_5posts_embeds.npy', post_embeds)
# np.save('/usr0/home/mamille2/tumblr/data/halfday_5posts_embeds_avg.npy', post_embeds)
# np.save('/usr0/home/mamille2/tumblr/data/halfday_5posts_embed_sum.npy', post_embeds)
np.save('/usr0/home/mamille2/tumblr/data/recent5posts_embeds_avg.npy', post_embeds)


# # Filter most recent 100 posts from Keith to just those with descriptions

# In[10]:


# Load description usernames I want
with open('/usr0/home/mamille2/tumblr/data/endesc_ids.txt') as f:
    names = set([int(l) for l in f.read().splitlines()])

len(names)


# In[3]:


# Load 27M posts (73GB)
data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent100.pkl')
print(data.columns)
print(len(data))


# In[36]:


no_nans_inds = series_no_nan(data['tumblog_id'])
len(no_nans_inds)


# In[37]:


data_filtered = data.loc[no_nans_inds]


# In[35]:


def series_no_nan(s):
    nan_s = s.apply(no_nan)
    return nan_s[nan_s==True].index


# In[34]:


def no_nan(thg):
    """ Get list of non-nan indices """
    
    if isinstance(thg, str) and len(thg) > 0 and thg.isdigit():
        return True
    
    elif isinstance(thg, int):
        return True
    
    elif isinstance(thg, float) and np.isfinite(thg):
        return True
        
    else:
        return False


# In[39]:


# See username overlap with my set
data_filtered['tumblog_id'] = data_filtered['tumblog_id'].astype(int)
# recent100_tumblogs = set([int(t) for t in data_filtered['tumblog_id'].unique()])


# In[40]:


data_filtered['tumblog_id'].dtype


# In[41]:


# Save filtered data
data_filtered.to_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent100.pkl')


# In[42]:


# See username overlap with my set
recent100_tumblogs = set(data_filtered['tumblog_id'].unique())
len(recent100_tumblogs)


# In[44]:


len(names)


# In[43]:


print(len(recent100_tumblogs.intersection(names)))


# In[47]:


# Load blog descriptions
desc_data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')
print(len(desc_data.columns))
len(desc_data)


# In[49]:


# Check how many of those blogs I have descriptions for
desc_names = set(desc_data['tumblog_id'].unique())
len(recent100_tumblogs.intersection(desc_names))


# In[50]:


data_filtered = data_filtered[data_filtered['tumblog_id'].isin(desc_names)]
len(data_filtered)


# In[52]:


# See how many posts/user
gped = data_filtered.groupby('tumblog_id')
recent5 = gped.size()[gped.size()>=5]
recent5


# In[55]:


recent5_data = data_filtered[data_filtered['tumblog_id'].isin(recent5.index)]
len(recent5_data)


# In[56]:


recent5_data.to_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent5.pkl')


# In[4]:


recent5_data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent5.pkl')
print(recent5_data.columns)
print(len(recent5_data))


# In[5]:


recent5_data.sort_values(['tumblog_id', 'activity_time_epoch'], inplace=True, ascending=False)
recent5_data


# In[7]:


len(recent5['tumblog_id'].unique())


# In[6]:


# Filter to just the top 5 posts
recent5 = recent5_data.groupby('tumblog_id').head(5).reset_index(drop=True)
print(len(recent5))
recent5


# In[10]:


recent5.to_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent5.pkl')


# # Organize most recent 100 posts from Keith

# In[4]:


hdr = ['post_id', 'activity_time_epoch', 'tumblog_id',
       'post_title', 'post_short_url', 'post_type', 'post_caption',
       'post_format', 'post_note_count', 'created_time_epoch', 'updated_time_epoch',
       'is_submission', 'source_title', 'source_url', 'post_classifier', 'blog_classifier',
       'accepts_answers', 'reblogged_from_post_id', 'reblogged_from_metadata', 
       'root_post_id', 'body', 'mentions', 'post_tags']


# In[17]:


data_dirpath = '/usr0/home/mamille2/tumblr/data/textposts/'
data = []

for dirname in tqdm(os.listdir(data_dirpath)):
    dirpath = os.path.join(data_dirpath, dirname)
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        
        data.append(pd.read_table(fpath, header=None, error_bad_lines=False).values)

len(data)


# In[18]:


data_stack = np.vstack(data)
data_df = pd.DataFrame(data_stack, columns=hdr)
len(data_df)


# In[19]:


data_df.columns


# In[20]:


data_df.to_pickle('/usr0/home/mamille2/tumblr/data/textposts_recent100.pkl')


# # Train Tumblr fasttext embeddings

# ## Construct input file

# In[2]:


# Initialize tokenizer
nlp = spacy.load('en')


# In[3]:


data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/halfday_text.pkl')
print(data.columns)
print(len(data))


# In[4]:


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()

def preprocess_post(post):
    
    # Strip html tags
    nohtml = strip_tags(post)
    
    # Tokenize with spaCy
    toks = [tok.text for tok in nlp.tokenizer(nohtml.lower())]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return toks


# In[5]:


# Tokenize, preprocess all posts
data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))
data['body_toks'].head()


# In[7]:


print(data.columns)
print(len(data))


# In[8]:


# Save out preprocessed body_toks
data.to_pickle('/usr0/home/mamille2/tumblr/data/halfday_text.pkl')


# In[10]:


# Iteratively write out lines of posts (should filter to just blogs with English, but would be small set for halfday)
with open('/usr0/home/mamille2/tumblr/data/halfday_tokenized_text.txt', 'w') as f:
    for _, desc in tqdm(data['body_toks'].iteritems(), total=len(data)):
        if len(desc) > 0:
            f.write(' '.join(desc) + '\n')


# ## Train fastText model

# In[ ]:


model = ft.skipgram('/usr0/home/mamille2/tumblr/data/halfday_tokenized_text.txt', '/usr0/home/mamille2/tumblr/data/halfday_ft.model', 
                    dim=300,
                    thread=40)


# # Select blog description embeddings that have halfday 5 recent text posts

# In[10]:


halfday_desc_top5 = pd.read_pickle('/usr0/home/mamille2/tumblr/data/halfday_desc_recent5.pkl')
tids = halfday_desc_top5['tumblog_id'].unique()
len(tids)


# In[11]:


# Should be the same order as when saved out the 5 recent text post embeddings

tids = []

for tid, row_inds in tqdm(halfday_desc_top5.groupby('tumblog_id').groups.items()):
    tids.append(tid)
    
print(len(tids))
tids == sorted(tids)


# In[12]:


# Load blog descriptions
desc_data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')
len(desc_data)


# In[15]:


# Load blog descriptions embeddings
desc_embed = np.load('/usr0/home/mamille2/tumblr/data/desc_embeddings_avg.npy')
# desc_embed = np.load('/usr0/home/mamille2/tumblr/data/desc_embeddings_sum.npy')
desc_embed.shape


# In[6]:


# Get blog desc rows
desc_top5 = desc_data[desc_data['tumblog_id'].isin(tids)]
# len(desc_top5)

# Remove duplicates
desc_top5.drop_duplicates(subset=['tumblog_id'], inplace=True, keep='first')
len(desc_top5)

desc_top5.sort_values(['tumblog_id'], inplace=True)
top5_inds = desc_top5.index.tolist()
len(top5_inds)


# In[9]:


# Save out desc_top5 dataframe
desc_top5.to_pickle('/usr0/home/mamille2/tumblr/data/desc_recent5.pkl')


# In[13]:


# Load desc_top5 dataframe
desc_top5 = pd.read_pickle('/usr0/home/mamille2/tumblr/data/desc_recent5.pkl')
top5_inds = desc_top5.index.tolist()
len(top5_inds)


# In[16]:


selected = desc_embed[top5_inds]
selected.shape


# In[17]:


np.save('/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_avg.npy', selected)
# np.save('/usr0/home/mamille2/tumblr/data/desc_recent5_embeddings_sum.npy', selected)


# # Get n most recent posts in halfday from blogs that have descriptions

# In[2]:


halfday = pd.read_pickle('/usr0/home/mamille2/tumblr/data/halfday_text.pkl')
print(halfday.columns)


# In[3]:


# Load tumblog_ids of blogs with descriptions
with open('/usr0/home/mamille2/tumblr/data/endesc_ids.txt', 'r') as f:
    desc_ids = [int(i) for i in f.read().splitlines()]
    
len(desc_ids)


# In[4]:


# Filter halfday text to just blogs with descriptions
halfday_desc = halfday[halfday['tumblog_id'].isin(desc_ids)]
len(halfday_desc)


# In[6]:


# Group by tumblog id
counts = halfday_desc.groupby('tumblog_id').size()

# counts10 = counts[counts>=10]
# len(counts10)

counts5 = counts[counts>=5]
len(counts5)


# In[7]:


halfday_desc5 = halfday_desc[halfday_desc['tumblog_id'].isin(counts5.index.tolist())]
len(halfday_desc5)


# In[12]:


halfday_desc5.groupby('tumblog_id').size()


# In[8]:


# Restrict to 5 most recent text posts

# Sort by tumblog_id, timestamp
halfday_desc5.sort_values(['tumblog_id', 'activity_time_epoch'], inplace=True, ascending=False)
len(halfday_desc5)


# In[13]:


halfday_desc_top5 = halfday_desc5.groupby('tumblog_id').head(5).reset_index(drop=True)
print(len(halfday_desc_top5))
halfday_desc_top5


# In[15]:


halfday_desc_top5.to_pickle('/usr0/home/mamille2/tumblr/data/halfday_desc_recent5.pkl')


# # Get fasttext word embeddings for blog posts

# ## Get restricted vocab for posts

# In[ ]:


# Initialize tokenizer
nlp = spacy.load('en')


# In[28]:


posts = halfday_desc_top5['body'].tolist()
len(posts)


# In[20]:


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()


# In[26]:


def preprocess_post(post):
    
    # Strip html tags
    nohtml = strip_tags(post)
    
    # Tokenize with spaCy
    toks = [tok.text for tok in nlp.tokenizer(nohtml.lower())]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return toks


# In[33]:


# Tokenize, preprocess all posts
halfday_desc_top5['body_toks'] = list(map(preprocess_post, posts))
halfday_desc_top5['body_toks'].head()


# In[82]:


# Filter out blogs with empty posts (could add another post in outside of 5 most recent)
print(len(halfday_desc_top5))


# In[95]:


halfday_desc_top5 = halfday_desc_top5[halfday_desc_top5['body_toks'].map(lambda x: len(x) > 0)]
counts = halfday_desc_top5.groupby('tumblog_id').size()
counts5 = counts[counts >= 5]
halfday_desc_top5 = halfday_desc_top5[halfday_desc_top5['tumblog_id'].isin(counts5.index)]
len(halfday_desc_top5)


# In[96]:


# Save out tokenized posts
halfday_desc_top5.to_pickle('/usr0/home/mamille2/tumblr/data/halfday_desc_recent5.pkl')


# In[97]:


vocab = {}
corpus = halfday_desc_top5['body_toks'].tolist()
vocab_ctr = Counter([tok for text in corpus for tok in text])
print(len(vocab_ctr))

vocab[100000] = set([wd for wd,_ in vocab_ctr.most_common(100000)])
print(vocab_ctr.most_common(10))


# In[98]:


# Save out vocab
with open('/usr0/home/mamille2/tumblr/data/halfday_top5_vocab100000.pkl', 'wb') as f:
    pickle.dump(vocab, f)


# In[3]:


# Load vocab
with open('/usr0/home/mamille2/tumblr/data/desc_vocab100000.pkl', 'rb') as f:
    vocab = pickle.load(f)


# In[3]:


wd_embed = ft.load_model('/usr0/home/mamille2/fasttext/en_wiki_stanford_model_300.bin')
# wd_embed = ft.load_model('/usr0/home/mamille2/tumblr/data/halfday_ft.bin')


# In[99]:


# Get fasttext vectors for restricted vocab
wordvec = {}
wordvec[100000] = {}
for v in vocab[100000]:
    wordvec[100000][v] = wd_embed[v]
    
len(wordvec[100000]) # if not 100k, probably has some UNKs


# In[100]:


# Save out vectors for restricted vocab
with open('/usr0/home/mamille2/tumblr/data/halfday_top5_ftvecs100000.pkl', 'wb') as f:
    pickle.dump(wordvec, f)


# In[101]:


# Look at post length
print(np.mean([len(text) for text in corpus]))
print(np.std([len(text) for text in corpus]))


# ## Embed posts

# In[4]:


# Load tokenized posts
halfday_desc_top5 = pd.read_pickle('/usr0/home/mamille2/tumblr/data/halfday_desc_recent5.pkl')


# In[3]:


# wd_embed = ft.load_model('/usr0/home/mamille2/fasttext/en_wiki_stanford_model_300.bin')
wd_embed = ft.load_model('/usr0/home/mamille2/tumblr/data/halfday_ft.bin')


# In[3]:


# Load vectors for restricted vocab
with open('/usr0/home/mamille2/tumblr/data/halfday_top5_ftvecs100000.pkl', 'rb') as f:
    wordvec = pickle.load(f)


# In[6]:


# Initialize variables
MAX_LEN = 300 # max word len for ea blog post
EMBED_LEN = 300
# unkvec = np.mean(np.array(list(wordvec[100000].values())), axis=0) # unk vector as avg


# In[7]:


def post2vec(post, max_len, preprocessed=False):
    """ Takes post str or toks, returns vector """
    
    if not preprocessed:
        toks = preprocess_post(post)
    else:
        toks = post
    
    # Embed
    vec = np.mean(np.array([wd_embed[tok] for tok in toks[:max_len]]), axis=0)
#     vec = np.sum(np.array([wordvec[100000].get(tok, unkvec) for tok in toks[:max_len]]), axis=0)
    
    # Pad to get max_len
#     vec = np.pad(vec, ((0, max_len - vec.shape[0])), 'constant', constant_values=0)
    
    return vec


# In[8]:


# Make embeddings of posts (training data)

post_embeds = []

for tid, row_inds in tqdm(halfday_desc_top5.groupby('tumblog_id').groups.items()):
    posts = halfday_desc_top5.loc[row_inds, 'body_toks']
    total_vec = np.hstack([post2vec(post, MAX_LEN, preprocessed=True) for post in posts])
    assert total_vec.shape == (EMBED_LEN * 5,)
#     if not total_vec.shape == (EMBED_LEN * 5,):
#         set_trace()
    post_embeds.append(total_vec)
    
post_embeds = np.vstack(post_embeds)
post_embeds.shape


# In[9]:


# Save embeddings of posts (training data)
# np.save('/usr0/home/mamille2/tumblr/data/halfday_5posts_embeds.npy', post_embeds)
np.save('/usr0/home/mamille2/tumblr/data/halfday_5posts_embeds_avg.npy', post_embeds)
# np.save('/usr0/home/mamille2/tumblr/data/halfday_5posts_embed_sum.npy', post_embeds)


# # Get fasttext word embeddings for blog descriptions

# In[2]:


# wd_embed = ft.load_model('/usr0/home/mamille2/fasttext/en_wiki_stanford_model_300.bin')
wd_embed = ft.load_model('/usr0/home/mamille2/tumblr/data/halfday_ft.bin')


# In[38]:


# Load Tumblr blog descriptions
data = pd.read_csv('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.csv')
print(len(data))
print(data['tokenized_blog_description'].dtype)


# ## Tokenize blog descriptions

# In[6]:


nlp = spacy.load('en')


# In[34]:


# Tokenize with nltk

blog_descs_toks = []

for desc in tqdm(blog_descs):
    toks = nltk.word_tokenize(desc.lower())
    blog_descs_toks.append(toks)


# In[77]:


# Tokenize with spaCy

blog_descs_toks = []

for desc in tqdm(blog_descs):
    toks = [tok.text for tok in nlp.tokenizer(desc.lower())]
    blog_descs_toks.append(toks)


# ## Save out tokenized descriptions

# In[78]:


data['tokenized_blog_description'] = blog_descs_toks
type(data.loc[1,'tokenized_blog_description'])


# In[79]:


data.to_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')


# ## Get restricted vocab and fasttext vectors for restricted vocab

# In[10]:


# Load tokenized data
data = pd.read_pickle('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.pkl')
blog_descs_toks = data['tokenized_blog_description'].tolist()
print(len(data))


# In[4]:


vocab = {}
vocab_ctr = Counter([tok for desc in blog_descs_toks for tok in desc])
print(len(vocab_ctr))

vocab[100000] = set([wd for wd,_ in vocab_ctr.most_common(100000)])
print(vocab_ctr.most_common(10))


# In[5]:


# Save out vocab
with open('/usr0/home/mamille2/tumblr/data/desc_vocab100000.pkl', 'wb') as f:
    pickle.dump(vocab, f)


# In[3]:


# Load vocab
with open('/usr0/home/mamille2/tumblr/data/desc_vocab100000.pkl', 'rb') as f:
    vocab = pickle.load(f)


# In[4]:


# Get fasttext vectors for restricted vocab
wordvec = {}
wordvec[100000] = {}
for v in vocab[100000]:
    wordvec[100000][v] = wd_embed[v]
    
len(wordvec[100000])


# In[5]:


# Save out vectors for restricted vocab
with open('/usr0/home/mamille2/tumblr/data/desc_ftvecs100000.pkl', 'wb') as f:
    pickle.dump(wordvec, f)


# ## See lengths of blog descriptions

# In[53]:


desc_lens = data['tokenized_blog_description'].map(lambda x: len(x)).values
len(desc_lens)


# In[54]:


print(np.mean(desc_lens))
print(np.std(desc_lens))
print(max(desc_lens))
print(min(desc_lens))


# ## Save description embeddings

# In[6]:


# Load vectors for restricted vocab
with open('/usr0/home/mamille2/tumblr/data/desc_ftvecs100000.pkl', 'rb') as f:
    wordvec = pickle.load(f)


# In[7]:


MAX_WORD_LEN = 50
EMBED_LEN = 300

unkvec = np.mean(np.array(list(wordvec[100000].values())), axis=0) # unk vector as avg
unkvec.shape


# In[14]:


# def desc2mat(desc_toks, wordvecs, unkvec):
# def desc2mat(desc_toks):
    
#     # fill in w word vectors
#     desc_mat = np.array([wordvec[100000].get(tok, unkvec) for tok in desc[:MAX_WORD_LEN]])

#     # pad to get to 50 words
#     desc_mat = np.pad(desc_mat, ((0, MAX_WORD_LEN - desc_mat.shape[0]), (0,0)), 'constant', constant_values=0)

#     return desc_mat

def desc2vec(desc_toks):
    
    # averaged/summed word vectors
#     desc_vec = np.mean(np.array([wordvec[100000].get(tok, unkvec) for tok in desc_toks[:MAX_WORD_LEN]]), axis=0)
#     desc_vec = np.sum(np.array([wordvec[100000].get(tok, unkvec) for tok in desc_toks[:MAX_WORD_LEN]]), axis=0)
    desc_vec = np.mean(np.array([wd_embed[tok] for tok in desc_toks[:MAX_WORD_LEN]]), axis=0)
    
    return desc_vec
    
    # concatenated word vectors
    # fill in w word vectors
#     desc_vec = np.array([wordvec[100000].get(tok, unkvec) for tok in desc_toks[:MAX_WORD_LEN]]).ravel()

#     # pad to get to 50 words
#     desc_vec = np.pad(desc_vec, ((0, EMBED_LEN*MAX_WORD_LEN - desc_vec.shape[0])), 'constant', constant_values=0)

# #     return desc_vec
#     return csr_matrix(desc_vec)


# In[15]:


desc_embeds = []

# for desc in tqdm(data['tokenized_blog_description'].tolist()[:3]):
# desc = data['tokenized_blog_description'].tolist()[2]
tokenized_descs = data['tokenized_blog_description'].tolist()

# for desc in tqdm(tokenized_descs[:1000]):
for desc in tqdm(tokenized_descs):
#     desc_embeds = vstack([desc_embeds, desc2vec(desc)])
    desc_embeds.append(desc2vec(desc))

# pool = Pool(30)
# pool.map(desc2mat, tokenized_descs)


# In[16]:


descs_emb = np.array(desc_embeds)
descs_emb.shape


# In[17]:


# Save description embeddings
outpath = '/usr0/home/mamille2/tumblr/data/desc_embeddings_avg.npy'
# outpath = '/usr0/home/mamille2/tumblr/data/desc_embeddings_sum.npy'

np.save(outpath, descs_emb)


# In[ ]:


descs_emb = vstack(desc_embeds)
descs_emb.shape


# In[ ]:


get_ipython().run_line_magic('xdel', 'desc_embeds')


# In[ ]:


# Save description embeddings
outpath = '/usr0/home/mamille2/tumblr/data/desc_embeddings.npz'

scipy.sparse.save_npz(outpath, descs_emb)


# # Print out blog description tumblog_ids

# In[43]:


with open('/usr0/home/mamille2/tumblr/data/endesc_ids.txt', 'w') as f:
    for id in data['tumblog_id'].tolist():
        f.write(str(int(id)) + '\n')


# # Filter blog description data

# In[ ]:


# Remove blog descriptions with no ASCII chars
data = pd.read_csv('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.csv')
len(data)


# In[5]:


string.printable


# In[6]:


ascii_data = data[data['parsed_blog_description'].map(lambda x: any([c in string.printable for c in x]))]
len(ascii_data)


# In[8]:


'★인생역전이벤트진행중★' in ascii_data['parsed_blog_description'].tolist()


# In[9]:


ascii_data.to_csv('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.csv', index=False)


# In[3]:


# Remove nans from parsed en_nan blog descs
data = pd.read_csv('/usr0/home/mamille2/tumblr/data/en_nan_blog_descriptions.csv')
len(data)


# In[4]:


# Build just en blog descriptions
en_data = data[data['language'].map(lambda x: str(x).startswith('en'))]
len(en_data)


# In[5]:


en_data.to_csv('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.csv', index=False)


# In[4]:


data = data[data['parsed_blog_description'].map(lambda x: isinstance(x, str) and len(x)>0)]
len(data)


# In[6]:


data.to_csv('/usr0/home/mamille2/tumblr/data/en_nan_blog_descriptions.csv', index=False)


# In[4]:


# Load data
data = pd.read_csv('/usr0/home/mamille2/tumblr/data/blog_descs_all.csv')


# In[5]:


data.columns


# In[12]:


len(data)


# In[ ]:


en_data = data[data['language'].map(lambda x: str(x).startswith('en') or isinstance(x, float))]
en_data = en_data[en_data['tumblog_id'].map(lambda x: not np.isnan(x))]
# en_data = data[data['language'].map(lambda x: str(x).startswith('en'))]
print(len(en_data))


# In[26]:


en_data[en_data['tumblr_blog_description'].map(lambda x: isinstance(x, float))].head()


# In[28]:


en_data = en_data[en_data['tumblr_blog_description'].map(lambda x: isinstance(x, str) and len(x) > 0)]
len(en_data)


# In[22]:


en_data.loc[:5, ['tumblog_id', 'tumblr_blog_name', 'tumblr_blog_title', 'language', 'tumblr_blog_description']]


# # Format blog descriptions

# In[2]:


en_data = pd.read_csv('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.csv')
len(en_data)


# In[4]:


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    text = str(html).strip()
    s.feed(text)
    return s.get_data()


# In[5]:


en_data['parsed_blog_description'] = en_data['tumblr_blog_description'].map(strip_tags)


# In[9]:


pd.set_option('display.max_colwidth', 999)


# In[10]:


en_data.loc[:, ['parsed_blog_description']]


# In[11]:


en_data.to_csv('/usr0/home/mamille2/tumblr/data/en_blog_descriptions.csv', index=False)


# In[16]:


chunksize = int(1e6)
offset = 0

for _ in tqdm(range(int(len(blog_descs_col)//chunksize)+1), desc="total"):
# for _ in range(int(len(blog_descs_col)//chunksize)+1)[:1]:
    blog_descs = []
    
    chunk = blog_descs_col[offset : min(offset+chunksize, len(blog_descs_col))]
    
    for i, b in enumerate(tqdm(chunk)):
#         blog_descs[i] = (strip_tags(b, s))
        blog_descs.append(strip_tags(b, s))

    # Save out preprocessed blog descriptions
    with open('/usr0/home/mamille2/tumblr/data/blog_descriptions.txt', 'a+') as f:
        for b in blog_descs:
            if len(b) > 0:
                f.write(b + '\n')
            
    offset = offset+chunksize


# In[16]:


chunksize = int(1e6)
offset = 0

for _ in tqdm(range(int(len(blog_descs_col)//chunksize)+1), desc="total"):
# for _ in range(int(len(blog_descs_col)//chunksize)+1)[:1]:
    blog_descs = []
    
    chunk = blog_descs_col[offset : min(offset+chunksize, len(blog_descs_col))]
    
    for i, b in enumerate(tqdm(chunk)):
#         blog_descs[i] = (strip_tags(b, s))
        blog_descs.append(strip_tags(b, s))

    # Save out preprocessed blog descriptions
    with open('/usr0/home/mamille2/tumblr/data/blog_descriptions.txt', 'a+') as f:
        for b in blog_descs:
            if len(b) > 0:
                f.write(b + '\n')
            
    offset = offset+chunksize


# ## Unnecessary multithreading

# In[6]:


import warnings; warnings.simplefilter('ignore')


# In[12]:


def process_desc(chunk):
    s = MLStripper()
    blog_descs = []
    
    for i, b in enumerate(chunk):
#         blog_descs[i] = (strip_tags(b, s))
        blog_descs.append(strip_tags(b, s))
    
    # Save out preprocessed blog descriptions
    with open('/usr0/home/mamille2/tumblr/data/blog_descriptions.txt', 'a+') as f:
#     with open('/usr0/home/mamille2/tumblr/data/blog_descriptions{}.txt'.format(random.random()), 'a+') as f:
        for b in blog_descs:
            if len(b) > 0:
                f.write(b + '\n')


# In[ ]:


# Multithreading

chunksize = int(10000)
offset = 0

for _ in tqdm(range(int(len(blog_descs_col)//chunksize)+1), desc="total"):
    chunk = blog_descs_col[offset : min(offset+chunksize, len(blog_descs_col))]
    
#     pool = Pool(20)
#     try:
#         pool.map(process_desc, chunk)
#     except TypeError as e:
#         continue

    output = mp.Queue()
    processes = [mp.Process(target=process_desc, args=) for x in range(10)]
    
#     for i, b in enumerate(tqdm(chunk)):
# #         blog_descs[i] = (strip_tags(b, s))
#         blog_descs.append(strip_tags(b, s))
    
    # Save out preprocessed blog descriptions
#     with open('/usr0/home/mamille2/tumblr/data/blog_descriptions.txt', 'a+') as f:
#         for b in blog_descs:
#             if len(b) > 0:
#                 f.write(b + '\n')
            
    offset = offset+chunksize


# In[ ]:


get_ipython().run_line_magic('xdel', 'blog_descs_col')


# In[18]:


list(blog_descs)[:10]


# In[14]:


blog_descs = [b for b in blog_descs if len(b) > 0]# remove empties
print(len(blog_descs))
blog_descs[:10]


# In[15]:


len(blog_descs[8])


# In[4]:


blog_descs[7]


# In[8]:


strip_tags(blog_descs[7])


# In[ ]:


# Save out preprocessed blog descriptions
with open('~/tumblr/data/blog_descs.txt', 'wb') as f:
    for b in blog_descs:
        f.write(b + '\n')


# # Get filled-out blog descriptions from Keith

# In[21]:


desc_path = '/usr2/kmaki/tumblr/blogs/tumblr_blog_descriptions.csv'

chunksize = 10 ** 6

chunked = pd.read_csv(desc_path, chunksize=chunksize, iterator=True, error_bad_lines=False, warn_bad_lines=False)


# In[10]:


desc_data = chunked.get_chunk(1000)
desc_data.head()


# In[19]:


# Calculate number of filled-out blog descriptions

n_descs = 0

for i, chunk in enumerate(chunked):
    print("{}M".format(i))
    n_descs += chunk['tumblr_blog_description'].count()

n_descs


# In[22]:


# Save out just the filled in blog description rows

for i, chunk in enumerate(chunked):
    print("{}M...".format(i))
    descs = chunk[chunk['tumblr_blog_description'].map(lambda x: isinstance(x, str))]
    
    outpath = '/usr2/mamille2/tumblr/data/blog_descriptions/from_keith/descs_{:02}M.csv'.format(i)
    descs.to_csv(outpath, index=False, header=False)
    print("Wrote to {}".format(outpath))


# # Assemble blog descriptions scraped from Tumblr API

# In[2]:


data_dirpath = '../../data/blog_descriptions/'

desc_text = []

for fname in os.listdir(data_dirpath):
    print(fname, end=': ')
    fpath = os.path.join(data_dirpath, fname)
    
    if fname.startswith('just_blog_desc'):
        with open(fpath, 'rb') as f:
                descs = pickle.load(f)
                desc_text.extend(descs)
                
        print(len(descs))
                
    elif fname.startswith('sample_blog_desc'):
        with open(fpath, 'rb') as f:
                descs = pickle.load(f)
                desc_text.extend(descs.values())
        print(len(descs))
                
# remove empty strings, duplicates
desc_text = list(set([d for d in desc_text if len(d) > 0]))
len(desc_text)


# # Stats on blog descriptions

# In[21]:


# Length
print("Mean #words: {}".format(np.mean([len(desc.split()) for desc in desc_text])))
print("Std dev #words: {}".format(np.std([len(desc.split()) for desc in desc_text])))
print("Max #words: {}".format(np.max([len(desc.split()) for desc in desc_text])))


# In[20]:


wd_ctr = Counter([len(desc.split()) for desc in desc_text])
wd_ctr


# In[32]:


docs_l = [d.split() for d in desc_text]
wds = [w.lower() for d in docs_l for w in d]
wds_ctr = Counter(wds)
for i, (wd, c) in enumerate(wds_ctr.most_common(1000)):
    print('{}\t\t{}\t{}'.format(wd,c, i+1))


# In[35]:


search_term = 'rp'
for d in [d for d in docs_l if search_term in d]:
    print(' '.join(d))


# In[39]:


for r in random.sample(desc_text, 10):
    print(r)
    print()


# # Investigate scraping user descriptions

# In[2]:


data = pd.read_table('/usr2/kmaki/tumblr/otherkin_antikin.tsv', error_bad_lines=False)


# In[4]:


have_urls = data[data['post_short_url'].map(lambda x: isinstance(x, str))]
len(have_urls)


# In[22]:


# Get page and find user description
url = data.loc[144883, 'post_short_url']
soup = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
soup.prettify()


# In[23]:


type(soup.span)


# In[24]:


soup.span.contents


# In[27]:


soup.span.text


# In[26]:


url


# In[7]:


# Assemble unique tumblog_ids, then get pages from that
tumblog_ids = have_urls['tumblog_id'].unique()
len(tumblog_ids)


# In[11]:


tumblog_ids[:10]


# In[28]:


# Scrape user descriptions
outlines = list()
for i, tumblog_id in enumerate(tumblog_ids[:10]):
    url = data.loc[(data['tumblog_id']==tumblog_id).idxmax(), 'post_short_url']
    print('{0}: {1}'.format(i,url))
    
    try:
        soup =  BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
    except urllib.error.HTTPError as _:
        print('404\n')
        continue
        
    # Want match all of them that returns the 'right' info, or perhaps biggest string
    
    # soup.span name='description'
    found_tag = soup.findAll("span", {"class": "description"})
    if found_tag:
        user_desc = found_tag.text
        print(user_desc)
    else:
        print()
        continue
        
    outlines.append([url, user_desc])
    print()
        
len(outlines)


# In[77]:


pprint(outlines)


# In[63]:


have_urls['post_short_url'].values[1]


# In[64]:


have_urls['post_short_url'].values[2]


# In[65]:


have_urls['post_short_url'].values[3]


# # Antikin/otherkin data

# In[18]:


data = pd.read_table('/usr2/kmaki/tumblr/otherkin_antikin.tsv', error_bad_lines=False)


# In[22]:


print(len(data))
print(data.columns)


# In[23]:


data['post_tags'][:10]


# In[28]:


data.iloc[:10,:]


# In[22]:


def split_tags(tags, sep=',', combine_words=False):
    """ 
        Returns list of tags from Tumblr's format (default) or Python list format 
        
        Args:
            combine_words: whether or not to return multi-word tags combined, with no spaces
    """
    
    # Check for NaN
    if not isinstance(tags, str):
        return []
    
    spaced_tags = [tag[1:-1] for tag in tags[1:-1].split(sep)]
    if not combine_words:
        return spaced_tags
    
    else:
        nospace_tags = [tag.replace(' ', '') for tag in spaced_tags]
        return nospace_tags


# In[29]:


# Distribution of tags
anti_p = re.compile('anti[-\s_]?kin', re.IGNORECASE)
other_p = re.compile('other[-\s_]?kin', re.IGNORECASE)
anti_rows = data[data['post_tags'].map(lambda x: has_tag_re(x, anti_p))].index
other_rows = data[data['post_tags'].map(lambda x: has_tag_re(x, other_p))].index

len_anti = len(anti_rows)
len_other = len(other_rows)
len_both = len(set(anti_rows).intersection(set(other_rows)))
print(len_anti)
print(len_other)
print(len_both)


# In[49]:


len(set(anti_rows).intersection(set(other_rows)))


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
v = venn2(subsets=(len_other-len_both,len_anti-len_both,len_both), set_labels=('otherkin', 'antikin', 'both'))
# v.get_label_by_id('10').set_text('')
# v.get_label_by_id('01').set_text('')
v.get_label_by_id('11').set_text('')
# venn2_circles(subsets=(90449-884,2332-884,884))
plt.savefig('/usr2/mamille2/tumblr/otherkin_antikin.png', dpi=400)


# In[45]:


def has_tag(text, tags):
    """ Takes in a Tumblr hashtag list and list of tags and returns whether any of them are present """
    
    return any(tag in split_tags(text) for tag in tags)


# In[25]:


def has_tag_re(text, regex):
    """ Takes in a Tumblr hashtag text list and regex and returns whether there are any matches """
    
    if not isinstance(text, str):
        return False
    
    else:
        return any(re.match(regex, tag) for tag in split_tags(text))


# ## Investigate rows with no tags--is newlines in post_content causing error

# In[27]:


notags = data[data['post_tags'].map(lambda x: isinstance(x, float))]
len(notags)


# In[32]:


# No tag data has *kin somewhere else?
notags_types = {}

# for field in ['post_id', 'body', 'post_caption', 'post_content']:
for field in ['post_id', 'post_title']:
    notags_types[field] = notags[notags[field].map(lambda x: 'kin' in x if isinstance(x, str) else False)]
    print("{0}:\t{1}".format(field, len(notags_types[field])))


# In[34]:


pd.set_option('display.max_colwidth', 999)


# In[38]:


pd.set_option('display.max_columns', 999)


# In[39]:


notags.iloc[:100,:]


# # Look at I'm ___ trash

# In[ ]:


# Load data
data = pd.read_pickle('/usr2/mamille2/tumblr/data/halfday_text.pkl')
print(data.columns)
print(len(data))


# In[ ]:


# Search for I'm ___ trash

trash_re = re.compile(r"(?:I'm|i'm|Im|im).{1,30}(?:trash|Trash|TRASH)")

def trash_present(text):
    """ Return T/F for presence I'm ____ trash """
    
    if not isinstance(text, str):
        return False
    
    if re.search(trash_re, text):
        return True
    else:
        return False

def trash_match(text):
    """ Return match object for I'm ____ trash """
    
    if not isinstance(text, str):
        return None
    
    return re.search(trash_re, text)


# In[ ]:


trash_present("I'm star wars trash")


# In[ ]:


trash_present("yada")


# In[ ]:


trash_data = data[data['body'].map(trash_present)]
len(trash_data)


# In[ ]:


trash_data


# In[ ]:


type(trash_data.loc[:,'body'])


# In[ ]:


pd.set_option('display.max_colwidth', 999)


# In[ ]:


trash_data.loc[:,['body']]

