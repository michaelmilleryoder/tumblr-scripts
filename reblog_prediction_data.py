
# coding: utf-8

# In[1]:


import pandas as pd
import os
# data_dirpath = '/usr2/mamille2/tumblr/data' # erebor
data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data' # misty


# # Using data that already have for reblog prediction

# ## Filter to only users who have ever reblogged anything in our set

# In[2]:


from pprint import pprint

# Load data
data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))
print(len(data))
pprint(sorted(data.columns.tolist()))


# In[4]:


# Randomly sampled posts from followers to build dataset, but here just consider reblogs that were randomly sampled

reblogs = data[data['reblogged']==True]
len(reblogs)


# In[5]:


reblog_followers = set(reblogs['tumblog_id_follower'].unique())
len(reblog_followers)


# In[6]:


data_rebloggers = data[data['tumblog_id_follower'].isin(reblog_followers)]
len(data_rebloggers)


# In[7]:


data_rebloggers.to_pickle(os.path.join(data_dirpath, 'posts_descs_3m_reblog_restricted.pkl'))


# ## Reframe reblog prediction to be TRUE if a user ever reblogged another

# In[2]:


from pprint import pprint

# Load data
data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))
print(len(data))
pprint(sorted(data.columns.tolist()))


# In[53]:


from tqdm import tqdm_notebook as tqdm

reblog_opportunities = {}

for followee_id, follower_id, reblogged in tqdm(list(zip(data['tumblog_id_followee'], data['tumblog_id_follower'], data['reblogged']))):
    if not (followee_id, follower_id) in reblog_opportunities:
        reblog_opportunities[(followee_id, follower_id)] = reblogged
        
    elif reblog_opportunities[(followee_id, follower_id)] == False:
        reblog_opportunities[(followee_id, follower_id)] = reblogged
        
    else: # value stored is True
        continue


# In[59]:


reblog_opportunities_df = pd.DataFrame([(followee_id, follower_id, reblogged) for ((followee_id, follower_id), reblogged) in reblog_opportunities.items()],
                                      columns = ['tumblog_id_followee', 'tumblog_id_follower', 'reblogged_ever'])
reblog_opportunities_df


# In[61]:


len(reblog_opportunities_df)


# In[62]:


len(data)


# In[60]:


sum(reblog_opportunities_df['reblogged_ever'])


# ## Restrict data to posts that occurred in 24-h window around when users reblogged something (didn't save time someone reblogged something, just original post time)

# In[2]:


from pprint import pprint

# Load data
data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))
print(len(data))
pprint(sorted(data.columns.tolist()))


# In[3]:


# Randomly sampled posts from followers to build dataset, but here just consider reblogs that were randomly sampled

reblogs = data[data['reblogged']==True]
user_reblogs = list(zip(reblogs['tumblog_id_follower'], reblogs['activity_time_epoch_post'])) # format (user, reblog_ts)
len(user_reblogs)


# In[4]:


from collections import defaultdict
from datetime import datetime

user_reblog_dict = defaultdict(list)
for user, ts in user_reblogs:
    user_reblog_dict[user].append(datetime.fromtimestamp(int(ts)/1000))
    
len(user_reblog_dict)


# In[5]:


data['activity_time_epoch_post'].dtype


# In[9]:


from datetime import datetime

eg_ts = int(data['activity_time_epoch_post'].sample(1))
print(eg_ts)
datetime.fromtimestamp(eg_ts/1000)


# In[16]:


users_ts = list(zip(data['tumblog_id_follower'], data['activity_time_epoch_post']))
users_ts[:100]


# In[18]:


# Why are so many activity timestamps the same?
print(len(data))
print(len(data['activity_time_epoch_post'].unique()))


# In[20]:


time_cols = [col for col in data.columns if 'time' in col]
time_cols


# In[23]:


print(len(data))

for col in time_cols:
    print(f'{col}\t{len(data[col].unique())}')
#     print()


# In[25]:


print(len(data['tumblog_id_followee'].unique()))
print(len(data['tumblog_id_follower'].unique()))


# In[29]:


eg_ts = 1486791859000
eg = data[data['activity_time_epoch_post'] == eg_ts]
len(eg)


# In[31]:


list(data.columns)


# In[30]:


eg


# In[33]:


eg['reblogged_from_post_id'].unique()


# In[34]:


eg['tumblog_id_follower']


# In[35]:


eg['reblogged'].value_counts()


# In[38]:


eg['tumblog_id_followee'].unique()


# In[36]:


len(reblogs)


# In[39]:


len(reblogs['activity_time_epoch_post'].value_counts())


# In[44]:


reblogs['activity_time_epoch_post'].value_counts().index


# In[50]:


pd.set_option('display.max_colwidth', -1)


# In[51]:


eg_reblog = reblogs[reblogs['activity_time_epoch_post']==1489042482000]
eg_reblog


# In[49]:


len(eg_reblog['tumblog_id_follower'].unique())


# In[14]:


# Filter to 12 h before and after reblogs by those users
from datetime import datetime, timedelta

def around_reblog(user, post_ts):
    dt_post_ts = datetime.fromtimestamp(int(post_ts)/1000)
    
    ans = False
    
    for reblog_ts in user_reblog_dict[user]:
        if  dt_post_ts < reblog_ts + timedelta(hours=12) and             dt_post_ts > reblog_ts - timedelta(hours=12):
        
            ans = True
            break
    
    return ans


# In[15]:


reblog_within_time = data[[around_reblog(u, ts) for u, ts in users_ts]]
print(len(reblog_within_time))


# ## Load followers

# In[2]:


follower_data = pd.read_pickle(os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl'))
print(len(follower_data))
print(follower_data.columns)


# ## Load followees

# In[2]:


follow_data = pd.read_pickle(os.path.join(data_dirpath, 'follow_data_recent100.pkl'))
print(len(follow_data))
print(follow_data.columns)


# ## Get followees (now a script)

# In[11]:


# Test follow structure

follow_fpath = '/usr2/kmaki/tumblr/follow_network.tsv' # 99G
chunksize = int(1e5)
hdr = ['tumblog_id', 'followed_tumblog_id', 'activity_time_epoch']
chunked = pd.read_csv(follow_fpath, sep='\t', header=None, names=hdr, iterator=True)
follow_data = chunked.get_chunk(chunksize)
print(len(follow_data))
print(follow_data.columns)


# In[3]:


# See what follow structure can find for set of followers
from tqdm import tqdm_notebook as tqdm
import numpy as np

follow_fpath = '/usr2/kmaki/tumblr/follow_network.tsv' # 99G
chunksize = int(1e5)
total_iters = int(np.ceil(3220088178 / chunksize))
hdr = ['tumblog_id', 'followed_tumblog_id', 'activity_time_epoch']
chunked = pd.read_csv(follow_fpath, sep='\t', header=None, names=hdr, chunksize=chunksize)

stx_follower_ids = set()
# follower_ids = follower_data['tumblog_id'].tolist()
# followee_data = []

for chunk in tqdm(chunked, total=total_iters):
    stx_follower_ids |= set(chunk['tumblog_id'].unique())
#     followee_data.append(chunk[chunk['tumblog_id'].isin(follower_ids)].values)
    
    
print(len(stx_follower_ids))

follower_ids = stx_follower_ids.intersection(set(follower_data['tumblog_id']))
len(follower_ids) # Have at least one piece of follow information for these users


# In[5]:


# Save out follower ids
with open(os.path.join(data_dirpath, 'halfday_followers_with_descriptions.txt'), 'w') as f:
    for i in sorted(follower_ids):
        f.write(f'{i}\n')


# In[4]:


# Get followee data
follow_fpath = '/usr2/kmaki/tumblr/follow_network.tsv' # 99G
chunksize = int(1e5)
total_iters = int(np.ceil(3220088178 / chunksize))
hdr = ['tumblog_id', 'followed_tumblog_id', 'activity_time_epoch']
chunked = pd.read_csv(follow_fpath, sep='\t', header=None, names=hdr, chunksize=chunksize)

followee_data = [] # list of np arrays to be concatenated

for chunk in tqdm(chunked, total=total_iters):
    followee_data.append(chunk[chunk['tumblog_id'].isin(follower_ids)].values)
    
follower_follow_data = pd.DataFrame(np.vstack(followee_data), columns=hdr)
len(follower_follow_data)


# In[29]:


# Save follow data from followers for which have descriptions
out_fname = 'follow_data_recent100.pkl'

follower_follow_data.to_pickle(os.path.join(data_dirpath, out_fname))


# ## Get followee descriptions

# In[3]:


# Load blog descriptions
descs = pd.read_pickle(os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl'))
print(len(descs))
print(descs.columns)


# In[5]:


followees = set(follow_data['followed_tumblog_id'])
print(len(followees))

followees_with_descriptions = followees.intersection(set(descs['tumblog_id']))
print(len(followees_with_descriptions))


# ##  Select posts from followees, posted after followers followed them

# In[6]:


# Load post data

post_fname = 'textposts_recent100.pkl' # ~75 GB in memory
post_data = pd.read_pickle(os.path.join(data_dirpath, post_fname))
print(len(post_data))
print(post_data.columns)


# In[8]:


followee_posts = post_data[post_data['tumblog_id'].isin(followees_with_descriptions)] # all posts regardless of time posted
print(len(followee_posts))


# In[9]:


# Save followee posts
followee_posts.to_pickle(os.path.join(data_dirpath, 'followee_posts_recent100.pkl'))


# In[3]:


# Load followee posts
# followee_posts = pd.read_pickle(os.path.join(data_dirpath, 'followee_posts_recent100.pkl'))
followee_posts = pd.read_pickle(os.path.join(data_dirpath, 'reblog_followee_posts_recent100.pkl'))
print(len(followee_posts))
print(followee_posts.columns)


# ### Merge text and user info for quick prediction (duplicate posts for followees with multiple followers in the dataset)
# Would filter followee posts based on timestamp of follows here

# #### Make a dictionary of followees -> followers

# In[3]:


# follow_data = pd.read_pickle(os.path.join(data_dirpath, 'follow_data_recent100.pkl'))
follow_data = pd.read_pickle(os.path.join(data_dirpath, 'reblog_follow_data_recent100.pkl'))
print(len(follow_data))
print(follow_data.columns)


# In[43]:


gped = follow_data.set_index('tumblog_id').groupby('followed_tumblog_id')
gped.groups

len(gped.groups)

follow_dict = {key: list(gped.groups[key]) for key in gped.groups}
print(len(follow_dict))


# In[25]:


# Average #followers/follower
len([d for l in follow_dict.values() for d in l]) / len(follow_dict)


# In[44]:


# Save
import pickle

# with open(os.path.join(data_dirpath, 'follow_recent100_dict.pkl'), 'wb') as f:
with open(os.path.join(data_dirpath, 'reblog_follow_recent100_dict.pkl'), 'wb') as f:
    pickle.dump(follow_dict, f)


# In[4]:


# Load follow dict
import pickle

# with open(os.path.join(data_dirpath, 'follow_recent100_dict.pkl'), 'wb') as f:
# followee: [followers]
with open(os.path.join(data_dirpath, 'reblog_follow_recent100_dict.pkl'), 'rb') as f: # followers are all rebloggers
    follow_dict = pickle.load(f)


# In[6]:


# Sample from posts

# posts = followee_posts.sample(int(1e4))
posts = followee_posts.sample(int(1e5))
print(len(posts))
posts.columns


# In[9]:


# Merge posts with follower/followee names, sample max 10 followers
import random
from tqdm import tqdm_notebook as tqdm

def safe_sample(l, t):
    if len(l) > t:
        return random.sample(l, t)
    else:
        return l

# posts['followers'] = posts['tumblog_id'].map(lambda x: safe_sample(follow_dict[x], 10))
posts['followers'] = posts['tumblog_id'].map(lambda x: follow_dict[x]) # All followers should be rebloggers
# posts['followers'] = list(map(lambda x: [u for u in follow_dict[x] if u in reblog_followers], tqdm(posts['tumblog_id']))) # Keep reblog followers

s = posts.apply(lambda x: pd.Series(x['followers']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'follower'
data = posts.join(s)
data.reset_index(drop=True, inplace=True)
data.drop('followers', axis=1, inplace=True)

print(len(data))


# #### Merge in follower, followee blog description info

# In[10]:


# Load blog descriptions
descs = pd.read_pickle(os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl')) # Old data--no identity labels
print(len(descs))
print(descs.columns)


# In[11]:


# Merge in followee info
merged = pd.merge(data, descs, on='tumblog_id', how='left', suffixes=('_post', '_desc'))
print(len(merged))
print(merged.columns)


# In[12]:


# Merge in follower info
merged = pd.merge(merged, descs, left_on='follower', right_on='tumblog_id', how='left', suffixes=('_followee', '_follower'))
print(len(merged))
print(merged.columns)


# In[13]:


# Save out merged doc
# merged.to_pickle(os.path.join(data_dirpath, 'posts_descs_54k.pkl'))
merged.to_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))


# ## Get outcome measure: whether or not a post was reblogged

# In[3]:


# Load post data

post_fname = 'textposts_recent100.pkl' # ~75 GB in memory
post_data = pd.read_pickle(os.path.join(data_dirpath, post_fname))
print(len(post_data))
print(post_data.columns)


# In[5]:


# Load data to have outcome added to (full)
followee_posts = pd.read_pickle(os.path.join(data_dirpath, 'followee_posts_recent100.pkl'))
print(len(followee_posts))
print(followee_posts.columns)


# In[4]:


# Load data to have outcome added to (sampled)
# data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_54k.pkl'))
data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))
print(len(data))
print(data.columns)


# In[5]:


print(post_data['reblogged_from_post_id'].dtype)
print(data['post_id'].dtype)


# In[6]:


post_data['reblogged_from_post_id'] = post_data['reblogged_from_post_id'].fillna(-1).astype(int)
print(post_data['reblogged_from_post_id'].dtype)

reblogged_posts = post_data['reblogged_from_post_id'].unique()
print(len(reblogged_posts))

data['post_id'] = data['post_id'].astype(int)
print(data['post_id'].dtype)
posts = data['post_id'].unique()
print(len(posts))


# In[7]:


# How many posts from followees were reblogged by anybody in our data?

intersect = set(posts).intersection(set(reblogged_posts))
len(intersect)


# In[8]:


# Just consider opportunities for posts that were reblogged
reblog_data = data[data['post_id'].isin(intersect)]
len(reblog_data)


# In[9]:


# Any reblogs by followers? 

# Construct set of tuples (post_id, tumblog_id_follower) to search through
reblog_tuples = set(zip(post_data['reblogged_from_post_id'], post_data['tumblog_id']))
len(reblog_tuples)


# In[10]:


# Any reblogs by followers? -- Just look at intersection as candidates for reblogs
from IPython.core.debugger import set_trace
from tqdm import tqdm_notebook as tqdm

# def is_reblog(x):
#     rows = post_data[(post_data['reblogged_from_post_id'] == x['post_id']) & \
#                                   (post_data['tumblog_id'] == x['tumblog_id_follower'])]
# data['reblogged'] = data.apply(is_reblog, axis=1)

def is_reblog(post_id, tumblog_id_follower):
    
    return (post_id, tumblog_id_follower) in reblog_tuples

# data['reblogged'] = [is_reblog(post_id, tumblog_id_follower) for post_id, tumblog_id_follower \
#                      in tqdm(zip(data['post_id'], data['tumblog_id_follower']), total=len(data))]
reblog_data['reblogged'] = [is_reblog(post_id, tumblog_id_follower) for post_id, tumblog_id_follower                      in tqdm(zip(reblog_data['post_id'], reblog_data['tumblog_id_follower']), total=len(reblog_data))]

sum(reblog_data['reblogged'])


# In[12]:


merged = data.join(reblog_data['reblogged'])
merged['reblogged']

merged['reblogged'] = merged['reblogged'].fillna(False)
sum(merged['reblogged'])


# In[13]:


merged.to_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))


# ## Downsample to just followers who have reblogged at least once
# More restrictive: users who have reblogged and we have text data for them

# In[ ]:


# Load post data

post_fname = 'textposts_recent100.pkl' # ~75 GB in memory
post_data = pd.read_pickle(os.path.join(data_dirpath, post_fname))
print(len(post_data))
print(post_data.columns)


# In[31]:


# Find list of users who have reblogged at least once
reblogs = post_data[post_data['reblogged_from_post_id'] >= 0]
print(len(reblogs))


# In[33]:


reblog_followers = set(reblogs['tumblog_id'].unique())
len(reblog_followers)


# In[46]:


reblog_followers = reblog_users


# In[30]:


followee_posts.columns


# In[35]:


follow_data = pd.read_pickle(os.path.join(data_dirpath, 'follow_data_recent100.pkl'))
print(len(follow_data))
print(follow_data.columns)


# In[36]:


# Restrict followee posts to followees that have at least one follower who has reblogged at least one post

reblog_follow_data = follow_data[follow_data['tumblog_id'].isin(reblog_users)]
reblog_followees = set(reblog_follow_data['followed_tumblog_id'].unique())
print(len(reblog_followees))


# In[37]:


len(follow_data['followed_tumblog_id'].unique())


# In[38]:


# Save out reblog follow data
reblog_follow_data.to_pickle(os.path.join(data_dirpath, 'reblog_follow_data_recent100.pkl')) # followers are rebloggers


# In[5]:


# Load reblog follow data
reblog_follow_data = pd.read_pickle(os.path.join(data_dirpath, 'reblog_follow_data_recent100.pkl'))
reblog_followees = set(reblog_follow_data['followed_tumblog_id'].unique())
print(len(reblog_followees))


# In[6]:


# Filter followee posts to just those that have followers who have reblogged
reblog_followee_posts = followee_posts[followee_posts['tumblog_id'].isin(reblog_followees)]
print(len(reblog_followee_posts))
print(len(followee_posts))


# In[8]:


# Save reblog followee posts
reblog_followee_posts.to_pickle(os.path.join(data_dirpath, 'reblog_followee_posts_recent100.pkl'))


# # Add blog description annotations to 3M sample (one-time)

# In[2]:


# Load blog descriptions
descs = pd.read_pickle(os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl'))
print(len(descs))
print(descs.columns)


# In[6]:


# Load data to have outcome added to (sampled)
# data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_54k.pkl'))
data = pd.read_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))
print(len(data))
print(data.columns)


# In[4]:


# Remove columns previously merged
merged_cols = [col for col in data.columns if not (col.endswith('_desc') or                col.endswith('_followee') or col.endswith('_follower'))]
data = data[merged_cols]
data.columns


# In[12]:


pprint(sorted(data.columns.tolist()))


# In[13]:


data.rename(columns={
    'activity_time_epoch': 'activity_time_epoch_follower',
    'activity_time_epoch_desc': 'activity_time_epoch_followee',
    'blog_classifier': 'blog_classifier_follower',
    'blog_classifier_desc': 'blog_classifier_followee',
    'created_time_epoch': 'created_time_epoch_follower',
    'created_time_epoch_desc': 'created_time_epoch_followee',
    'updated_time_epoch': 'updated_time_epoch_follower',
    'updated_time_epoch_desc': 'updated_time_epoch_followee',
}, inplace=True)

pprint(sorted(data.columns.tolist()))


# In[14]:


# Merge in followee info
merged = pd.merge(data, descs, left_on='tumblog_id_followee', right_on='tumblog_id', how='left', 
#                   suffixes=('_post', '_followee')
                 )
print(len(merged))
pprint(sorted(merged.columns.tolist()))


# In[15]:


merged.drop('activity_time_epoch', axis=1, inplace=True)
merged.drop('blog_classifier', axis=1, inplace=True)
merged.drop('created_time_epoch', axis=1, inplace=True)
merged.drop('generated_date', axis=1, inplace=True)
merged.drop('is_group_blog', axis=1, inplace=True)
merged.drop('is_primary', axis=1, inplace=True)
merged.drop('is_private', axis=1, inplace=True)
merged.drop('language', axis=1, inplace=True)
merged.drop('parsed_blog_description', axis=1, inplace=True)
merged.drop('tumblr_blog_description', axis=1, inplace=True)
merged.drop('tumblr_blog_name', axis=1, inplace=True)
merged.drop('tumblr_blog_theme', axis=1, inplace=True)
merged.drop('tumblr_blog_title', axis=1, inplace=True)
merged.drop('tumblr_blog_url', axis=1, inplace=True)
merged.drop('updated_time_epoch', axis=1, inplace=True)
pprint(sorted(merged.columns.tolist()))


# In[16]:


desc_cats = [
    'age',
    'ethnicity/nationality',
    'fandoms',
    'gender',
    'gender/sexuality',
    'interests',
    'location',
    'personality_type',
    'pronouns',
    'roleplay',
    'roleplay/fandoms',
    'sexual orientation',
    'weight',
]


# In[19]:


for cat in desc_cats:
    merged.rename(columns={
        cat: f'{cat.replace(" ","_")}_followee',
        f'{cat}_terms': f'{cat.replace(" ","_")}_terms_followee',
    }, inplace=True)
        
    if f'{cat}_hegemonic_present' in merged.columns:
        merged.rename(columns={
            f'{cat}_hegemonic_present': f'{cat.replace(" ","_")}_hegemonic_present_followee',
            f'{cat}_opposite_present': f'{cat.replace(" ","_")}_opposite_present_followee',
        }, inplace=True)

pprint(sorted(merged.columns.tolist()))


# In[20]:


# Merge in follower info
merged = pd.merge(merged, descs, left_on='follower', right_on='tumblog_id', how='left',
#                   suffixes=('_followee', '_follower')
                 )
print(len(merged))
pprint(sorted(merged.columns))


# In[21]:


merged.drop('activity_time_epoch', axis=1, inplace=True)
merged.drop('blog_classifier', axis=1, inplace=True)
merged.drop('created_time_epoch', axis=1, inplace=True)
merged.drop('generated_date', axis=1, inplace=True)
merged.drop('is_group_blog', axis=1, inplace=True)
merged.drop('is_primary', axis=1, inplace=True)
merged.drop('is_private', axis=1, inplace=True)
merged.drop('language', axis=1, inplace=True)
merged.drop('parsed_blog_description', axis=1, inplace=True)
merged.drop('timezone_x', axis=1, inplace=True)
merged.drop('timezone_y', axis=1, inplace=True)
merged.drop('tumblog_id_x', axis=1, inplace=True)
merged.drop('tumblog_id_y', axis=1, inplace=True)
merged.drop('tumblr_blog_description', axis=1, inplace=True)
merged.drop('tumblr_blog_name', axis=1, inplace=True)
merged.drop('tumblr_blog_theme', axis=1, inplace=True)
merged.drop('tumblr_blog_title', axis=1, inplace=True)
merged.drop('tumblr_blog_url', axis=1, inplace=True)
merged.drop('updated_time_epoch', axis=1, inplace=True)
pprint(sorted(merged.columns.tolist()))


# In[22]:


for cat in desc_cats:
    merged.rename(columns={
        cat: f'{cat.replace(" ","_")}_follower',
        f'{cat}_terms': f'{cat.replace(" ","_")}_terms_follower',
    }, inplace=True)
        
    if f'{cat}_hegemonic_present' in merged.columns:
        merged.rename(columns={
            f'{cat}_hegemonic_present': f'{cat.replace(" ","_")}_hegemonic_present_follower',
            f'{cat}_opposite_present': f'{cat.replace(" ","_")}_opposite_present_follower',
        }, inplace=True)

pprint(sorted(merged.columns.tolist()))


# In[23]:


merged.to_pickle(os.path.join(data_dirpath, 'posts_descs_3m_rebloggers.pkl'))

