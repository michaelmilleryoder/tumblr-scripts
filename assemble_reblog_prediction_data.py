import pandas as pd
import os
from tqdm import tqdm as tqdm
import numpy as np


# # See if can use data that already have for reblog prediction
def get_followees():

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
    return follower_ids # Have at least one piece of follow information for these users


def load_follower_ids(fpath):

    with open(fpath, 'r') as f:
        follower_ids = set([int(l) for l in f.read().splitlines()])

    return follower_ids


def main():

    data_dirpath = '/usr2/mamille2/tumblr/data'

    # ## Load followers
    print("Loading followers...")
    follower_data = pd.read_pickle(os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl'))
    print(len(follower_data))
    print(follower_data.columns)

    # Load followers who have follow info
    follower_ids = load_follower_ids(os.path.join(data_dirpath,'halfday_followers_with_descriptions.txt'))

    # ## Load followees
    # See what follow structure can find for set of followers

    # Get followee data
    print("Getting followee data...")
    follow_fpath = '/usr2/kmaki/tumblr/follow_network.tsv' # 99G
    chunksize = int(1e5)
    total_iters = int(np.ceil(3220088178 / chunksize))
    hdr = ['tumblog_id', 'followed_tumblog_id', 'activity_time_epoch']
    chunked = pd.read_csv(follow_fpath, sep='\t', header=None, names=hdr, chunksize=chunksize)

    followee_data = [] # list of np arrays to be concatenated

    for chunk in tqdm(chunked, total=total_iters):
        followee_data.append(chunk[chunk['tumblog_id'].isin(follower_ids)].values)
        
    follower_follow_data = pd.DataFrame(np.vstack(followee_data), columns=hdr)

    # Save follow data from followers for which have descriptions
    out_fpath = os.path.join(data_dirpath, 'follow_data_recent100.pkl')

    follower_follow_data.to_pickle()
    print(f"Saved follower follow data to {out_fpath}")

    ## For how many followees do we have blog descriptions for?
    #followees = set(follower_follow_data['followed_tumblog_id'])
    #print(len(followees))

    #followees_with_descriptions = followees.intersection()


    ## ## Load posts

    ## In[ ]:


    #post_fname = 'textposts_recent100.pkl'
    #post_data = pd.read_pickle(os.path.join(data_dirpath, post_fname))

if __name__ == '__main__':
    main()
