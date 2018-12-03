import pandas as pd
import os
from tqdm import tqdm as tqdm
import numpy as np
import pickle
import gc
import pdb

#def get_followees(follow_fpath):
#
#    chunksize = int(1e5)
#    total_iters = int(np.ceil(3220088178 / chunksize))
#    hdr = ['tumblog_id', 'followed_tumblog_id', 'activity_time_epoch']
#    chunked = pd.read_csv(follow_fpath, sep='\t', header=None, names=hdr, chunksize=chunksize)
#
#    stx_follower_ids = set()
#    # follower_ids = follower_data['tumblog_id'].tolist()
#    # followee_data = []
#
#    for chunk in tqdm(chunked, total=total_iters):
#        stx_follower_ids |= set(chunk['tumblog_id'].unique())
#    #     followee_data.append(chunk[chunk['tumblog_id'].isin(follower_ids)].values)
#        
#    print(len(stx_follower_ids))
#
#    follower_ids = stx_follower_ids.intersection(set(follower_data['tumblog_id']))
#
#    return follower_ids # Have at least one piece of follow information for these users


def get_followee_data(follow_fpath):
    # Unnecessary?

    chunksize = int(1e5)
    total_iters = int(np.ceil(3220088178 / chunksize))
    hdr = ['tumblog_id', 'followed_tumblog_id', 'activity_time_epoch']
    chunked = pd.read_csv(follow_fpath, sep='\t', header=None, names=hdr, chunksize=chunksize)

    followee_data = [] # list of np arrays to be concatenated

    for chunk in tqdm(chunked, total=total_iters, ncols=20):
        followee_data.append(chunk[chunk['tumblog_id'].isin(follower_ids)].values)
        
    follower_follow_data = pd.DataFrame(np.vstack(followee_data), columns=hdr)

    follower_data = pd.read_pickle(os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl'))

    return follower_follow_data


def load_follower_ids(fpath):

    with open(fpath, 'r') as f:
        follower_ids = set([int(l) for l in f.read().splitlines()])

    return follower_ids


def get_rebloggers(post_data, descs_data, follower_ids, follow_data):

    # Restrict followees to those who have blog descriptions
    #followee_ids = set(follow_data.loc[follow_data['followed_tumblog_id'].isin(descs_data['tumblog_id']), 'followed_tumblog_id'].unique())
    followee_ids = set(follow_data['followed_tumblog_id']).intersection(set(descs_data['tumblog_id']))

    # Get posts by followees
    followee_posts = post_data[post_data['tumblog_id'].isin(followee_ids)]
    print(len(followee_posts))

    # Find list of users who have reblogged from set of followees at least once
    reblogs = followee_posts[followee_posts['reblogged_from_post_id'] >= 0]
    reblog_followers = set(reblogs['tumblog_id'].unique())
    print(len(reblog_followers))

    return reblog_followers


def make_reblog_opportunities(reblog_follow_data, follower_ids, followee_posts, opportunities_outpath, data_dirpath, chunksize):

    # Make a dictionary of followees -> followers
    gped = reblog_follow_data.set_index('tumblog_id').groupby('followed_tumblog_id')

    #print("\tMaking follow dictionary...")
    #follow_dict = {key: list(gped.groups[key]) for key in tqdm(gped.groups)}
    # Save
    #with open(os.path.join(data_dirpath, 'temp_follow_dict.pkl'), 'wb') as f:
    #    pickle.dump(follow_dict, f)
    # Load
    #print("\tLoading follow dictionary...")
    #with open(os.path.join(data_dirpath, 'temp_follow_dict.pkl'), 'rb') as f:
    #    follow_dict = pickle.load(f)

    # Lookup table for follow times 
    #print("\tMaking time lookup dictionary...")
    #time_dict = {(follower, followee): time for follower, followee, time in tqdm(list(zip(
    #                reblog_follow_data['tumblog_id'], 
    #                reblog_follow_data['followed_tumblog_id'],
    #                reblog_follow_data['activity_time_epoch']
    #            )))}
    # Save
    #with open(os.path.join(data_dirpath, 'temp_time_dict.pkl'), 'wb') as f:
    #    pickle.dump(time_dict, f)
    # Load
    #print("\tLoading time lookup dictionary...")
    #with open(os.path.join(data_dirpath, 'temp_time_dict.pkl'), 'rb') as f:
    #    time_dict = pickle.load(f)

    # Merge posts with followers/followee, make opportunities
    #print("\tMerging follower information with posts...")
    #followee_posts['followers'] = [get_followers(post_ts, followee, follow_dict, time_dict) for followee, post_ts in tqdm(list(zip(followee_posts['tumblog_id'], followee_posts['activity_time_epoch'])))]
    # Save
 #   followee_posts['followers'].to_pickle(os.path.join(data_dirpath, 'temp_follow_info.pkl'))
    # Load
    print("\tLoading follower info for posts...")
    followee_posts['followers'] = pd.read_pickle(os.path.join(data_dirpath, 'temp_follow_info.pkl'))
 
    # Posts -> opportunities, each row a specific follower/followee/post triple

    # Split up
    n_splits = len(followee_posts)//chunksize + 1
    splits = np.array_split(followee_posts, n_splits)

    for i, split in tqdm(enumerate(splits), total=n_splits):
        s = split.apply(lambda x: pd.Series(x['followers']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'follower'
        data = split.join(s)
        data.reset_index(drop=True, inplace=True)
        data.drop('followers', axis=1, inplace=True)
        #print(f"\tLength of opportunities table: {len(data)}")

        # Save opportunities
        tqdm.write(f'\tSaving opportunities data to {opportunities_outpath.format(i)}')
        data.to_csv(opportunities_outpath.format(i), index=False)


def get_followers(post_ts, followee, follow_dict, time_dict):
    # Return followers who followed after a post was posted
    
    return [follower for follower in follow_dict[followee] if int(post_ts) >= int(time_dict[(follower, followee)])]


def main():

    ###### In I/O #######
    #data_dirpath = '/usr2/mamille2/tumblr/data' # erebor
    data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data' # misty
    follow_fpath = '/usr2/kmaki/tumblr/follow_network.tsv' # 99G, only erebor
    followers_fpath = os.path.join(data_dirpath,'halfday_followers_with_descriptions.txt')
    #posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.pkl') # ~75 GB in memory
    posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.csv') # ~75 GB in memory
    descs_fpath = os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl')

    ###### Out I/O ######
    follow_out_fpath = os.path.join(data_dirpath, 'follow_data_recent100.pkl')
    opportunities_outpath = os.path.join(data_dirpath, 'posts_descs', 'posts_descs_{:04d}.csv')

    # ## Load followers
    print("Loading followers...")
    follower_data = pd.read_pickle(descs_fpath)

    # Load followers who have follow info, descriptions
    follower_ids = load_follower_ids(followers_fpath)
    #followee_ids = get_followees()

    # Get followee data
    #print("Getting followee data...")
    #follower_follow_data = get_followee_data(follow_fpath)
    #follower_follow_data.to_pickle(follow_out_fpath)
    #print(f"Saved follower follow data to {follow_out_fpath}")

    print("Loading followee data...")
    follow_data = pd.read_pickle(follow_out_fpath)

    # Restrict followers to those who have ever reblogged
    print("Loading rebloggers...")
    with open(os.path.join(data_dirpath, 'rebloggers.pkl'), 'rb') as f:
        rebloggers = pickle.load(f)

    # Filter to posts posted from followees
    print("Filtering posts to just those from followees...")
    followees = set(follow_data.loc[follow_data['tumblog_id'].isin(rebloggers), 'followed_tumblog_id'])

    print("Loading posts...")
    #post_data = pd.read_pickle(posts_fpath) # 75 GB
    #print("Filtering followers to rebloggers...")
    #rebloggers = get_rebloggers(post_data, follower_data, follower_ids, follow_data)

    #read_chunksize = 1e6
    #total_lines = 28e6 # from wc -l
    #followee_posts_lines = []
    #ctr = 0
    #for chunk in pd.read_csv(posts_fpath, chunksize=read_chunksize, error_bad_lines=False):
    #    print(f"{ctr} / {int(total_lines/read_chunksize)}")
    #    hdr = chunk.columns
    #    if ctr < 27: # may be problem with num columns in CSV
    #        followee_posts_lines.append(chunk[chunk['tumblog_id'].isin(followees)])
    #    else:
    #        break
    #    ctr += 1

    #stack = np.vstack([d.values for d in followee_posts_lines])
    #followee_posts = pd.DataFrame(stack, columns=hdr)
    ## Save
    #followee_posts.to_pickle(os.path.join(data_dirpath, 'temp_followee_posts.pkl'))
    # Load
    followee_posts = pd.read_pickle(os.path.join(data_dirpath, 'temp_followee_posts.pkl'))

    # Restrict follow data to rebloggers
    reblog_follow_data = follow_data[follow_data['tumblog_id'].isin(follower_ids)]
    del follow_data
    gc.collect()

    # Find posts from followees that followers might have seen (posted after follow)
    print("Making table with reblog opportunities...")
    write_chunksize = 1e4
    make_reblog_opportunities(reblog_follow_data, follower_ids, followee_posts, opportunities_outpath, data_dirpath, write_chunksize)

    # Annotate reblog opportunities as reblogged or not

    # Split reblogged opportunities from not (should do earlier by figuring out first which followee posts have been rebloggedby followers)

    # Sample negative example, pair with reblogs


if __name__ == '__main__':
    main()
