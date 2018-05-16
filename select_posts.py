import pandas as pd
import os
import sys
import pdb

""" Select most recent 100 text posts from users with descriptions in provided file """

# I/O
data_dirpath = '/usr2/mamille2/tumblr/data'
descs_path = os.path.join(data_dirpath, 'blog_descriptions_recent100_100posts.pkl')
posts_path = os.path.join(data_dirpath, 'textposts_recent100.pkl')
outpath = os.path.join(data_dirpath, 'textposts_{}posts.pkl')

# Settings
#post_thresholds = [100, 50]
post_thresholds = [100]

# Load data
print("Loading data...", end=' ')
sys.stdout.flush()
descs = pd.read_pickle(descs_path)
posts = pd.read_pickle(posts_path) # 72 GB in RAM
print('done.')
sys.stdout.flush()

# Select posts
print("Selecting posts...")
sys.stdout.flush()
posts = posts[posts['tumblog_id'].isin(descs['tumblog_id'])]
gped = posts.groupby('tumblog_id').size()

for t in post_thresholds:
    print(f'{t} minimum posts')
    t_gped = gped[gped>=t] # Must have minimum #posts
    t_posts = posts[posts['tumblog_id'].isin(t_gped.index)]
    t_outpath = outpath.format(t)
    print('done.')
    print(f"#descs with minimum posts: {len(t_posts['tumblog_id'].unique())}")
    sys.stdout.flush()

    # Save posts
    print(f'Saving posts to {t_outpath}...', end=' ')
    sys.stdout.flush()
    t_posts.to_pickle(t_outpath)
    print('done.')
    print()
    sys.stdout.flush()
