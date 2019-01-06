import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import pdb

""" Makes sure all nonreblogs have a matching reblog (post id)"""

# I/0
#data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample1k'
data_dirpath = '/usr2/mamille2/tumblr/data/sample1k'
reblogs_fpath = os.path.join(data_dirpath, 'reblogs_descs_annotated', 'reblogs_descs.tsv')
nonreblogs_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs_annotated')
nonreblogs_fnames = sorted(os.listdir(nonreblogs_dirpath))
out_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs_match') # for nonreblogs
if not os.path.exists(out_dirpath):
    os.mkdir(out_dirpath)

# Load reblogs
print("Loading reblogs...")
reblogs = pd.read_csv(reblogs_fpath, sep='\t', low_memory=False)
# Remove any index columns
reblogs = reblogs.loc[:, ~reblogs.columns.str.contains('^Unnamed')]


def process_csv(fname):
    fpath = os.path.join(nonreblogs_dirpath, fname)
    out_fpath = os.path.join(out_dirpath, fname)
    #if os.path.exists(out_fpath): return

    nonreblogs = pd.read_csv(fpath, sep='\t', low_memory=False)

    # Remove any index columns
    nonreblogs = nonreblogs.loc[:, ~nonreblogs.columns.str.contains('^Unnamed')]

    #tqdm.write(f'Original length: {len(nonreblogs)}')
    nonreblog_matches = nonreblogs[nonreblogs['paired_reblog_post_id'].isin(reblogs['post_id_follower'])]
    #tqdm.write(f'Matches: {len(nonreblog_matches)}')

    # Save out
    nonreblog_matches.to_csv(out_fpath, sep='\t', index=False)


def main():
    
    # Load nonreblogs, process
    #for fname in tqdm(nonreblogs_fnames, ncols=50):
    #    process_csv(fname)
    with Pool(15) as pool:
        list(tqdm(pool.imap(process_csv, nonreblogs_fnames), total=len(nonreblogs_fnames)))

    # debugging
    #list(map(process_csv, tqdm(nonreblogs_fnames)))

if __name__ == '__main__':
    main()
