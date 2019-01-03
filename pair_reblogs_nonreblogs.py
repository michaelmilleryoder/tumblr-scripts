import pandas as pd
import numpy as np
import os
from tqdm import tqdm as tqdm
import pdb
from collections import defaultdict

# # Create a match file between reblogs and nonreblogs, sample200
def main():

    # I/O
    data_dirpath = '/usr2/mamille2/tumblr/data/sample1k' # erebor
    reblogs_fpath = os.path.join(data_dirpath, 'reblogs_descs_annotated', 'reblogs_descs.tsv')
    nonreblogs_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs_match')

    out_nonreblogs_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs_paired')
    if not os.path.exists(out_nonreblogs_dirpath):
        os.mkdir(out_nonreblogs_dirpath)
    pairings_fpath = os.path.join(data_dirpath, 'pairings_reblogs_nonreblogs.csv')

    # Load reblogs
    print("Loading reblogs...")
    reblogs = pd.read_csv(reblogs_fpath, sep='\t')

    # Load nonreblogs
    nonreblogs_fnames = sorted(os.listdir(nonreblogs_dirpath))
    #nonreblogs_fpaths = [os.path.join(nonreblogs_dirpath, fname) for fname in nonreblogs_fnames]

    #print("Loading nonreblogs...")
    #nonreblogs_dfs = [pd.read_csv(fpath, sep='\t') for fpath in nonreblogs_fpaths]
    #nonreblogs = pd.DataFrame(np.vstack([df.values for df in nonreblogs_dfs]), columns=nonreblogs_dfs[0].columns)
    #print(f'\tNumber of nonreblogs: {len(nonreblogs)}')
    #print(nonreblogs.columns)

    print("Pairing...")

    # Shuffle, then choose up to top 5 nonreblogs, restricting to get different followees
    #gps = nonreblogs.groupby(['paired_reblog_post_id', 'tumblog_id_follower'])

    #pairings = [] # (reblog_row_idx, nonreblog_row_idx)

    ## For every reblog, select the paired nonreblog group and sample from it
    #for reblog_idx, post_id, tid_follower, tid_followee in tqdm(zip(reblogs.index, reblogs['post_id'], reblogs['tumblog_id_follower'], reblogs['tumblog_id_followee'])):
    #    nonreblog_match = gps.groups.get((post_id, tid_follower), None) # row indices in that group
    #    if nonreblog_match is None: continue
    #    
    #    shuffled = nonreblog_match.tolist()
    #    np.random.shuffle(shuffled)
    #    
    #    keep = [] # row indices of nonreblogs
    #    nonreblog_followees = set()
    #    
    #    for row in shuffled:
    #        if len(keep) == 5:
    #            break
    #        else:
    #            # Check if followee is different from reblog followee, if followee unique
    #            nonreblog_followee = nonreblogs.loc[row, 'tumblog_id_followee']
    #            if tid_followee != nonreblog_followee: 
    #                if not nonreblog_followee in nonreblog_followees:
    #                    keep.append(row)
    #                    nonreblog_followees.add(nonreblog_followee)
    #                    
    #    pairings.extend([(reblog_idx, row_idx) for row_idx in keep])

    paired_dict = defaultdict(list) # (reblog_post_id, tumblog_id_follower): [(nonreblog_row_idx, tumblog_id_followee)]
    pairings = [] # (reblog_row_idx, nonreblog_fname, nonreblog_row_idx)
    nonreblog_offset = 0

    
    for fname in tqdm(nonreblogs_fnames):
        fpath = os.path.join(nonreblogs_dirpath, fname)
        nonreblogs = pd.read_csv(fpath, sep='\t')

        # Add offset to index
        nonreblogs.reset_index(drop=True)
        nonreblogs.index = nonreblogs.index + nonreblog_offset

        for reblog_row_idx, reblog_post_id, tumblog_id_follower, nonreblog_row_idx, tumblog_id_followee in zip(
                reblogs.index,
                nonreblogs['paired_reblog_post_id'],
                nonreblogs['tumblog_id_follower'],
                nonreblogs.index,
                nonreblogs['tumblog_id_followee'],
            ):

            # Check if have already reached 5 nonreblogs
            if len(paired_dict[(reblog_post_id, tumblog_id_follower)]) == 5:
                continue

            # Check that is a new followee
            existing_nonreblog_followees = [followee for _,followee in paired_dict[(reblog_post_id, tumblog_id_follower)]]
        
            if not tumblog_id_followee in existing_nonreblog_followees:
                paired_dict[(reblog_post_id, tumblog_id_follower)].append((nonreblog_row_idx, tumblog_id_followee))
                pairings.append((reblog_row_idx, fname, nonreblog_row_idx))

        # Save out nonreblogs with new row indices
        nonreblogs.to_csv(fpath, sep='\t')
        
        nonreblog_offset += len(nonreblogs)
        
    print(f'Number of pairings: {len(pairings)}')


    # Save out pairings
    print("Saving pairings file...")
    with open(pairings_fpath, 'w') as f:
        for pair in pairings:
            f.write(f'{pair[0]},{pair[1]},{pair[2]}\n') # reblog_row_idx, nonreblog_fname, nonreblog_row_idx


    # Save out nonreblogs with consistent row indices in new splits
    #print("Saving nonreblogs with new row indices...")
    #n_outlines = 10000
    #for i in tqdm(range(len(nonreblogs)//n_outlines + 1)):
    #    outpath = os.path.join(out_nonreblogs_dirpath, f'part{i:03d}.tsv')
    #    nonreblogs[np.arange(len(nonreblogs))//n_outlines==i].to_csv(outpath, sep='\t')

if __name__ == '__main__':
    main()
