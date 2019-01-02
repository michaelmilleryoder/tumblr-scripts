import os
import pandas as pd
from tqdm import tqdm

""" Makes sure all nonreblogs have a matching reblog """

def main():
    
    # I/0
    data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample1k'
    reblogs_fpath = os.path.join(data_dirpath, 'reblogs_descs_annotated.tsv')
    nonreblogs_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs_annotated')
    nonreblogs_fnames = sorted(os.listdir(nonreblogs_dirpath))
    out_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs_match') # for nonreblogs
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)

    # Load reblogs
    reblogs = pd.read_csv(reblogs_fpath, sep='\t')
    # Remove any index columns
    reblogs = reblogs.loc[:, ~reblogs.columns.str.contains('^Unnamed')]

    # Load nonreblogs, process
    for fname in tqdm(nonreblogs_fnames, ncols=50):
        fpath = os.path.join(nonreblogs_dirpath, fname)
        out_fpath = os.path.join(out_dirpath, fname)
        nonreblogs = pd.read_csv(fpath, sep='\t')
        # Remove any index columns
        nonreblogs = nonreblogs.loc[:, ~nonreblogs.columns.str.contains('^Unnamed')]

        tqdm.write(f'Original length: {len(nonreblogs)}')
        nonreblog_matches = nonreblogs[nonreblogs['paired_reblog_post_id'].isin(reblogs['post_id'])]
        tqdm.write(f'Matches: {len(nonreblog_matches)}')

        # Save out
        nonreblog_matches.to_csv(out_fpath, sep='\t', index=False)

if __name__ == '__main__':
    main()
