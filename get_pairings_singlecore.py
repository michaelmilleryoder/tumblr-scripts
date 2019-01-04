from collections import defaultdict
import csv
import os
import pandas as pd
import random
import re
from tqdm import tqdm
import pdb
import warnings
from multiprocessing import Pool

#
## Define output structures
#reblog_write = []
#nonreblog_write = []
#labels_write = []
#
## Read in pairings
#pairings = pd.read_csv(pairings_fpath, header=None, names=['reblog_row_idx', 'nonreblog_fname', 'nonreblog_row_idx']).sort_values('nonreblog_fname')
#nonreblog_fname_reblogs_map = pairings.set_index(['reblog_row_idx', 'nonreblog_row_idx']).groupby('nonreblog_fname').groups
#
#reblog_labels_to_extract = ['post_id_follower', 'tumblog_id_follower', 'tumblog_id_followee',
#                         'post_tags_followee', 'post_type_followee', 'post_note_count_followee',
#                         'processed_blog_description_follower', 'processed_blog_description_followee',
#                         'age_terms_follower', 'age_terms_followee',
#                         'ethnicity/nationality_terms_follower', 'ethnicity/nationality_terms_followee',
#                         'fandoms_terms_follower', 'fandoms_terms_followee',
#                         'gender_terms_follower', 'gender_terms_followee',
#                         'gender/sexuality_terms_follower', 'gender/sexuality_terms_followee',
#                         'interests_terms_follower', 'interests_terms_followee',
#                         'location_terms_follower', 'location_terms_followee',
#                         'personality type_terms_follower', 'personality type_terms_followee',
#                         'pronouns_terms_follower', 'pronouns_terms_followee',
#                         'relationship status_terms_follower', 'relationship status_terms_followee',
#                         'roleplay_terms_follower', 'roleplay_terms_followee',
#                         'roleplay/fandoms_terms_follower', 'roleplay/fandoms_terms_followee',
#                         'sexual orientation_terms_follower', 'sexual orientation_terms_followee',
#                         'weight_terms_follower', 'weight_terms_followee',
#                         'zodiac_terms_follower', 'zodiac_terms_followee']
#
#nonreblog_labels_to_extract = ['post_id', 'tumblog_id_follower', 'tumblog_id_followee',
#                         'post_tags', 'post_type', 'post_note_count',] + \
#                         reblog_labels_to_extract[6:]


def paren_process(s):
    lparen_count = 0
    rparen_count = 0
    for i in range(1, len(s)):
        if s[i] == '(' and s[i-1] != ':' and s[i-1] != ';' and s[i-1] != '-' and s[i-1] != '(':
            lparen_count += 1
        if s[i] == ')' and s[i-1] != ':' and s[i-1] != ';' and s[i-1] != '-' and s[i-1] != ')':
            rparen_count += 1

    if rparen_count == lparen_count:
         return s[1:]

    return s[1:-1]

def parse_tags(tags):
    paren_in_tag_re = r'[,{]\(.*?\)[,}]'
    matches = re.findall(paren_in_tag_re, str(tags))
    matches = [paren_process(x[1:-1]) for x in matches]
    return matches

def process_df(df):
    """ Remove nan strings, fix tag parentheses issue """

    # Remove nan strings
    nan_str_cols = {
                 'post_tags': '{}',
                'processed_blog_description_follower': '',
                'processed_blog_description_followee': ''
                }
    for col, replacement in nan_str_cols.items():
        df[col] = df[col].str.replace('nan', replacement)
        #df.loc[df[col]=='nan', col] = replacement

    # Remove nans from float columns
    nan_float_cols = {
                 'post_note_count': 0.0,
    }
    for col in nan_float_cols:
        df[col] = df[col].fillna(0.0)

    # Fix tag final parentheses issue
    df['post_tags'] = df['post_tags'].map(parse_tags)

    return df

def process_nonreblog_file(nonreblog_fname):

    reblog_write_lines = []
    nonreblog_write_lines = []
    labels_write_lines = []

    nonreblogs_fpath = os.path.join(nonreblogs_dirpath, nonreblog_fname)
    nonreblogs = pd.read_csv(nonreblogs_fpath, sep='\t', low_memory=False)

    if not nonreblog_fname in nonreblog_fname_reblogs_map:
        return

    for reblog_id, nonreblog_id in nonreblog_fname_reblogs_map[nonreblog_fname]:
        reblog_row = reblog_dataframe.iloc[reblog_id]
        reblog_row_values = [reblog_row[label] for label in reblog_labels_to_extract]
        #nonreblog_row_values = nonreblogs.iloc[nonreblog_id], nonreblog_labels_to_extract]
        nonreblog_row = nonreblogs.iloc[nonreblog_id]
        nonreblog_row_values = [nonreblog_row[label] for label in nonreblog_labels_to_extract]
        label = random.randint(0, 1)

        reblog_write_lines.append(reblog_row_values)
        nonreblog_write_lines.append(nonreblog_row_values)
        labels_write_lines.append([label])

    #reblog_write += reblog_write_lines
    #nonreblog_write += nonreblog_write_lines
    #labels_write += labels_write_lines

    pdb.set_trace()
    reblog_write.extend(reblog_write_lines)
    nonreblog_write.extend(nonreblog_write_lines)
    labels_write.extend(labels_write_lines)

def main():

    # I/O
    tumblr_dir = '/usr2/mamille2/tumblr/data/sample1k/'
    reblog_fpath = os.path.join(tumblr_dir, 
        'reblogs_descs_annotated',
        'reblogs_descs.tsv')
    nonreblogs_dirpath = os.path.join(tumblr_dir,
        'nonreblogs_descs_match')
    pairings_fpath = os.path.join(tumblr_dir,
        'pairings_reblogs_nonreblogs.csv')
    out_dirpath = os.path.join(tumblr_dir, 'feature_tables')
    if not os.path.exists(out_dirpath):
        os.mkdir(out_dirpath)
    reblogs_outpath = os.path.join(out_dirpath, 'reblog_features.csv')
    nonreblogs_outpath = os.path.join(out_dirpath, 'nonreblog_features.csv')
    labels_outpath = os.path.join(out_dirpath, 'ranking_labels.csv')

    # Load reblogs
    print("Loading reblogs...")
    reblog_dataframe = pd.read_csv(reblog_fpath, sep='\t', low_memory=False)

    # Read in non-reblogs - might need to change if you have memory issues
    nonreblog_dir = os.path.join(tumblr_dir, 'nonreblogs_descs_match')
    nonreblog_filenames = sorted(os.listdir(nonreblog_dir))
    #nonreblog_dataframes = [pandas.read_csv(os.path.join(nonreblog_dir, filename),
        #sep='\t') for filename in nonreblog_filenames]
    #nonreblog_combined_dataframe = pandas.concat(nonreblog_dataframes)

    # Read in pairings
    pairings = pd.read_csv(pairings_fpath, header=None, names=['reblog_row_idx', 'nonreblog_fname', 'nonreblog_row_idx']).sort_values('nonreblog_fname')
    #pairings_reader = csv.reader(open(os.path.join(tumblr_dir, 'pairings_reblogs_nonreblogs.csv')))
    #for reblog_id, nonreblog_fname, nonreblog_id in pairings_reader:
    #    reblog_nonreblogs_map[int(reblog_id)].append((nonreblog_fname, int(nonreblog_id)))
    #reblog_nonreblogs_map = pairings.set_index(['nonreblog_fname', 'nonreblog_row_idx']).groupby('reblog_row_idx').groups
    nonreblog_fname_reblogs_map = pairings.set_index(['reblog_row_idx', 'nonreblog_row_idx']).groupby('nonreblog_fname').groups

    reblog_labels_to_extract = ['post_id_follower', 'tumblog_id_follower', 'tumblog_id_followee',
                             'post_tags_followee', 'post_type_followee', 'post_note_count_followee',
                             'processed_blog_description_follower', 'processed_blog_description_followee',
                             'age_terms_follower', 'age_terms_followee',
                             'ethnicity/nationality_terms_follower', 'ethnicity/nationality_terms_followee',
                             'fandoms_terms_follower', 'fandoms_terms_followee',
                             'gender_terms_follower', 'gender_terms_followee',
                             'gender/sexuality_terms_follower', 'gender/sexuality_terms_followee',
                             'interests_terms_follower', 'interests_terms_followee',
                             'location_terms_follower', 'location_terms_followee',
                             'personality type_terms_follower', 'personality type_terms_followee',
                             'pronouns_terms_follower', 'pronouns_terms_followee',
                             'relationship status_terms_follower', 'relationship status_terms_followee',
                             'roleplay_terms_follower', 'roleplay_terms_followee',
                             'roleplay/fandoms_terms_follower', 'roleplay/fandoms_terms_followee',
                             'sexual orientation_terms_follower', 'sexual orientation_terms_followee',
                             'weight_terms_follower', 'weight_terms_followee',
                             'zodiac_terms_follower', 'zodiac_terms_followee']

    nonreblog_labels_to_extract = ['post_id', 'tumblog_id_follower', 'tumblog_id_followee',
                             'post_tags', 'post_type', 'post_note_count',] + \
                             reblog_labels_to_extract[6:]

    # Extract relevant information from matched posts
    reblog_write = []
    nonreblog_write = []
    labels_write = [['ranking_label']]

    #with Pool(15) as pool:
    #    list(tqdm(pool.imap(process_nonreblog_file, nonreblog_filenames), total=len(nonreblog_filenames)))

    for nonreblog_fname in tqdm(nonreblog_filenames, ncols=50):
    #for nonreblog_fname in tqdm(list(nonreblog_fname_reblogs_map.keys())[:1]):
        nonreblogs_fpath = os.path.join(nonreblogs_dirpath, nonreblog_fname)
        nonreblogs = pd.read_csv(nonreblogs_fpath, sep='\t', low_memory=False)

        for reblog_id, nonreblog_id in nonreblog_fname_reblogs_map[nonreblog_fname]:
            reblog_row = reblog_dataframe.iloc[reblog_id]
            reblog_row_values = [reblog_row[label] for label in reblog_labels_to_extract]
            #nonreblog_row_values = nonreblogs.iloc[nonreblog_id], nonreblog_labels_to_extract]
            nonreblog_row = nonreblogs.iloc[nonreblog_id]
            nonreblog_row_values = [nonreblog_row[label] for label in nonreblog_labels_to_extract]
            label = random.randint(0, 1)

            reblog_write.append(reblog_row_values)
            nonreblog_write.append(nonreblog_row_values)
            labels_write.append([label])

    reblog_file_writer = csv.writer(open(reblogs_outpath, 'w'))
    reblog_file_writer.writerows(reblog_write)

    nonreblog_file_writer = csv.writer(open(nonreblogs_outpath, 'w'))
    nonreblog_file_writer.writerows(nonreblog_write)

    labels_writer = csv.writer(open(labels_outpath, 'w'))
    labels_writer.writerows(labels_write)

    # Postprocess
    #out_reblog_df = pd.DataFrame(reblog_write, columns=nonreblog_labels_to_extract)
    #out_nonreblog_df = pd.DataFrame(nonreblog_write, columns=nonreblog_labels_to_extract)

    #out_reblog_df = process_df(out_reblog_df)
    #out_nonreblog_df = process_df(out_nonreblog_df)

    #out_reblog_df.to_csv(reblogs_outpath, index=False)
    #out_nonreblog_df.to_csv(nonreblogs_outpath, index=False)

if __name__ == '__main__': main()
