import os
import pandas as pd
from tqdm import tqdm
import pdb

data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample200'
#data_fpath = os.path.join(data_dirpath, 'sample200', 'reblogs_descs_cleaned.tsv')
#data_fpaths = [os.path.join(data_dirpath, 'nonreblogs_descs_cleaned', f) for f in sorted(os.listdir(os.path.join(data_dirpath, 'nonreblogs_descs_cleaned')))]
data_fnames = sorted(os.listdir(os.path.join(data_dirpath, 'nonreblogs_descs_cleaned')))
#out_fpath = os.path.join(data_dirpath, 'sample200', 'reblogs_descs.tsv')
out_dirpath = os.path.join(data_dirpath, 'nonreblogs_descs')

def main():

    for data_fname in tqdm(data_fnames):
        tqdm.write(data_fname)
        data_fpath = os.path.join(data_dirpath, 'nonreblogs_descs_cleaned', data_fname)
        cols = [
            'tumblog_id_followee',
            'blog_description_followee',
            'blog_name_followee',
            'blog_title_followee',
            'blog_url_followee',
            'is_group_blog_followee',
            'is_private_followee',
            'created_time_epoch_followee',
            'updated_time_epoch_followee',
            'timezone_followee',
            'language_followee',
            'blog_classifier_followee',
            'tumblog_id_follower',
            'blog_description_follower',
            'blog_name_follower',
            'blog_title_follower',
            'blog_url_follower',
            'is_group_blog_follower',
            'is_private_follower',
            'created_time_epoch_follower',
            'updated_time_epoch_follower',
            'timezone_follower',
            'language_follower',
            'blog_classifier_follower',
            'post_id',
            'EXTRA0',
            'EXTRA1',
            'activity_time_epoch_post',
            'is_private',
            'post_title',
            'post_short_url',
            'post_slug',
            'post_type',
            'post_caption',
            'post_format',
            'post_note_count',
            'post_tags',
            'post_content',
            'reblogged_from_post_id',
            'reblogged_from_metadata',
            'created_time_epoch_post',
            'updated_time_epoch_post',
            'is_submission',
            'mentions',
            'source_title',
            'source_url',
            'post_classifier',
            'EXTRA2',
            'activity_date_post',
            'activity_time_epoch_follow',
            'paired_reblog_post_id',
        ]

        #if data_fname == 'nonreblogs_descs01_cleaned.tsv':
        #    pdb.set_trace()
        data = pd.read_csv(data_fpath, sep='\t', header=None, names=cols)
        for i in range(3):
            data.drop(f"EXTRA{i}", axis=1, inplace=True)

        # Check for header rows
        if 'post_id' in data['post_id']:
            hdr_rows = data[data['post_id']=='post_id'].index
            data.drop(hdr_rows, inplace=True)


        # Rename, drop columns (for reblogs)
        #data.drop('followee::tumblog_id', axis=1, inplace=True)
        #data.drop('reblogs::reblogs::tumblog_id_follower', axis=1, inplace=True)
        #data.drop('reblogs::reblogs::blog_classifier_follower', axis=1, inplace=True)
        #data.rename(columns={
        #    'followee::blog_description': 'blog_description_followee',
        #    'followee::blog_name': 'blog_name_followee',
        #    'followee::blog_title': 'blog_title_followee',
        #    'followee::blog_url': 'blog_url_followee',
        #    'followee::is_group_blog': 'is_group_blog_followee',
        #    'followee::is_private': 'is_private_followee',
        #    'followee::created_time_epoch': 'created_time_epoch_followee',
        #    'followee::updated_time_epoch': 'updated_time_epoch_followee',
        #    'followee::timezone': 'timezone_followee',
        #    'followee::language': 'language_followee',
        #    'followee::blog_classifier': 'blog_classifier_followee',
        #    'reblogs::follower::tumblog_id': 'tumblog_id_follower',
        #    'reblogs::follower::blog_description': 'blog_description_follower',
        #    'reblogs::follower::blog_name': 'blog_name_follower',
        #    'reblogs::follower::blog_title': 'blog_title_follower',
        #    'reblogs::follower::blog_url': 'blog_url_follower',
        #    'reblogs::follower::is_group_blog': 'is_group_blog_follower',
        #    'reblogs::follower::is_private': 'is_private_follower',
        #    'reblogs::follower::created_time_epoch': 'created_time_epoch_follower',
        #    'reblogs::follower::updated_time_epoch': 'updated_time_epoch_follower',
        #    'reblogs::follower::timezone': 'timezone_follower',
        #    'reblogs::follower::language': 'language_follower',
        #    'reblogs::follower::blog_classifier': 'blog_classifier_follower',
        #    'reblogs::reblogs::post_id': 'post_id',
        #    'reblogs::reblogs::activity_time_epoch_post': 'activity_time_epoch_post',
        #    'reblogs::reblogs::is_private': 'is_private_post',
        #    'reblogs::reblogs::post_title': 'post_title',
        #    'reblogs::reblogs::post_short_url': 'post_short_url',
        #    'reblogs::reblogs::post_slug': 'post_slug',
        #    'reblogs::reblogs::post_type': 'post_type',
        #    'reblogs::reblogs::post_caption': 'post_caption',
        #    'reblogs::reblogs::post_format': 'post_format',
        #    'reblogs::reblogs::post_note_count': 'post_note_count',
        #    'reblogs::reblogs::post_tags': 'post_tags',
        #    'reblogs::reblogs::post_content': 'post_content',
        #    'reblogs::reblogs::reblogged_from_post_id': 'reblogged_from_post_id',
        #    'reblogs::reblogs::reblogged_from_metadata': 'reblogged_from_metadata',
        #    'reblogs::reblogs::created_time_epoch_post': 'created_time_epoch_post',
        #    'reblogs::reblogs::updated_time_epoch_post': 'updated_time_epoch_post',
        #    'reblogs::reblogs::is_submission': 'is_submission',
        #    'reblogs::reblogs::mentions': 'mentions',
        #    'reblogs::reblogs::source_title': 'source_title',
        #    'reblogs::reblogs::source_url': 'source_url',
        #    'reblogs::reblogs::post_classifier': 'post_classifier',
        #    'reblogs::reblogs::activity_date_post': 'activity_date_post',
        #    'reblogs::reblogs::tumblog_id_followee': 'tumblog_id_followee',
        #    'reblogs::reblogs::activity_time_epoch_follow': 'activity_time_epoch_follow',
        #}, inplace=True)

        tqdm.write("Removing duplicates")
        tqdm.write(f"Original length: {len(data)}")
        
        data.dropna(subset=['post_id', 'tumblog_id_followee', 'tumblog_id_follower', 'paired_reblog_post_id'], inplace=True)
        data.drop_duplicates(subset=['post_id', 'tumblog_id_followee', 'tumblog_id_follower', 'paired_reblog_post_id'], inplace=True)
        tqdm.write(f"De-duplicated length: {len(data)}")

        out_fpath = os.path.join(out_dirpath, data_fname[:data_fname.find('_cleaned')] + '.tsv')
        data.to_csv(out_fpath, sep='\t', index=False)
        tqdm.write(f"Wrote CSV to {out_fpath}")

if __name__ == '__main__':
    main()
