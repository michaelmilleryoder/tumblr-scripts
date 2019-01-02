import os
import pandas as pd
from tqdm import tqdm
import pdb

#data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample200'
data_dirpath = '/usr2/mamille2/tumblr/data/sample1k'
in_dirpath = os.path.join(data_dirpath, 'reblogs_descs_cleaned')
#data_fpath = os.path.join(data_dirpath, 'sample200', 'reblogs_descs_cleaned.tsv')
#data_fpaths = [os.path.join(data_dirpath, 'nonreblogs_descs_cleaned', f) for f in sorted(os.listdir(os.path.join(data_dirpath, 'nonreblogs_descs_cleaned')))]
data_fnames = sorted(os.listdir(in_dirpath))
#out_fpath = os.path.join(data_dirpath, 'sample200', 'reblogs_descs.tsv')
out_dirpath = os.path.join(data_dirpath, 'reblogs_descs_nodups')
if not os.path.exists(out_dirpath):
    os.mkdir(out_dirpath)

selector = 'reblogs'

def main():


    for data_fname in tqdm(data_fnames):
        tqdm.write(data_fname)
        data_fpath = os.path.join(in_dirpath, data_fname)

        data = pd.read_csv(data_fpath, sep='\t')
        #data = pd.read_csv(data_fpath, sep='\t', header=None, names=cols)
        #for i in range(3):
        #    data.drop(f"EXTRA{i}", axis=1, inplace=True)

        # Check for header rows
#        if 'post_id' in data['post_id']:
#            hdr_rows = data[data['post_id']=='post_id'].index
#            data.drop(hdr_rows, inplace=True)

        # Rename, drop columns
        data.drop('followee::tumblog_id', axis=1, inplace=True)
        data.drop('reblogs::reblogs::tumblog_id_follower', axis=1, inplace=True)
        data.drop('reblogs::reblogs::blog_classifier_follower', axis=1, inplace=True)
        data.rename(columns={
            'followee::blog_description': 'blog_description_followee',
            'followee::blog_name': 'blog_name_followee',
            'followee::blog_title': 'blog_title_followee',
            'followee::blog_url': 'blog_url_followee',
            'followee::is_group_blog': 'is_group_blog_followee',
            'followee::is_private': 'is_private_followee',
            'followee::created_time_epoch': 'created_time_epoch_followee',
            'followee::updated_time_epoch': 'updated_time_epoch_followee',
            'followee::timezone': 'timezone_followee',
            'followee::language': 'language_followee',
            'followee::blog_classifier': 'blog_classifier_followee',
            'reblogs::follower::tumblog_id': 'tumblog_id_follower',
            'reblogs::follower::blog_description': 'blog_description_follower',
            'reblogs::follower::blog_name': 'blog_name_follower',
            'reblogs::follower::blog_title': 'blog_title_follower',
            'reblogs::follower::blog_url': 'blog_url_follower',
            'reblogs::follower::is_group_blog': 'is_group_blog_follower',
            'reblogs::follower::is_private': 'is_private_follower',
            'reblogs::follower::created_time_epoch': 'created_time_epoch_follower',
            'reblogs::follower::updated_time_epoch': 'updated_time_epoch_follower',
            'reblogs::follower::timezone': 'timezone_follower',
            'reblogs::follower::language': 'language_follower',
            'reblogs::follower::blog_classifier': 'blog_classifier_follower',
            'reblogs::reblogs::post_id': 'post_id',
            'reblogs::reblogs::activity_time_epoch_post': 'activity_time_epoch_post',
            'reblogs::reblogs::is_private': 'is_private_post',
            'reblogs::reblogs::post_title': 'post_title',
            'reblogs::reblogs::post_short_url': 'post_short_url',
            'reblogs::reblogs::post_slug': 'post_slug',
            'reblogs::reblogs::post_type': 'post_type',
            'reblogs::reblogs::post_caption': 'post_caption',
            'reblogs::reblogs::post_format': 'post_format',
            'reblogs::reblogs::post_note_count': 'post_note_count',
            'reblogs::reblogs::post_tags': 'post_tags',
            'reblogs::reblogs::post_content': 'post_content',
            'reblogs::reblogs::reblogged_from_post_id': 'reblogged_from_post_id',
            'reblogs::reblogs::reblogged_from_metadata': 'reblogged_from_metadata',
            'reblogs::reblogs::created_time_epoch_post': 'created_time_epoch_post',
            'reblogs::reblogs::updated_time_epoch_post': 'updated_time_epoch_post',
            'reblogs::reblogs::is_submission': 'is_submission',
            'reblogs::reblogs::mentions': 'mentions',
            'reblogs::reblogs::source_title': 'source_title',
            'reblogs::reblogs::source_url': 'source_url',
            'reblogs::reblogs::post_classifier': 'post_classifier',
            'reblogs::reblogs::activity_date_post': 'activity_date_post',
            'reblogs::reblogs::tumblog_id_followee': 'tumblog_id_followee',
            'reblogs::reblogs::activity_time_epoch_follow': 'activity_time_epoch_follow',
        }, inplace=True)

        tqdm.write("Removing duplicates...")
        tqdm.write(f"Original length: {len(data)}")
        
        if selector == 'nonreblogs':
            data.dropna(subset=['post_id', 'tumblog_id_followee', 'tumblog_id_follower', 'paired_reblog_post_id'], inplace=True)
            data.drop_duplicates(subset=['post_id', 'tumblog_id_followee', 'tumblog_id_follower', 'paired_reblog_post_id'], inplace=True)
        elif selector == 'reblogs':
            data.dropna(subset=['post_id', 'tumblog_id_followee', 'tumblog_id_follower'], inplace=True)
            data.drop_duplicates(subset=['post_id', 'tumblog_id_followee', 'tumblog_id_follower'], inplace=True)
        tqdm.write(f"De-duplicated length: {len(data)}")

        out_fpath = os.path.join(out_dirpath, data_fname)
        data.to_csv(out_fpath, sep='\t', index=False)
        tqdm.write(f"Wrote CSV to {out_fpath}")

if __name__ == '__main__':
    main()
