import pytumblr
import pandas as pd
import re
import random
from tqdm import tqdm
import pickle
import os
import urllib.request
import urllib.error
from datetime import datetime
import time

def save_output(out_infopath, out_imagepath, num_attempted, count_successful, num_rejected, blognames_with_default_images, blognames_with_images):

    with open(out_infopath.format(datetime.now().strftime("%Y-%m-%dT%H%M")), 'w') as f:
        f.write(f'Outpath: {out_imagepath}\n')
        f.write(f'Number of profile images attempted: {num_attempted}\n')
        f.write(f'Number of profile images scraped: {count_successful}\n')
        f.write(f'Number of profile images rejected: {num_rejected}\n')
        f.write(f'Number with default images: {len(blognames_with_default_images)}\n')
        f.write(f'Number with other images: {len(blognames_with_images)}\n\n')
        f.write("Blog names with default images:\n")
        for name in blognames_with_default_images:
            f.write(f'{name}\n')
        f.write("\n")
        f.write("Blog names with other images:\n")
        for name in blognames_with_images:
            f.write(f'{name}\n')

def main():

    # I/O
    blognames_path = '/data/websci2020_tumblr_identity/icwsm2020_sample1k/blog_names.txt'
    #blognames_path = '/data/icwsm2020_tumblr_identity/test_set_blog_names.txt'
    #out_image_dirpath = '/data/icwsm2020_tumblr_identity/profile_images/test_set'
    out_image_dirpath = '/data/websci2020_tumblr_identity/profile_images/icwsm2020_sample1k_nondefault'
    if not os.path.exists(out_image_dirpath):
        os.mkdir(out_image_dirpath)
    out_image_subdir = os.path.join(out_image_dirpath, '{}')
    out_imagepath = os.path.join(out_image_subdir, '{}.png')
    out_infopath = '/projects/websci2020_tumblr_identity/logs/scrape_info_{}.txt'

    # Settings
    num_lines = None # numeric limit if want to, put None if not
    offset = 0 # already done
    offset_successful = 0

    # OAuth
    with open('../oauth.txt') as f:
        lines = f.read().splitlines()
        
    client = pytumblr.TumblrRestClient(lines[0], lines[1], lines[2], lines[3])

    # Load selected blog names
    print("Loading blog names...")
    if blognames_path.endswith('.csv'):
        blog_info = pd.read_csv(blognames_path, escapechar='\\', engine='python', encoding='utf8', error_bad_lines=False)
        blognames = blog_info['tumblr_blog_name'].tolist()

    else:
        with open(blognames_path) as f:
            blognames = f.read().splitlines()

    selection = [b for b in blognames if isinstance(b, str)][offset:num_lines]
    print(f"Found {len(selection)} blog names\n")

    # Scrape images
    print("Scraping images...")
    max_num = len(selection)
    count_successful = 0
    blognames_with_default_images = []
    blognames_with_images = []
    num_blognames_with_images = 0
    num_rejected = 0

    pbar = tqdm(total=max_num)
    #pbar = tqdm(total=int(.71*num_lines))

    for i, blogname in enumerate(selection):
        if count_successful >= max_num:
            break

        try:
            response = client.avatar(blogname, size=512)
            
            # Check for rate limit exceedance
            if 'errors' in response and len(response['errors']) > 0:
                if 'title' in response['errors'][0]:
                    if 'Limit Exceeded' in response['errors'][0]['title']:
                        print(f"limit exceeded, waiting one hour")
                        print(f"\t{response['errors']}")
                        time.sleep(3600)
                
            if 'avatar_url' in response:
                count_successful += 1
                #pbar.update(1)
                avatar_url = response['avatar_url']
                
                if avatar_url.startswith('https://assets.tumblr.com/images/default_avatar/'):
                    blognames_with_default_images.append(blogname)
                else:    
                    blognames_with_images.append(blogname)
                    num_blognames_with_images += 1
                    r = urllib.request.urlopen(avatar_url).read()
                    #with open(out_imagepath.format(count_successful), 'wb') as f:
                    #with open(out_imagepath.format(offset_successful + num_blognames_with_images), 'wb') as f:
                    out_dirpath = out_image_subdir.format(blogname[:2]) # subdir first 2 chars of blogname
                    if not os.path.exists(out_dirpath):
                        os.mkdir(out_dirpath)
                    outpath = out_imagepath.format(blogname[:2], blogname)
                    with open(outpath, 'wb') as f: # actual saving of the image
                        f.write(r)

                tqdm.write(f"{i} blogs attempted")
                tqdm.write(f"{count_successful} images scraped")
                tqdm.write(f"{len(blognames_with_default_images)} default images")
                tqdm.write(f"{len(blognames_with_images)} with other images\n")

            else:
                num_rejected += 1
                tqdm.write(f"{num_rejected} rejected\n")

            pbar.update(1)
            time.sleep(.1)

        except urllib.error.HTTPError as e:
            print(e)
            print("Waiting one hour...")
            time.sleep(3600)
            continue

        except Exception as e:
            print(e)
            save_output(out_infopath, out_imagepath, i, count_successful, num_rejected, blognames_with_default_images, blognames_with_images)

    save_output(out_infopath, out_imagepath, i, count_successful, num_rejected, blognames_with_default_images, blognames_with_images)


if __name__ == '__main__':
    main()
