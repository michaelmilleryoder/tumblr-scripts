import pytumblr
import pandas as pd
import re
import random
from tqdm import tqdm
import pickle
import os
import urllib.request
from datetime import datetime
import time

def main():

    # I/O
    blognames_path = '/usr0/home/mamille2/new_home/tumblr/icwsm2020/icwsm2020_blogs_1m.csv'
    out_image_dirpath = '/data/icwsm2020_tumblr_identity/profile_images/nondefault_sample1m'
    if not os.path.exists(out_image_dirpath):
        os.mkdir(out_image_dirpath)
    out_imagepath = os.path.join(out_image_dirpath, '{:03}.png')
    out_infopath = '/projects/icwsm2020_tumblr_identity/output/scrape_info_{}.txt'

    # OAuth
    with open('../oauth.txt') as f:
        lines = f.read().splitlines()
        
    client = pytumblr.TumblrRestClient(lines[0], lines[1], lines[2], lines[3])


    # Load selected blog names
    print("Loading blog names...")
    blog_info = pd.read_csv(blognames_path, escapechar='\\', engine='python', encoding='utf8', error_bad_lines=False)
    blognames = blog_info['tumblr_blog_name'].tolist()

    # Random selection of 500 blognames (if already unique)
    num_lines = 1000000
    selection = [b for b in blognames if isinstance(b, str)][:1000000]
    print(f"Found {len(selection)} blog names\n")

    # Scrape images
    print("Scraping images...")
    count_successful = 0
    max_num = 1000000
    blognames_with_default_images = []
    blognames_with_images = []
    num_blognames_with_images = 0
    num_rejected = 0

    #pbar = tqdm(total=max_num)
    pbar = tqdm(total=int(.71*num_lines))
    for blogname in selection:
        
        if count_successful >= max_num:
            break
        
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
            pbar.update(1)
            avatar_url = response['avatar_url']
            
            if avatar_url.startswith('https://assets.tumblr.com/images/default_avatar/'):
                blognames_with_default_images.append(blogname)
            else:    
                blognames_with_images.append(blogname)
                num_blognames_with_images += 1
                r = urllib.request.urlopen(avatar_url).read()
                #with open(out_imagepath.format(count_successful), 'wb') as f:
                with open(out_imagepath.format(num_blognames_with_images), 'wb') as f:
                    f.write(r)

            tqdm.write(f"{count_successful} images scraped")
            tqdm.write(f"{len(blognames_with_default_images)} default images")
            tqdm.write(f"{len(blognames_with_images)} with other images\n")

        else:
            num_rejected += 1
            tqdm.write(f"{num_rejected} rejected\n")

        time.sleep(.1)

    # Save out info on the run
    with open(out_infopath.format(datetime.now().strftime("%Y-%m-%dT%H$M")), 'w') as f:
        f.write(f'Outpath: {out_imagepath}\n')
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

if __name__ == '__main__':
    main()
