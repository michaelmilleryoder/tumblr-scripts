import pandas as pd
from tqdm import tqdm as tqdm

collections = [
     'bts',
     'miraculous_ladybug',
     'riverdale',
     'simblr',
     'south_park',
     'svtfoe',
     'steven_universe',
     'studyblr',
     'voltron',
    'writers_on_tumblr',
]

for collection in tqdm(collections):
    
    # Load text posts
    fpath = f'/mnt/interns/myoder/textposts_captions/{collection}/textposts.pkl'
    posts = pd.read_pickle(fpath)
    posts['text_type'] = ['text_post'] * len(posts)
    
    # Load captions
    fpath = f'/mnt/interns/myoder/textposts_captions/{collection}/captions.pkl'
    captions = pd.read_pickle(fpath)
    captions['text_type'] = ['caption'] * len(captions)
    
    # Save out total
    total = pd.concat([posts, captions])
    total.to_pickle(f'/mnt/interns/myoder/textposts_captions/{collection}/{collection}.pkl')
    tqdm.write(collection)
#     print(total.columns)
    tqdm.write(total.shape)
