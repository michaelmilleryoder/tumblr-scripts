import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
import pickle
import os, sys
from html.parser import HTMLParser
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import warnings
import pdb
import nltk


def load_data(desc_fpath, sep='\t')
    tqdm.write("Loading data...")
    sys.stdout.flush()
    
    if desc_fpath.endswith('.csv') or desc_fpath.endswith('.tsv'):
        desc_data = pd.read_csv(desc_fpath, sep=sep, escapechar='\\')
    elif desc_fpath.endswith('.pkl'):
        desc_data = pd.read_pickle(desc_fpath)
    else:
        raise ValueError("Descriptions path not csv or pickle.")

    # Remove duplicates by tumblog id
    #desc_data.drop_duplicates(inplace=True, subset=['tumblog_id'])
    #tqdm.write(f'#descriptions: {len(desc_data)}')
    
    return desc_data

    #if 'parsed_blog_description' in desc_data.columns:
    #    tqdm.write("Descriptions already preprocessed")
    #    sys.stdout.flush()
    #    preprocessed = True

    #else:
    #    preprocessed = False
    #
    #return desc_data, preprocessed

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def clean_html(html):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        soup = BeautifulSoup(html, 'lxml')
    for s in soup(['script', 'style']):
        s.decompose()
    return ' '.join(soup.stripped_strings)

def strip_tags(html):
    s = MLStripper()
    text = clean_html(str(html).strip())
    s.feed(text)
    return s.get_data()

def process_url(url):
    url = re.sub(r'https?:\/\/', '', url).replace('www.','').replace('.', '-')
    url = re.sub(r'\/$', '', url).replace('/', '-')
    return url

def process_urls(desc):
    processed = desc
    
    for m in re.finditer(r'\S+(?:\.com|\.org|\.edu)\S*|https?:\/\/\S*', desc):
        processed = processed.replace(m.group(), process_url(m.group()))
    
    return processed

def process_dates(desc):
    processed = desc
    date_p = re.compile(r'(?:\d{4}|\d{2})(?:\.|\/)\d{2}(?:\.|\/)(?:\d{4}|\d{2})')
    
    for m in re.finditer(date_p, desc):
        processed = processed.replace(m.group(), process_date(m.group()))
    
    return processed

def process_date(datestr):
    return datestr.replace('/', '-').replace('.', '-')

def tokenize(text):
    return ' '.join([tok.lower() for tok in nltk.word_tokenize(str(text))])

def preprocess(desc_data, desc_colname):
    # Remove HTML tags
    tqdm.write("Removing HTML tags...")
    sys.stdout.flush()
    new_colname = f"processed_{desc_colname}"

    with Pool(15) as pool:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #desc_data[new_colname] = list(map(strip_tags, tqdm(desc_data[desc_colname].tolist())))
            desc_data[new_colname] = list(pool.map(strip_tags, desc_data[desc_colname].tolist()))

    # Tokenize
    print('Tokenizing...')
    desc_data[new_colname] = [tokenize(d) for d in tqdm(desc_data[new_colname].tolist())]

    # Remove empty parsed blogs
    desc_data = desc_data[desc_data[new_colname].map(lambda x: len(x) > 0)]

    # # Process URLs (to hyphens) so don't interact with list descriptions
    #tqdm.write("Processing URLs...", end=' ')
    #sys.stdout.flush()

    #desc_data['parsed_blog_description'] = list(map(process_urls, tqdm(desc_data['parsed_blog_description'].tolist())))
    #tqdm.write("done\n")

    ## # Process dates so don't interact with list descriptions
    #tqdm.write("Processing dates...", end=' ')
    #sys.stdout.flush()

    #desc_data['parsed_blog_description'] = list(map(process_dates, tqdm(desc_data['parsed_blog_description'].tolist())))
    #tqdm.write("done\n")

    return desc_data


def is_list_desc(in_str):
    if re.search(desc_re, in_str):
        return True
    else:
        return False


def segment_desc(desc, delims):
    #delims = ['|', '/', '.', '\\']
    
    # take max segmentation length to select delimiter (may be a bit inefficient)
    delim_ctr = {d: desc.count(d) for d in delims}
    sep = sorted(delim_ctr, key=delim_ctr.get, reverse=True)[0]
    
    return sep, [el.strip().lower() for el in desc.split(sep) if len(el) > 0 and el != ' ']


def split_rm_punct(segments):
    """ Return segments split on punctuation, punctuation removed """
    
    new_segs = []
    
    for seg in segments:
        new_seg = ' '.join(re.split(r'\W', seg))
        new_seg = re.sub(r'\W', ' ', new_seg)
        new_seg = re.sub(r'\s+', ' ', new_seg).strip()
        new_segs.append(new_seg)
        
    return new_segs


def list_descriptions(desc_data, sep_chars_fpath, char_limit, list_desc_fpath):
    # Segment blog descriptions into list descriptions
    tqdm.write("Identifying list descriptions...")
    sys.stdout.flush()

    ## Identify list descriptions
    with open(sep_chars_fpath, 'rb') as f:
        sep_counts = pickle.load(f)
    #seps = ['|', '/', '\\', '.']

    # Load separators, but drop alphanumeric ones
    seps = [s for s in sep_counts.keys() if len(s) > 0 and not re.search('[a-zA-Z0-9]', s)]

    # Optionally remove punctuation chars
    #punc = ['.', ',', ';', '!', '?', "'", '"']
    punc = ['!', '?', "'", '"']
    #seps = [s for s in seps if not any([p in s for p in punc])]

    # Add some escape characters for the re search
    check_chars = ['[', ']', '(', ')', '|']
    for i, s in enumerate(seps):
        if len(s) > 1 and any([c in s for c in check_chars]):
            s = s[0] + s[1:].replace(')', '\)').replace('(','\(').replace('[','\[').replace(']','\]').replace('|', '\|')
            seps[i] = s

    tqdm.write('{} total separators'.format(len(seps)))


    desc_re = '|'.join([r'^.*\{0}.+\{0}.*$'.format(s) for s in seps])
    mask = list(map(lambda x: True if re.search(desc_re, x) else False, tqdm(desc_data['parsed_blog_description'].tolist())))
    list_desc_data = desc_data.loc[mask].copy()

    tqdm.write("Segmenting descriptions...")
    sys.stdout.flush()
    #list_desc_data['segments'] = list(map(segment_desc, tqdm(list_desc_data['parsed_blog_description'].tolist())))
    seps, segs = list(zip(*[segment_desc(d, seps) for d in tqdm(list_desc_data['parsed_blog_description'])]))
    list_desc_data['separator'] = seps
    list_desc_data['segments'] = segs

    # Remove lines with 0 or 1 segments
    list_desc_data = list_desc_data.loc[list_desc_data['segments'].map(lambda x: len(x) > 1)]
    tqdm.write(f'#list descriptions: {len(list_desc_data)}\n')

    # Restrict segment lengths
    tqdm.write("Restricting segment lengths...")
    sys.stdout.flush()

    rm_list = [
        'relax and watch!',
        'my snapChat: lovewet9x',
        '/_        @will4i20    _\\',
        'my snapchat: sexybaby9x'
        'Please wait' # think this one is like a 404 error, not user-entered
    ]

    list_desc_data.loc[:, f'restr_segments_{char_limit}'] = list_desc_data['segments'].map(
        lambda d: [s for s in d if (len(s) <= char_limit and len(s) > 1) if not s in rm_list]
        )

    # Filter out empty restricted segments
    restr_desc_data = list_desc_data.loc[list_desc_data[f'restr_segments_{char_limit}'].map(lambda x: len(x) > 0)].copy()
    tqdm.write(f'#restricted list descriptions: {len(restr_desc_data)}')

    # Remove punctuation from segments
    tqdm.write("Removing punctuation from segments...")
    restr_desc_data.loc[:, 'segments_25_nopunct'] = list(map(split_rm_punct, tqdm(restr_desc_data['restr_segments_25'].tolist())))

    tqdm.write(f"Saving list descriptions to {list_desc_fpath}...")
    sys.stdout.flush()
    restr_desc_data.to_pickle(list_desc_fpath)

    return restr_desc_data

def save(desc_data, desc_fpath):
    tqdm.write(f"Saving parsed blog descriptions to {desc_fpath}...")
    sys.stdout.flush()
    #desc_data.to_pickle(desc_fpath)
    desc_data.to_csv(desc_fpath, sep='\t', index=False)


def main():

    # Settings
    char_limit = 25

    # Folder input I/O
    #data_dirpath = '/usr0/home/mamille2/tumblr/data' 
    #data_dirpath = '/usr2/mamille2/tumblr/data' 
    #data_dirpath = '/usr0/home/mamille2/erebor/tumblr/data/sample200/' 
    #data_dirpath = '/usr2/mamille2/tumblr/data/sample1k/' 
    #in_dirpath = os.path.join(data_dirpath, 'reblogs_descs_nodups')
    #out_dirpath = os.path.join(data_dirpath, 'reblogs_descs_preprocessed')
    #if not os.path.exists(out_dirpath):
    #    os.mkdir(out_dirpath)
    ##desc_fpath = os.path.join(data_dirpath, 'blog_descriptions_recent100.pkl')
    ##desc_fpath = os.path.join(data_dirpath, 'reblogs_descs.tsv')
    #desc_fnames = sorted(os.listdir(in_dirpath))
    ##list_desc_fpath = os.path.join(data_dirpath, f'bootstrapped_list_descriptions_recent100_restr{char_limit}.pkl')
    ##sep_chars_fpath = os.path.join(data_dirpath, "common_sep_chars.pkl")
    ##desc_colnames = ['blog_description_follower', 'blog_description_followee']    
    #desc_colnames = ['tumblr_blog_description']    

    #for desc_fname in tqdm(desc_fnames, ncols=50):
    #    desc_fpath = os.path.join(in_dirpath, desc_fname)
    #    out_fpath = os.path.join(out_dirpath, desc_fname)

    #    # Load data
    #    #descs, preprocessed = load_data(desc_fpath)
    #    descs = load_data(desc_fpath)
    #    if len(descs) == 0:
    #        continue

    #    #if not preprocessed:
    #    #    descs = preprocess(descs)

    #    for desc_colname in desc_colnames:
    #        descs = preprocess(descs, desc_colname)

    #    # Save data
    #    save(descs, out_fpath)

    #    #list_descs = list_descriptions(descs, sep_chars_fpath, char_limit, list_desc_fpath)

    # File input I/O
    desc_fpath = '/data/tumblr_community_identity/dataset1114k/blog_info_dataset114k.csv'
    out_fpath = desc_fpath[:-4] + '_processed.csv'
    desc_colnames = ['tumblr_blog_description']    

    # Load data
    descs = load_data(desc_fpath)
    for desc_colname in desc_colnames:
        descs = preprocess(descs, desc_colname)

    # Save data
    save(descs, out_fpath)

    #list_descs = list_descriptions(descs, sep_chars_fpath, char_limit, list_desc_fpath)

    tqdm.write("FINISHED PREPROCESSING")

if __name__=='__main__':
    main()
