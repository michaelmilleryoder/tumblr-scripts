import pandas as pd
import re
import numpy as np
import pickle
import os, sys
from html.parser import HTMLParser
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.corpus import words
import argparse
import pdb
import csv
import warnings
import spacy
from multiprocessing import Pool, Manager
import multiprocessing
import pdb
import itertools

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])


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
        warnings.simplefilter("ignore")
        soup = BeautifulSoup(html, 'lxml')
        for s in soup(['script', 'style']):
            s.decompose()

    return ' '.join(soup.stripped_strings)

def strip_tags(html):
    s = MLStripper()
    text = clean_html(str(html)).strip()
    s.feed(text)
    return s.get_data()

def find_body(post_content):
    m = re.search(r'body=(.*), note_count', post_content)
    if not m: return ''
    body = m.group(1)
    return body


def preprocess(column, post_type, lowercase=False, n_jobs=multiprocessing.cpu_count()):
    """ Takes a dataframe column, preprocesses all text """

    with Pool(n_jobs) as p:
        params = list(zip(column,
                itertools.repeat(post_type),
                itertools.repeat(lowercase),
                ))
        processed = list(tqdm(p.imap(preprocess_post_content, params), total=len(column), ncols=70))

    return processed

def preprocess_post_content(params):

    text, post_type, lowercase = params

    if not isinstance(text, str):
        return ''

    # Find post content
    if post_type == 'text':
        text = find_body(text)
        if text == '': return text
        
    # Strip html tags
    nohtml = strip_tags(text)
    
    # Tokenize with spaCy
    if lowercase:
        nohtml = nohtml.lower()
    toks = [tok.text for tok in nlp.tokenizer(nohtml)]
    
    # Remove whitespace tokens
    toks = [t for t in toks if not all(c==' ' for c in t)]
    
    return ' '.join(toks)

#def check_username_text(params):
def check_username_text(name):
    #name, text = params
    if name in original_text:
        present[name] = True


def remove_usernames_text(text, blog_names, debug, n_jobs):

    #if debug:
    #    for name in tqdm(list(blog_names)[:100], ncols=50):
    #        text = text.replace(name, ' ')
    #else:

    manager = Manager()
    global present
    present = manager.dict()
    global original_text
    original_text = text

    # Check what usernames are in the text (multiprocess)
    with Pool(n_jobs) as p:
        #params = list(zip(blog_names,
    #           itertools.repeat(text),
    #           ))
        #list(tqdm(p.imap(check_username_text, params), total=len(blog_names), ncols=70))
        list(tqdm(p.imap(check_username_text, blog_names), total=len(blog_names), ncols=70))

    for name in tqdm(present, total=len(present), ncols=70):
    #for name in tqdm(blog_names, total=len(blog_names), ncols=100):
        text = text.replace(name, ' ')

    return text


def remove_usernames_column(data, source=None, debug=False, n_jobs=multiprocessing.cpu_count()):
    """ Removes usernames, saves in separate columns.
        Args:
            source: Filepath to a line-separated list of usernames. 
                If None, will build a list of usernames from the post data.
    """

    if source:
        with open(source, 'r') as f:
            if debug:   
                blog_names = set()
                for i, line in enumerate(f):
#                    if i >= 100:
#                        break
                    if len(line) > 5:
                        blog_names.add(line)
            else:
                blog_names = set([name for name in f.read().splitlines() if len(name) > 5])
    else:
        blog_names = set(data['source_title'].unique()) # might not all be strings

    dict_wds = set(words.words())
    blog_names = blog_names - dict_wds
    #data['post_body_no_blognames'] = list(map(remove_usernames_text, tqdm(data['post_body'], ncols=50)))
    #data['post_body_no_blognames'] = list(map(lambda x: re.sub(p, x, ' '), tqdm(data['post_body'], ncols=50)))
    sep = ' |a|a|a|a| '
    all_text = sep.join(data['post_body'].tolist())
    all_text_no_blog_names = remove_usernames_text(all_text, blog_names, debug, n_jobs)
    if all_text_no_blog_names.count(sep) + 1 != len(data):
        pdb.set_trace()
    data['post_body_no_blognames'] = all_text_no_blog_names.split(sep)

    return data

def save_text_file(text_rows, outpath):
    with open(outpath, 'a') as f:
        for post in text_rows:
            f.write(post + '\n')


def read_dir_all(posts_dirpath, debug, batch=False):
    
    posts = []
    tqdm.write('\nConcatenating files into dataframe...')
    if debug:
        fnames = os.listdir(posts_dirpath)[:1]
    else:
        fnames = os.listdir(posts_dirpath)

    for fname in tqdm(fnames, ncols=50):
        fpath = os.path.join(posts_dirpath, fname)
        #part = pd.read_csv(fpath, sep='\t', low_memory=False)
        #part = pd.read_csv(fpath, sep='\t', low_memory=True)
        part = pd.read_csv(fpath, sep='\t', engine='python', quoting=csv.QUOTE_NONE, error_bad_lines=False, warn_bad_lines=False)
        posts.extend(part.values.tolist())

    data = pd.DataFrame(posts, columns=part.columns)
    data.drop_duplicates('post_id', inplace=True)
    data.dropna(subset=['post_id'], inplace=True)
    data = data[data['post_content'].map(lambda x: isinstance(x, str) and len(x) > 0)]

    if debug:
        return data.head(1000)

    else:
        return data


def main():

    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', help='Input path to dir or file')
    parser.add_argument('output', nargs='?', help='Output filepath')
    parser.add_argument('--post-type', dest='post_type', nargs='?', help='Post type: {text, caption}', default='text')
    #parser.add_argument('--remove-usernames', dest='remove_usernames', action='store_true')
    parser.add_argument('--remove-usernames', nargs='?', dest='remove_usernames', default=None)
    parser.add_argument('--n_jobs', nargs='?', dest='n_jobs', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--lower', dest='lower', help='Lowercase text', action='store_true', default=False)
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()

    input_format = 'dir' # {dir, tsv, pickle}
    save_text = False
    save_final = True # might be such a big file that don't want to add to pickle
    batch = False
    debug = args.debug

    # I/O
    #data_dirpath = '/usr2/mamille2/tumblr/data'
    #dataset = args.dataset
    #posts_fpath = os.path.join(data_dirpath, 'textposts_recent100.pkl') # recent 100 posts, even if don't have 100
    #posts_path = f'../data/textposts_captions/{dataset}/posts'
    #posts_outpath = f'../data/textposts_captions/{dataset}_posts.pkl'
    #blog_names_fpath = '../data/blog_names.txt'
    blog_names_fpath = args.remove_usernames
    posts_path = args.input
    if debug:
        posts_outpath = args.output + '.debug'
    else:
        posts_outpath = args.output
    text_outpath = posts_outpath[:-3] + 'txt'
    if args.post_type == 'text':
        text_col = 'post_content'
    elif args.post_type == 'caption':
        text_col = 'post_caption'
    else:
        raise ValueError("Must specify `text' or `caption' input")

    # See if dir size is too big to concatenate and process
    #if input_format == 'dir':
    #    dataset_size = sum(os.path.getsize(f) for f in os.listdir(posts_path))

    #    if dataset_size > 1e10: # 10 GB
    #        batch = True
        

    # Load posts
    print("Loading data...", end=' ')
    sys.stdout.flush()

    if input_format == 'dir':
        data = read_dir_all(posts_path, debug)

    elif input_format == 'pickle':
        if debug:
            data = pd.read_pickle(posts_fpath).head(100)
        else:
            data = pd.read_pickle(posts_fpath)
        print('done')
        sys.stdout.flush()


    # Tokenize, preprocess all posts
    print("Preprocessing posts...")
    sys.stdout.flush()

    if input_format == 'pickle':    
        data['body_toks'] = list(map(preprocess_post, tqdm(data['body'].tolist())))
        data['body_str'] = data['body_toks'].map(lambda x: ' '.join(x))

    elif input_format == 'dir':
        #data['post_body'] = list(map(preprocess_post_content, tqdm(data[text_col], ncols=50)))
        #data['post_body'] = [preprocess_post_content(t, args.post_type) for t in tqdm(data[text_col], ncols=50)]
        data['post_body'] = preprocess(data[text_col], args.post_type, lowercase=args.lower, n_jobs=args.n_jobs)
        data = data[data['post_body'].map(lambda x: len(x) > 0)]

    # Remove usernames (separate because iterate through usernames since #usernames >> #docs)
    if args.remove_usernames:
        print("Removing usernames...")
        sys.stdout.flush()
        data = remove_usernames_column(data, source=blog_names_fpath, debug=debug, n_jobs=args.n_jobs)

    # Save text file (for eg training word embeddings)
    if save_text:
        print(f"Writing text file to {text_outpath}...", end=' ')
        sys.stdout.flush()
        save_text_file(data['body_str'].tolist(), text_outpath) 
        print("done")
        sys.stdout.flush()

    # Save data
    if save_final:
        print(f"Saving tokenized file to {posts_outpath}...", end=' ')
        sys.stdout.flush()
        data.to_pickle(posts_outpath)
        print("done")
        sys.stdout.flush()

if __name__ == '__main__':
    main()
