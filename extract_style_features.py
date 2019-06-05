import pandas as pd
import os, sys
from tqdm import tqdm as tqdm
from string import punctuation
from nltk.corpus import words
import argparse
from emoji import UNICODE_EMOJI
import regex
from collections import Counter
import pdb

en_words = set(words.words())

def extract_emoji(text):
    """ Return a dict with emoji and count """
        
    text_emoji = []
    data = regex.findall(r'\X', text)
    text_flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', text)
    
    for word in data:
        if any(char in UNICODE_EMOJI for char in word):
            text_emoji.append(word)
            
    counter = Counter(text_emoji + text_flags)
    
    return dict(counter)

def extract_style_features(text):
    
    features = {}
    
    # Number of words
    toks = text.split()
    n_words = len(toks)
    features['n_words'] = n_words
    features['n_characters'] = len(text)
    
    if n_words == 0: return features
   
    # Number, which punctuation
    total_punctuation = 0
    for p in punctuation:
        p_count = text.count(p)
        features[f'avg_{p}_per_word'] = p_count/n_words
        total_punctuation += p_count
    features['avg_punctuation'] = total_punctuation/n_words
    
    # Capitalization
    total_capitals = sum(1 for c in text if c.isupper())
    word_initial_capitals = sum(1 for w in toks if w[0].isupper())
    features['n_capitals'] = total_capitals
    features['avg_capitalized_words'] = word_initial_capitals/n_words
    features['avg_capitalized_letters'] = total_capitals/features['n_characters']
    
    # Repeated characters
    total_char_repeats = 0
    for char in set(text):
        repeated_count = text.count(''.join([char]*3))
        if repeated_count > 0:
            features[f'repeated_{char}'] = repeated_count
            total_char_repeats += repeated_count
            
    features['total_char_repeats'] = total_char_repeats

    # Emoji
    for e, val in extract_emoji(text).items():
        features[f'avg_{e}_per_word'] = val/n_words
    
    # Out-of-vocabulary words
    features['n_oov'] = sum(1 for w in toks if not w in en_words)
    features['avg_oov'] = features['n_oov']/features['n_words']

    return features


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_fpath', nargs='?', help='Filepath of pandas pickled dataframe to be processed. Saves to the same filename.')
    args = parser.parse_args()

    # I/O
    posts_path = args.input_fpath
    data = pd.read_pickle(posts_path)

    data['style_features'] = list(map(extract_style_features, tqdm(data['post_body_no_blognames'])))
    data.to_pickle(posts_path)

if __name__ == '__main__':
    main()
