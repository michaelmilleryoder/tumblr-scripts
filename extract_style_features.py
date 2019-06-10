import pandas as pd
import os, sys
from tqdm import tqdm as tqdm
from string import punctuation
from nltk.corpus import words
import argparse
from emoji import UNICODE_EMOJI
import regex
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import pdb

en_words = set(words.words())

old_internet_terms = ['lol', 'ur', 'l8r', 'gr8', '2', '4', 'b4', 'teh', 'pwn',
                        'pwned', 'wht', 'r', 'u', '*~', 'y', ':-)', ';-)',
                        ':)', ':0', ':(', ':-(',
                        ] # should make a class

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

def old_internet_speak(text):

    old_features = []

    for t in old_internet_terms:
        p = re.compile(r'\b{}\b'.format(re.escape(t)), re.IGNORECASE)
        if re.search(p, text):
            old_features.append(t)

    return dict(Counter(old_features))

def spaced_words(text):

    p = re.compile(r'(\w\ ){3}')
    return len(re.findall(p, text))
    #if re.search(p, text):
    #    return len(re.findall(p, text))

    #else:
    #    return 0


def word_shapes(text):

    toks = text.split()
    n_words = len(toks)

    lowercase = 0
    uppercase = 0
    capitalized = 0
    studly_caps = 0

    for w in toks:
        if all([c.islower() for c in w]):
            lowercase += 1
        elif all([c.isupper() for c in w]):
            uppercase += 1
        elif w[0].isupper():
            capitalized += 1
        elif w[0].islower() and any([c.isupper() for c in w[1:]]):
            studly_caps += 1

    return lowercase/n_words, uppercase/n_words, capitalized/n_words, studly_caps/n_words


def extract_style_features(text, vectorizers):
    
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
        if p_count > 0:
            features[f'avg_{p}_per_word'] = p_count/n_words
        total_punctuation += p_count
    features['avg_punctuation'] = total_punctuation/n_words

    ## Punctuation bigrams
    #punc = r'|'.join(punctuation)
    #punc_text = ' '.join(re.split(punc, text))
    bigram_char_vectorizer = vectorizers[0]
    bigram_counts = bigram_char_vectorizer.transform([text]).A
    pdb.set_trace()
    punct_bigrams = {ngram: bigram_counts[0][i]/n_words for ngram, i in bigram_char_vectorizer.vocabulary_.items() if bigram_counts[0][i] != 0}
    features.update(punct_bigrams)
    
    # Capitalization
    total_capitals = sum(1 for c in text if c.isupper())
    #word_initial_capitals = sum(1 for w in toks if w[0].isupper())
    #features['n_capitals'] = total_capitals
    #features['avg_capitalized_words'] = word_initial_capitals/n_words
    features['avg_capitalized_letters'] = total_capitals/features['n_characters']

    ## Word shapes
    features['aa'], features['AA'], features['Aa'], features['aA'] = word_shapes(text)
    
    # Repeated characters
    total_char_repeats = 0
    for char in set(text):
        repeated_count = text.count(''.join([char]*3))
        if repeated_count > 0:
            features[f'repeated_{char}'] = repeated_count
            total_char_repeats += repeated_count
            
    features['char_repeats_per_word'] = total_char_repeats/n_words

    # Emoji
    total_emoji = 0
    for e, val in extract_emoji(text).items():
        features[f'avg_{e}_per_word'] = val/n_words
        total_emoji += val

    features['avg_emoji_per_word'] = total_emoji/n_words

    # Old internet speak
    for f, val in old_internet_speak(text).items():
        features[f'avg_{f}_per_word'] = val/n_words

    # Space between characters
    n_spaced = spaced_words(text)
    if n_spaced > 0:
        features[f'avg_spaced_words'] = n_spaced/n_words

    # Other symbols
    other_symbols = ['™']
    for s in other_symbols:
        if s in text:
            features[f'avg_{s}_per_word'] = text.count(s)/n_words
    
    # Out-of-vocabulary words
    total_oov = sum(1 for w in toks if not w.lower() in en_words)
    features['avg_oov_per_word'] = total_oov/n_words

    return features


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_fpath', nargs='?', help='Filepath of pandas pickled dataframe to be processed. Saves to the same filename.')
    args = parser.parse_args()

    # I/O
    posts_path = args.input_fpath
    print("Loading data...")
    data = pd.read_pickle(posts_path)

    # Build vectorizers
    print("Building vectorizers...")
    vectorizers = []

    ## Get all punct bigrams
    bigram_char_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2))
    bigram_char_vectorizer.fit(data['post_body_no_blognames'])
    punct_bigrams = [ngram for ngram in bigram_char_vectorizer.vocabulary_ if all(w == ' ' or w in punctuation for w in ngram)]

    bigram_char_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2), vocabulary=punct_bigrams)
    bigram_char_vectorizer._validate_vocabulary()
    
    vectorizers.append(bigram_char_vectorizer)

    # Extracting style features
    #data['style_features'] = list(map(extract_style_features, tqdm(data['post_body_no_blognames'])))
    print("Extracting features...")
    data['style_features'] = [extract_style_features(text, vectorizers) for text in tqdm(data['post_body_no_blognames'].tolist())]
    data.to_pickle(posts_path)

if __name__ == '__main__':
    main()
