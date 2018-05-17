import os
import sys
import numpy as np
import pandas as pd
import html
import pdb
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support, cohen_kappa_score
import argparse


class DataHandler():

    def __init__(self, data_dirpath,
                max_num_posts=100, 
                max_post_length=200, 
                max_num_words = 50000,
                test_dev_split = 0.1):

        self.max_num_posts = max_num_posts
        self.max_post_length = max_post_length
        self.max_num_words = max_num_words
        self.test_dev_split = test_dev_split
        self.descs = None
        self.posts = None
        self.tids = None
        self.tids_split = {}
        self.cats = None
        self.X = {} # dicts ['train', 'dev', 'test'] of word indices
        self.y = {} # dicts of word indices
        self.data_dirpath = data_dirpath # where will save processed data

    def load_processed_data(self, name):
        """ Load preprocessed DataHandler object """

        vectorized_datapath = os.path.join(self.data_dirpath, f"{name}_preprocessed_data.pkl")
        with open(vectorized_datapath, 'rb') as f:
            tmp = pickle.load(f)

        self.__dict__.update(tmp)


def set_scores(preds, actual):
        """ Returns averaged measures of precision, recall and f1.
            """
        
        precisions = []
        recalls = []

        total_prec = precision_score(actual, preds, average='weighted')
        total_rec = recall_score(actual, preds, average='weighted')
        total_f1 = f1_score(actual, preds, average='weighted')
        
        return {'precision': total_prec,
                'recall': total_rec,
                'f1': total_f1}

def category_scores(preds, y, cats):
    metrics = {'kappa': cohen_kappa_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score}
    scores = {}

    for cat, i in zip(cats, list(range(preds.shape[1]))):
        pred_col = preds.T[i]
        actual_col = y.T[i]
        scores[cat] = {}
        
        for name, scorer in metrics.items():
            scores[cat][name] = scorer(actual_col, pred_col)
    
    return scores


def evaluate(preds, y, cats, output_dirpath, clf_name):

    scores = {}

    if len(cats) > 1:
        scores['set'] = set_scores(preds, y)

    # Per-category scores
    scores['category'] = category_scores(preds, y, cats)

    # Save scores
    metrics = ['precision', 'recall', 'f1', 'kappa']
    if len(cats) > 1:
        outlines = [['all'] + [scores['set'][m] for m in metrics[:-1]], \
                    *[[c] + [scores['category'][c][m] for m in metrics] for c in cats]]
    else:
        outlines = [*[[c] + [scores['category'][c][m] for m in metrics] for c in cats]]

    outlines = pd.DataFrame(outlines, columns=['category'] + metrics)
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)
    outpath = os.path.join(output_dirpath, f"{clf_name}_scores.csv")
    outlines.to_csv(outpath, index=False)

    return scores

def vectorize(dh, feats='unigrams'):
    word_inds = {}
    
    for fold in dh.X:
        word_inds[fold] = [np.array_str(d.ravel())[1:-1] for d in dh.X[fold]] # space-separated string of word indices for each list of posts

    feats_ngram_range = {'unigrams': (1,1), 'bigrams': (1,2)}
    #vec = CountVectorizer(token_pattern='\b\d+\b', ngram_range=feats_ngram_range[feats])
    vec = CountVectorizer(token_pattern='\d+', ngram_range=feats_ngram_range[feats])

    vec.fit(word_inds['train'])

    return (vec.transform(word_inds['train']), \
            vec.transform(word_inds['dev']), \
            vec.transform(word_inds['test']))


def main():

    parser = argparse.ArgumentParser(description="Train and run baselines")
    parser.add_argument('--base-dirpath', nargs='?', help="Path to parent directory with data, where should save models and output directories", default='/usr0/home/mamille2/tumblr/')
    parser.add_argument('--model-name', nargs='?', dest='model_name', help="Name of model comparing with, whose output dir will save baseline results in")
    parser.add_argument('--load-data', nargs='?', dest='load_dataname', help="Name of preprocessed data to load")
    args = parser.parse_args()

    base_dirpath = args.base_dirpath
    data_dirpath = os.path.join(base_dirpath, 'data')
    output_dirpath = os.path.join(base_dirpath, 'output', args.model_name)

    dh = DataHandler(data_dirpath)

    print("Loading preprocessed data...", end=" ")
    sys.stdout.flush()
    #dh.load_processed_data(train_data_dirpath, test_data_dirpath, train_prefix=train_prefix, test_prefix=test_prefix)
    dh.load_processed_data(args.load_dataname)
    print("done.")
    sys.stdout.flush()

    #X_train, y_train, X_dev, y_dev, X_test, y_test = dh.process_data('tweet', 'label', 
    #                feats=feats, multiclass_transform=multiclass_transform) 

    # Vectorize
    X_train, X_dev, X_test = vectorize(dh)
    y_train, y_dev, y_test = dh.y['train'], dh.y['dev'], dh.y['test']
    clfs = {'logistic_regression': LogisticRegression(),
            'svm': LinearSVC()}

    for clf_name, clf in clfs.items():
        print(f"Training classifier {clf_name}...", end=" ")
        sys.stdout.flush()

        if len(dh.cats) > 1:
            clf = OneVsRestClassifier(clf)
        clf.fit(X_train, y_train)
        print("done.")
        sys.stdout.flush()

        print("Evaluating classifier...")
        sys.stdout.flush()
        preds = clf.predict(X_test)

        results = evaluate(preds, y_test, dh.cats, output_dirpath, clf_name)
        print(results)

        print("done.")
        sys.stdout.flush()
        print()

if __name__ == '__main__':
    main()
