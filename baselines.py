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
from sklearn.model_selection import train_test_split
import argparse
import datetime
os.environ['KERAS_BACKEND']='theano'

# Use CPU
#os.environ['THEANO_FLAGS'] = 'device=cpu'

# Use GPU
os.environ['CUDA_VISIBLE_DEVICES']='3'
os.environ['THEANO_FLAGS'] = 'device=cuda'
os.environ['THEANO_FLAGS'] = 'floatX=float32'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Flatten
from keras.callbacks import ModelCheckpoint
from keras import optimizers

class DataHandler():

    def __init__(self, data_dirpath, 
                name=datetime.datetime.now().strftime('%Y-%m-%dT%H-%M'),
                input_type='text',
                max_num_posts=100, 
                max_post_length=200, 
                max_num_words = 50000,
                test_dev_split = 0.1):

        if not input_type in ['text', 'tags']:
            raise ValueError('Input type must be "text" or "tags"')
    
        self.name = name
        self.input_type = input_type
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

    def load_data(self, descs_filepath, posts_filepath):

        # Load descriptions
        self.descs = pd.read_pickle(descs_filepath)
        self.posts = pd.read_pickle(posts_filepath)
        self.tids = sorted(self.descs['tumblog_id'].tolist())

    def load_processed_data(self, name):
        """ Load preprocessed DataHandler object """

        vectorized_datapath = os.path.join(self.data_dirpath, f"{name}_preprocessed_data.pkl")
        with open(vectorized_datapath, 'rb') as f:
            tmp = pickle.load(f)

        self.__dict__.update(tmp)

    def process_data(self, outcome_colname='all', save=True):
        """ Preprocesses data and returns vectorized form """
        print("Preprocessing data...", end=" ")
        sys.stdout.flush()

        # Get posts
        if self.input_type == 'text':
            input_colname = 'body_toks_str_no_titles'
        elif self.input_type == 'tags':
            input_colname = 'parsed_tags_minfreq3'
        posts_by_blog = [[p for p in self.posts[self.posts['tumblog_id']==tid][input_colname].tolist()] for tid in self.tids] # list of 100 posts/user
        all_posts = [p for posts in posts_by_blog for p in posts]

        # Tokenize posts, build vocab
        if self.input_type == 'text':
            tokenizer = Tokenizer(num_words=self.max_num_words,   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”')
        elif self.input_type == 'tags':
            tokenizer = Tokenizer(filters='') # don't strip punctuation, but lowercase
        tokenizer.fit_on_texts(all_posts)
        self.word_index = tokenizer.word_index
        self.vocab = list(self.word_index.keys())[:self.max_num_words]

        # Fill in posts with word IDs from vocab
        data = np.zeros((len(posts_by_blog), self.max_num_posts, self.max_post_length), dtype='int32')

        for i, posts in enumerate(posts_by_blog):
            for j, post in enumerate(posts):
                if j < self.max_num_posts:
                    wordTokens = text_to_word_sequence(post)
                    k=0
                    for _, word in enumerate(wordTokens):
                        if k < self.max_post_length and word in self.word_index and self.word_index[word] < self.max_num_words:
                            data[i,j,k] = tokenizer.word_index[word]
                            k=k+1                    

        # Prepare description categories (labels)
        if outcome_colname == 'all':
            cols = self.descs.columns.tolist()
            self.cats = sorted([c for c in cols[cols.index('parsed_blog_description')+1:] if not c.endswith('terms')])

        else:
            self.cats = [outcome_colname] # can only handle 1 colname
        
        labels = np.array(list(zip(*[self.descs[cat] for cat in self.cats])))

        # Shuffle, split into train/dev/test
        test_size = int(self.test_dev_split * len(data))
        X_train, self.X['test'], y_train, self.y['test'], \
                tids_train, self.tids_split['text'] = \
            train_test_split(data, labels, self.tids, test_size=test_size, random_state=0)

        self.X['train'], self.X['dev'], self.y['train'], self.y['dev'], tids_train, self.tids_split['dev'] = train_test_split(X_train, y_train, tids_train, test_size=test_size, random_state=0)

        # Save vectorized data
        if save:
            vectorized_datapath = os.path.join(self.data_dirpath, f"{self.name}_preprocessed_data.pkl")

            dict_save = self.__dict__.copy()
            dict_save.pop('descs')
            dict_save.pop('posts')

            with open(vectorized_datapath, 'wb') as f:
                pickle.dump(dict_save, f)

        print("done.")
        sys.stdout.flush()

        print(f"Saved preprocessed data to {vectorized_datapath}")        
        sys.stdout.flush()


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
    """ Evaluates, saves and returns scores """

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
    outpath = os.path.join(output_dirpath, f"{clf_name.replace(' ', '_')}_scores.csv")
    outlines.to_csv(outpath, index=False)

    return scores

def vectorize_tags(dh):

    """ Flatten post tags for simple blog representation of tags used """

    flattened_X = {}
    for fold, data in dh.X.items():
        flattened_X[fold] = data.reshape(data.shape[0], data.shape[1]*data.shape[2])

    return flattened_X['train'], \
            flattened_X['dev'], \
            flattened_X['test']

def vectorize_text(dh, feats='unigrams', n_feats=100000):

    """ For vectorizing ngrams. Uses word indices of posts, which by default is stored as lists in dh.X """

    word_inds = {}
    
    for fold in dh.X:
        word_inds[fold] = [np.array_str(d.ravel())[1:-1] for d in dh.X[fold]] # space-separated string of word indices for each list of posts

    feats_ngram_range = {'unigrams': (1,1), 'bigrams': (1,2)}
    vec = CountVectorizer(token_pattern='\d+', ngram_range=feats_ngram_range[feats], max_features=n_feats)

    vec.fit(word_inds['train'])

    return (vec.transform(word_inds['train']), \
            vec.transform(word_inds['dev']), \
            vec.transform(word_inds['test']))


def feedforward(X_train, y_train, X_dev, y_dev, X_test, vocab_size, max_post_length, max_num_posts, n_cats, model_dirpath):

    """ Train feedforward neural network, return filepath of saved best model. """

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = max_post_length * max_num_posts))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Flatten())
    model.add(Dense(n_cats, activation='sigmoid'))

    sgd = optimizers.SGD(lr=0.01) # 0.2 learning something
    model.compile(loss='binary_crossentropy',
                      optimizer=sgd, # was better with rmsprop
                      metrics=['acc'])
    model.summary()

    if not os.path.exists(model_dirpath):
        os.mkdir(model_dirpath)
    out_modelpath = os.path.join(model_dirpath, 'feedforward_tags.h5')
    model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=100, batch_size=10, callbacks=[ModelCheckpoint(out_modelpath, monitor='val_acc', verbose=1, save_best_only=True)])

    return out_modelpath

def model_predict(model_path, X):

    # Load best model
    model = load_model(model_path)

    preds = model.predict(X)

    preds[preds>=0.5] = True
    preds[preds<0.5] = False

    return model.predict(X)


def main():

    parser = argparse.ArgumentParser(description="Train and run baselines")
    parser.add_argument('--base-dirpath', nargs='?', help="Path to parent directory with data, where should save models and output directories", default='/usr0/home/mamille2/tumblr/')
    parser.add_argument('--model-name', nargs='?', dest='model_name', help="Name of model comparing with, whose output dir will save baseline results in")
    parser.add_argument('--dataset-name', nargs='?', dest='dataname', help="Name to save preprocessed data to")
    parser.add_argument('--outcome', nargs='?', dest='outcome_colname', help="Name of column/s to predict")
    parser.add_argument('--load-data', nargs='?', dest='load_dataname', help="Name of preprocessed data to load")
    parser.add_argument('--input-type', nargs='?', dest='input_type', help="Type of input data: {text, tags}", default='text')
    parser.add_argument('--neural', dest='neural', help='Run neural baselines in addition to non-neural', action='store_true')
    args = parser.parse_args()

    base_dirpath = args.base_dirpath
    data_dirpath = os.path.join(base_dirpath, 'data')
    output_dirpath = os.path.join(base_dirpath, 'output', args.model_name)
    model_dirpath = os.path.join(base_dirpath, 'models', args.model_name)

    descs_path = os.path.join(data_dirpath, 'blog_descriptions_recent100_100posts.pkl')
    posts_path = os.path.join(data_dirpath, 'textposts_100posts.pkl')

    if args.outcome_colname:
        outcome_colname = args.outcome_colname
    else:
        outcome_colname = 'all'


    if args.load_dataname:
        dh = DataHandler(data_dirpath)

        print("Loading preprocessed data...", end=" ")
        sys.stdout.flush()
        #dh.load_processed_data(train_data_dirpath, test_data_dirpath, train_prefix=train_prefix, test_prefix=test_prefix)
        dh.load_processed_data(args.load_dataname)
        print("done.")
        sys.stdout.flush()

    else:
        print("Loading data...", end=' ')
        sys.stdout.flush()
        if args.dataname:
            dh = DataHandler(data_dirpath, name=args.dataname, max_num_words=100000)
        else:
            dh = DataHandler(data_dirpath, max_num_words=100000)
        dh.load_data(descs_path, posts_path)
        print("done.")
        sys.stdout.flush()
        dh.process_data(outcome_colname=outcome_colname)

    # Vectorize
    if args.neural:
        X_train, X_dev, X_test = vectorize_tags(dh)
    else:
        n_feats = 10000
        X_train, X_dev, X_test = vectorize_text(dh, n_feats=n_feats)
    y_train, y_dev, y_test = dh.y['train'], dh.y['dev'], dh.y['test']

    if args.neural:
        # Run neural baselines
        clf_name = 'feedfoward neural network'
        print(f"Training {clf_name}...", end=" ")
        sys.stdout.flush()
        model_path = os.path.join(model_dirpath, 'feedforward_tags.h5')
        #model_path = feedforward(X_train, y_train, X_dev, y_dev, X_test, len(dh.vocab), dh.max_post_length, dh.max_num_posts, len(dh.cats), model_dirpath)
        preds = model_predict(model_path, X_test)

        print("done.")
        sys.stdout.flush()
        print(f"Evaluating {clf_name}...", end=" ")
        sys.stdout.flush()
        results = evaluate(preds, y_test, dh.cats, output_dirpath, clf_name)
        print("done.")
        sys.stdout.flush()

        #clf_name = 'convolutional neural network'
        #print(f"Training {clf_name}...", end=" ")
        #sys.stdout.flush()
        #preds = cnn(X_train, y_train, X_dev, y_dev, X_test, len(dh.vocab), dh.max_post_length, dh.max_num_posts, len(dh.cats), model_dirpath)
        #print("done.")
        #sys.stdout.flush()
        #print(f"Evaluating {clf_name}...", end=" ")
        #sys.stdout.flush()
        #results = evaluate(preds, y_test, dh.cats, output_dirpath, clf_name)
        #print("done.")
        #sys.stdout.flush()

    else:
        clfs = {'logistic_regression': LogisticRegression(),
                'svm': LinearSVC()}
        # Run non-neural baselines
        for clf_name, clf in clfs.items():
            print(f"Training classifier {clf_name}...", end=" ")
            sys.stdout.flush()

            if len(dh.cats) > 1:
                clf = OneVsRestClassifier(clf)
            clf.fit(X_train, y_train)
            print("done.")
            sys.stdout.flush()

            print("Evaluating classifier...", end=" ")
            sys.stdout.flush()
            preds = clf.predict(X_test)

            results = evaluate(preds, y_test, dh.cats, output_dirpath, clf_name)

            print("done.")
            sys.stdout.flush()
            print()

if __name__ == '__main__':
    main()
