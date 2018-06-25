import os
import sys
import numpy as np
import pandas as pd
import html
import pdb
import pickle
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split
import argparse
import datetime
from tqdm import tqdm
from scipy.sparse import csr_matrix

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
                input_type='tags',
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
        self.restr_word_index = {wd: idx for wd, idx in self.word_index.items() if idx < self.max_num_words} 
        self.vocab = list(self.word_index.keys())[:self.max_num_words]

        # For each blog, create count vector of tags
        data = np.zeros((len(posts_by_blog), len(self.vocab)), dtype='int32')
        for i, posts in enumerate(tqdm(posts_by_blog)):
            tag_inds = np.zeros(len(self.vocab))
            for post in posts:
                for tag in post:
                    if tag in self.restr_word_index:
                        tag_inds[self.restr_word_index[tag]] += 1
            data[i] = tag_inds

        data = csr_matrix(data)

        # Prepare description categories (labels)
        if outcome_colname == 'all':
            cols = self.descs.columns.tolist()
            self.cats = sorted([c for c in cols[cols.index('parsed_blog_description')+1:] if not c.endswith('terms')])

        else:
            self.cats = [outcome_colname] # can only handle 1 colname
        
        labels = np.array(list(zip(*[self.descs[cat] for cat in self.cats])))

        # Shuffle, split into train/dev/test
        test_size = int(self.test_dev_split * data.shape[0])
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

class FeedForward():
    """ Basic feed-forward neural network """

    def __init__(self, base_dirpath, name=None):
        self.model_path = None
        self.model = None
        if name:
            self.model_name = name
        else:
            self.model_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')
        self.base_dirpath = base_dirpath
        self.model_dirpath = os.path.join(base_dirpath, 'models', self.model_name)
        if not os.path.exists(self.model_dirpath):
            os.mkdir(self.model_dirpath)
        self.output_dirpath = os.path.join(base_dirpath, 'output', self.model_name)
        if not os.path.exists(self.output_dirpath):
            os.mkdir(self.output_dirpath)

    def build_model(self, vocab_len, n_cats, n_outcomes=14):
        model = Sequential()
        model.add(Dense(1024, input_dim=vocab_len, activation='relu'))
        model.add(Dense(512, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        #model.add(Flatten())
        model.add(Dense(n_cats, activation='sigmoid'))

        sgd = optimizers.SGD(lr=0.01)
        model.compile(loss='binary_crossentropy',
                          optimizer=sgd, # was better with rmsprop
                          metrics=['acc'])
        model.summary()
        self.model = model

    def train_model(self, X_train, y_train, X_dev, y_dev, epochs=100, batch_size=16):
        self.model_path = os.path.join(self.model_dirpath, 'feedforward_tags_countvec.h5')
        self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size, callbacks=[ModelCheckpoint(self.model_path, monitor='val_acc', verbose=1, save_best_only=True)])

    def set_scores(self, preds, actual):
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

    def category_scores(self, preds, y, cats):
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

    def predict(self, X):

        # Load model
        model = load_model(self.model_path)

        preds = model.predict(X)
        preds[preds>=0.5] = True
        preds[preds<0.5] = False
        return preds
    

    def evaluate(self, preds, y, cats):
        """ Evaluates, saves and returns scores """

        scores = {}

        if len(cats) > 1:
            scores['set'] = self.set_scores(preds, y)

        # Per-category scores
        scores['category'] = self.category_scores(preds, y, cats)

        # Save scores
        metrics = ['precision', 'recall', 'f1', 'kappa']
        if len(cats) > 1:
            outlines = [['all'] + [scores['set'][m] for m in metrics[:-1]], \
                        *[[c] + [scores['category'][c][m] for m in metrics] for c in cats]]
        else:
            outlines = [*[[c] + [scores['category'][c][m] for m in metrics] for c in cats]]

        outlines = pd.DataFrame(outlines, columns=['category'] + metrics)
        outpath = os.path.join(self.output_dirpath, "feedforward_countvec_10k_scores.csv")
        outlines.to_csv(outpath, index=False)

        return scores


def main():

    parser = argparse.ArgumentParser(description="Train and run baselines")
    parser.add_argument('--base-dirpath', nargs='?', help="Path to parent directory with data, where should save models and output directories", default='/usr0/home/mamille2/tumblr/')
    parser.add_argument('--model-name', nargs='?', dest='model_name', help="Name of model comparing with, whose output dir will save baseline results in")
    parser.add_argument('--dataset-name', nargs='?', dest='dataname', help="Name to save preprocessed data to")
    parser.add_argument('--outcome', nargs='?', dest='outcome_colname', help="Name of column/s to predict")
    parser.add_argument('--load-data', nargs='?', dest='load_dataname', help="Name of preprocessed data to load")
    parser.add_argument('--input-type', nargs='?', dest='input_type', help="Type of input data: {text, tags}", default='text')
    args = parser.parse_args()

    base_dirpath = args.base_dirpath
    data_dirpath = os.path.join(base_dirpath, 'data')

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
            dh = DataHandler(data_dirpath, name=args.dataname, max_num_words=10000)
        else:
            dh = DataHandler(data_dirpath, max_num_words=10000)
        dh.load_data(descs_path, posts_path)
        print("done.")
        sys.stdout.flush()
        dh.process_data(outcome_colname=outcome_colname)

    # Run feedforward network
    clf_name = 'feedfoward neural network'
    print(f"Training {clf_name}...", end=" ")
    sys.stdout.flush()
    ff = FeedForward(args.base_dirpath, name=args.model_name)
    ff.build_model(len(dh.vocab), len(dh.cats))
    ff.train_model(dh.X['train'], dh.y['train'], dh.X['dev'], dh.y['dev'])
    #ff.model_path = os.path.join(base_dirpath, f'models/{args.model_name}/feedforward_tags_countvec.h5')
    preds = ff.predict(dh.X['test'])
    print("done.")
    sys.stdout.flush()

    print(f"Evaluating {clf_name}...", end=" ")
    sys.stdout.flush()
    results = ff.evaluate(preds, dh.y['test'], dh.cats)
    print("done.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
