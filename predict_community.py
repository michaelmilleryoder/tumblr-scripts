from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
import pickle
import pandas as pd
import argparse
import pdb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('features', nargs='+', help='unigrams style')
    parser.add_argument('--unigram-feature-selection', dest='unigrams_fs', nargs='?', help='how many selected features for unigrams', type=int)
    args = parser.parse_args()

    # # Predict community from style features (vs unigrams)
    # Load data
    print('Loading data...')
    data = pd.read_pickle('/mnt/interns/myoder/data/textposts_captions_1m.pkl')

    # ## Split data, format for sklearn
    train, test = train_test_split(data, test_size=0.1, random_state=9)
    y_train = train['community']
    y_test = test['community']

    # Style features vectorizer
    print("Fitting and running vectorizer...")
    train_features = []
    test_features = []

    if 'style' in args.features:
        style_vectorizer = DictVectorizer()
        style_vectorizer.fit(train['style_features'])
        style_train = style_vectorizer.transform(train['style_features'])
        style_test = style_vectorizer.transform(test['style_features'])
        train_features.append(style_train)
        test_features.append(style_test)

    if 'unigrams' in args.features:
        word_vectorizer = TfidfVectorizer(ngram_range=(1,1))
        word_vectorizer.fit(train['post_body_no_blognames'])
        words_train = word_vectorizer.transform(train['post_body_no_blognames'])
        words_test = word_vectorizer.transform(test['post_body_no_blognames'])

        if args.unigrams_fs:
            print('Selecting top word features...')
            selector = SelectKBest(chi2, k=args.unigrams_fs).fit(words_train, y_train)
            words_train = selector.transform(words_train)
            words_test = selector.transform(words_test)
            

        train_features.append(words_train)
        test_features.append(words_test)


    X_train = hstack(train_features)
    X_test = hstack(test_features)

    print("Normalizing...")
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # ## Classify communities
    clf = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=5000, verbose=2)
    # clf = LogisticRegression(solver='sag', multi_class='ovr', n_jobs=20, verbose=2) # extremely slow probs bc ovr?

    print("Fitting classifier...")
    clf.fit(X_train, y_train)

    print(f'Mean accuracy over classes: {clf.score(X_test, y_test)}') # mean accuracy for just style features; majority class is 10%

    # Save classifier
    if args.unigrams_fs:
        outpath = f'/mnt/interns/myoder/models/community_lr_{"_".join(features)}_{args.unigram_fs}.pkl'
    else:
        outpath = f'/mnt/interns/myoder/models/community_lr_{"_".join(features)}.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    main()
