import csv
import os
import codecs
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
import pickle
import pdb
import argparse
import copy
from operator import itemgetter
from scipy.sparse import vstack

"""

This script contains code for experiments predicting Tumblr reblog behavior (content propagation) from post content 
and identity features of users.

This includes:
* Feature extraction
    * baseline features from post content: hashtags, post like count, post media type
    * identity features: configurations of matches and mismatches from self-presented identity labels between users 
        who may or may not reblog each others' posts
* Experiments 
    * learning-to-rank machine learning formulation with pairs of users who did share a post and pairs who did not 
        (the predicted outcome measure)
    * machine learning models from scikit-learn: logistic regression, SVM, feedforward neural network

Entrance point: main function.

"""


def run_mcnemar(baseline_pred, experiment_pred, y_test):
    """ McNemar's Test (Significance) 
    """

    a = 0
    b = 0 # Baseline correct, experiment incorrect
    c = 0 # Baseline incorrect, experiment correct
    d = 0
    for b_pred, ex_pred, true in zip(baseline_pred, experiment_pred, y_test):
        if b_pred == true and ex_pred == true:
            a += 1
        elif b_pred == true and ex_pred != true:
            b += 1
        elif b_pred != true and ex_pred == true:
            c += 1
        else:
            d += 1
            
    table = [[a, b],
             [c, d]]

    # Example of calculating the mcnemar test
    # calculate mcnemar test
    result = mcnemar(table, exact=False, correction=False)
    # summarize the finding
    #print('statistic=%.3f, p-value=%.6f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
    else:
            print('Different proportions of errors (reject H0)')
    
    return result


def _str2list(in_str):
    """ Utility function """
    return [el[1:-1] for el in in_str[1:-1].split(', ')]


def update_tag_counts(tag_counts, counted_ids, candidate):
    """ Update hashtag count for posts """
    candidate_tags = [tag.lower() for tag in _str2list(candidate['post_tags'])] # uses tokens provided in feature tables
    followee_id = candidate['tumblog_id_followee']    
    for tag in candidate_tags:
        if not followee_id in counted_ids[tag]: # only counts the tag if user hasn't already used the tag
            tag_counts[tag] += 1
            counted_ids[tag].add(followee_id)


def extract_features_post_baseline(reblog_candidate, nonreblog_candidate, label, categories=[], extras=[]):
    """ Extract baseline features from Tumblr posts """
    ### Post baseline
    features = defaultdict(float) # {feat: count} for each instance
    tag_vocab = extras[0]

    def _extract_features_post_baseline_candidate(candidate, incr):
        try:
            candidate_tags = [tag.lower() for tag in eval(candidate['post_tags'])]
        except:
            pdb.set_trace()
        for tag in candidate_tags:
            if tag.lower() in tag_vocab:
                feat_tag = ('tag=%s' % tag.lower())
                features[feat_tag] += incr

        post_type = candidate['post_type']
        feat_tag = ('post_type=%s' % post_type)
        features[feat_tag] += incr
        
        try:
            post_note_count = float(candidate['post_note_count'])
        except ValueError as e:
            post_note_count = 0.0
            
        features['post_note_count'] += incr * post_note_count
        

    if label == 1: 
        _extract_features_post_baseline_candidate(nonreblog_candidate, incr=-1)
        _extract_features_post_baseline_candidate(reblog_candidate, incr=1)
    else:
        _extract_features_post_baseline_candidate(reblog_candidate, incr=-1)
        _extract_features_post_baseline_candidate(nonreblog_candidate, incr=1)

    return features


def extract_features_experiment_1(reblog_candidate, nonreblog_candidate, label, categories=[], extras=[]):
    """ Extract features for experiment 1: identity category features 
        (such as whether one or both users being compared presents age, gender, etc) """

    features = defaultdict(float)


    # Follower-followee comparison space features
    def _extract_features_experiment_1_candidate(candidate, incr):
        
        num_matches = 0
        num_mismatched_follower_presents = 0
        num_mismatched_followee_presents = 0
        
        for identity_category in categories:
            identity_category_follower = eval(candidate[identity_category + '_terms_follower'])
            follower_presence = len(identity_category_follower) > 0
            identity_category_followee = eval(candidate[identity_category + '_terms_followee'])
            followee_presence = len(identity_category_followee) > 0
    #             if followee_presence:
    #                 feat_tag = ('followee_cat=%s' % identity_category)
    #                 features[feat_tag] += incr

            # Alignment features
    #             if ((follower_presence and followee_presence) or
    #                 (not follower_presence and not followee_presence)):
            # AND
            if (follower_presence and followee_presence): # AND
                feat_tag = ('aligned_cat=%s' % identity_category)
                features[feat_tag] += incr
                num_matches += 1
                
            # XOR
            if (follower_presence and not followee_presence):
                feat_tag = ('mismatched_follower_presents_cat=%s' % identity_category)
                features[feat_tag] += incr
                feat_tag = ('xor_cat=%s' % identity_category)
                features[feat_tag] += incr
                num_mismatched_follower_presents += 1
            elif (not follower_presence and followee_presence):
                feat_tag = ('mismatched_followee_presents_cat=%s' % identity_category)
                features[feat_tag] += incr
                feat_tag = ('xor_cat=%s' % identity_category)
                features[feat_tag] += incr
                num_mismatched_followee_presents += 1
                
        # Number of matches
        if len(categories) > 1:
            features['num_matches'] += num_matches * incr
            features['num_mismatched_follower_presents'] += num_mismatched_follower_presents * incr
            features['num_mismatched_followee_presents'] += num_mismatched_followee_presents * incr
    
            
    # Candidate comparison space
    if label == 1:
        _extract_features_experiment_1_candidate(nonreblog_candidate, incr=-1)
        _extract_features_experiment_1_candidate(reblog_candidate, incr=1)
    else:
        _extract_features_experiment_1_candidate(reblog_candidate, incr=-1)
        _extract_features_experiment_1_candidate(nonreblog_candidate, incr=1)

    return features

                
def extract_features_experiment_2(reblog_candidate, nonreblog_candidate, label, categories=[], extras=[]):
    """ Extract features for experiment 2: identity label features 
        (individual labels within categories of age, gender, etc) """

    features = defaultdict(float)

    category_vocabs = extras[1]

    def _extract_features_experiment_2_candidate(candidate, incr):

        # Comparison space features
        for identity_category in categories:
            identity_category_follower = [x.lower() for x in eval(candidate[identity_category + '_terms_follower'])]
            identity_category_followee = [x.lower() for x in eval(candidate[identity_category + '_terms_followee'])]

            if len(identity_category_follower) == 0 and len(identity_category_followee) == 0:
                identity_category_followee = []
                identity_category_follower = []
            else:
                if len(identity_category_followee) == 0:
                    identity_category_followee = ['empty']
                if len(identity_category_follower) == 0:
                    identity_category_follower = ['empty']

            union = len(set(identity_category_follower).union(set(identity_category_followee)))
            intersection = len(set(identity_category_follower).intersection(set(identity_category_followee)))

            # XOR
            features[f'cat={identity_category},xor_label'] += (union-intersection) * incr

            # AND
            features[f'cat={identity_category},aligned_label'] += intersection * incr

            # Label interaction features
            for identity_label_follower in identity_category_follower:
                for identity_label_followee in identity_category_followee:
                    feat_tag = ('cat=%s,follower_label=%s,followee_label=%s' % (identity_category, identity_label_follower, identity_label_followee))
                    features[feat_tag] += incr

    # Candidate comparison space
    if label == 1:
        _extract_features_experiment_2_candidate(nonreblog_candidate, incr=-1)
        _extract_features_experiment_2_candidate(reblog_candidate, incr=1)
    else:
        _extract_features_experiment_2_candidate(reblog_candidate, incr=-1)
        _extract_features_experiment_2_candidate(nonreblog_candidate, incr=1)

    return features


def load_data(features_dir, filenames, fpath):
    """ Load or merge the input user data and Tumblr posts """

    # if fpath exists, will load from there
    if fpath is not None and os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            instances, instance_labels = pickle.load(f)

    else:
        joined_filenames = [os.path.join(features_dir, filename) for filename in filenames]
        csv_readers = [csv.DictReader(x.replace('\0', '') for x in open(filename, 'r')) for filename in joined_filenames]

        instances = []
        instance_labels = []
        for row in zip(*csv_readers):
            reblog_features = row[0]
            nonreblog_features = row[1]
            label = int(row[2]['ranking_label'])
            instance = (reblog_features, nonreblog_features) # reblog always first, nonreblog always second
            instances.append(instance)
            instance_labels.append(label)
        
        # Save out
        parent_dir = os.path.dirname(fpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with open(fpath, 'wb') as f:
            pickle.dump((instances, instance_labels), f)
        
    return instances, instance_labels


def get_tag_vocab(instances, fpath):
    """ Build the hashtag vocab. """

    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            tag_vocab = pickle.load(f)

    else:
        counted_ids = defaultdict(lambda: set()) # for each tag, a set of followees who used those tags
        tag_counts = defaultdict(int) # count of unique followees who used each tag
        for reblog_candidate, nonreblog_candidate in instances:
            update_tag_counts(tag_counts, counted_ids, reblog_candidate)
            update_tag_counts(tag_counts, counted_ids, nonreblog_candidate)

        tag_counts_filtered = {k:v for k,v in tag_counts.items() if v > 1} # at least 2 users used the tag
        tag_vocab = tag_counts_filtered.keys()

        with open(fpath, 'wb') as f:
            pickle.dump(set(tag_vocab), f)

    return tag_vocab


def get_category_vocabs(instances, categories, fpath):
    """ Build the category vocab. """

    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            category_vocabs = pickle.load(f)

    else:
        # ### Count category label instances
        category_label_counts = defaultdict(lambda: defaultdict(int)) # {category: {value: count_of_unique_users}}
        # counted_ids = set()
        for category in categories:
            counted_ids = set() # for each category, ids already considered
            for reblog_candidate, nonreblog_candidate in instances:
                category_followee = category + '_terms_followee'
                followee_id = reblog_candidate['tumblog_id_followee']
                if not followee_id in counted_ids: # only counts labels from first instance seen of a followee, since constant
                    category_value = [x.lower() for x in eval(reblog_candidate[category_followee])]
                    for value in category_value:
                        category_label_counts[category][value] += 1
                    counted_ids.add(followee_id)
                    
                followee_id = nonreblog_candidate['tumblog_id_followee']
                if not followee_id in counted_ids:
                    category_value = [x.lower() for x in eval(nonreblog_candidate[category_followee])]
                    for value in category_value:
                        category_label_counts[category][value] += 1
                    counted_ids.add(followee_id)
                
                category_follower = category + '_terms_follower'
                follower_id = reblog_candidate['tumblog_id_follower']
                if not follower_id in counted_ids:
                    category_value = [x.lower() for x in eval(reblog_candidate[category_follower])]
                    for value in category_value:
                        category_label_counts[category][value] += 1
                    counted_ids.add(follower_id)

        # Make category vocab
        category_vocabs = defaultdict(lambda: set())
        for identity_category in category_label_counts:
            category_labels_filtered_vocab = set([k for k,v in category_label_counts[identity_category].items() if v > 1]) # min 2 users using label
            category_vocabs[identity_category] = category_labels_filtered_vocab

        with open(fpath, 'wb') as f:
            pickle.dump(dict(category_vocabs), f)

    return category_vocabs

def get_informative_features(features_vectorizer, model, model_name, output_dirpath, n=10000):
    """ Examine informative features from machine learning models. """

    feats_index2name = {v: k for k, v in features_vectorizer.vocabulary_.items()}
    feature_weights = model.coef_[0]
    
    top_indices = np.argsort(feature_weights)[-1*n:]
    top_weights = np.sort(feature_weights)[-1*n:]
    bottom_indices = np.argsort(feature_weights)[:n]
    bottom_weights = np.sort(feature_weights)[:n]

    nontag_lines = [] # to sort and print
    lines = [] # to sort and print
    
    for i, (j, w) in enumerate(zip(reversed(top_indices), reversed(top_weights))):
        feature_name = feats_index2name[j]
        if not feature_name.startswith('tag'):
            nontag_lines.append([i, feature_name, w, abs(w)])
        lines.append([i, feature_name, w, abs(w)])
            
    for i, (j, w) in enumerate(zip(bottom_indices, bottom_weights)):
        feature_name = feats_index2name[j]
        if not feature_name.startswith('tag'):
            nontag_lines.append([i, feature_name, w, abs(w)])
        lines.append([i, feature_name, w, abs(w)])

    nontag_lines = list(reversed(sorted(nontag_lines, key=itemgetter(3))))
    lines = list(reversed(sorted(lines, key=itemgetter(3))))

    # Save out
    dirpath = os.path.join(output_dirpath, 'output', 'informative_features')
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    outpath = os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_informative_features_nontag.txt')
    with open(outpath, 'w') as f:
        for l in nontag_lines:
            f.write(f'{l}\n')

    outpath = os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_informative_features.txt')
    with open(outpath, 'w') as f:
        for l in lines:
            f.write(f'{l}\n')
    
    print(f"\nSaved informative features to {outpath}")


def extract_features(feature_sets, instances, instance_labels, identity_categories, test_instances=None, test_instance_labels=None, remove_zeros=False, initialization=None, test_initialization=None, categories=['all'], model_name=None, output_dirpath=None, save=False, extras=[]):
    """ 
        Main feature extraction function that selects utility feature extractors.
    
        Args:
            remove_zeros: whether or not to remove instances where the follower and all of their followees do not give the category
    """
    # Try loading data
    if output_dirpath and model_name:
        features_fpath = os.path.join(output_dirpath, 'output', 'features', f'{model_name.replace("/", "_").replace(" ", "_")}_features.pkl')
        vectorizer_fpath = os.path.join(output_dirpath, 'output', 'feature_vectorizers', f'{model_name.replace("/", "_").replace(" ", "_")}_feature_vec.pkl')

        if os.path.exists(features_fpath):
            with open(features_fpath, 'rb') as f:
                X_train, y_train, X_test, y_test = pickle.load(f)
            with open(vectorizer_fpath, 'rb') as f:
                features_vectorizer = pickle.load(f)

            return X_train, y_train, X_test, y_test, vstack([X_train, X_test]), features_vectorizer

    feature_set_extractors = {
        'post_baseline': extract_features_post_baseline,
        'experiment1': extract_features_experiment_1,
        'experiment2': extract_features_experiment_2,
    }

    X = []
    y = []

    if categories == ['all']:
        categories = identity_categories

    if remove_zeros:
        # Build hashmap of followers that have zero presence of the category and all their followees do, too, for each category
        category_user_remove = variance_analysis(instances, identity_categories)
        remove_ids = set.intersection(*[set(category_user_remove[c]) for c in categories])


    def _extract_features(feature_sets, reblog_candidate, nonreblog_candidate, label, initial_features={}, categories=categories, extras=extras):
        instance_features = initial_features

        for feature_set in feature_sets:
            instance_features.update(feature_set_extractors[feature_set](reblog_candidate, nonreblog_candidate, label, categories=categories, extras=extras))
    
        return instance_features


    if initialization:
        initial_features = initialization
    else:
        initial_features = [{} for _ in range(len(instances))]

    if test_instances:
        initial_features_test = test_initialization

    keep_indices = []

    # Extract features for individual reblog/nonreblog pairings
    for i, ((reblog_candidate, nonreblog_candidate), label, initial) in enumerate(tqdm(zip(instances, instance_labels, initial_features), total=len(instances), ncols=50)):

        if remove_zeros:
            follower_id = reblog_candidate['tumblog_id_follower']
            if not follower_id in remove_ids:
                X.append(_extract_features(feature_sets, reblog_candidate, nonreblog_candidate, label, initial_features=initial, categories=categories, extras=extras))
                y.append(label)
                keep_indices(i)

        else:
            X.append(_extract_features(feature_sets, reblog_candidate, nonreblog_candidate, label, initial_features=initial, categories=categories, extras=extras))
            y.append(label)

    features_vectorizer = feature_extraction.DictVectorizer()
    features_scaler = preprocessing.StandardScaler(with_mean=False) # normalization standard scaler

    if test_instances:
        X_test = []
        y_test = []

        # Extract features for individual reblog/nonreblog pairings
        for i, ((reblog_candidate, nonreblog_candidate), label, initial) in enumerate(tqdm(zip(test_instances, test_instance_labels, initial_features_test), total=len(test_instances), ncols=50)):

            X_test.append(_extract_features(feature_sets, reblog_candidate, nonreblog_candidate, label, initial_features=initial, categories=categories, extras=extras))
            y_test.append(label)

        X_train = X
        y_train = y

    else:
        # split into train/test
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=12345)

    X_train = features_vectorizer.fit_transform(X_train)
    X_train = features_scaler.fit_transform(X_train)
    X_test = features_vectorizer.transform(X_test)
    X_test = features_scaler.transform(X_test)

    # Save feature vectorizer for error analysis
    if output_dirpath and model_name:
        outpath = os.path.join(output_dirpath, 'output', 'feature_vectorizers', f'{model_name.replace("/", "_").replace(" ", "_")}_feature_vec.pkl')
        if not os.path.exists(os.path.join(output_dirpath, 'output', 'feature_vectorizers')):
            os.mkdir(os.path.join(output_dirpath, 'output', 'feature_vectorizers'))
        with open(outpath, 'wb') as f:
            pickle.dump(features_vectorizer, f)

    # Save features
    if save and output_dirpath and model_name:
        dirpath = os.path.join(output_dirpath, 'output', 'features')
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        outpath = os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_features.pkl')
        with open(outpath, 'wb') as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
        

    # Save row indices of instances kept
    #if data_dirpath and model_name:
    #    outpath = os.path.join(data_dirpath, 'output', f'{model_name.replace("/", "_").replace(" ", "_")}_instances_kept.txt')
    #    with open(outpath, 'w') as f:
    #        for i in keep_indices:
    #            f.write(f"{i}\n")
    

    return X_train, y_train, X_test, y_test, X, features_vectorizer


def run_model(model_name, clf, X_train, y_train, X_test, y_test, output_dirpath):
    """ Train model, make predictions """

    model = clf.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    train_pred = model.predict(X_train)
    model_pred = model.predict(X_test)

    # Save predictions
    dirpath = os.path.join(output_dirpath, 'output', 'predictions')
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    np.savetxt(os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_test_preds.txt'), model_pred)
    np.savetxt(os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_train_preds.txt'), train_pred)

    # Save classifier (with weights)
    dirpath = os.path.join(output_dirpath, 'output', 'models')
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    with open(os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return model, score, model_pred


def main():

    # Command-line arguments
    parser = argparse.ArgumentParser(description='Extract features and run models')
    parser.add_argument('--remove-zeros', dest='remove_zeros', action='store_true')
    parser.add_argument('--experiment1', dest='experiment1', action='store_true')
    parser.add_argument('--experiment2', dest='experiment2', action='store_true')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='Run only baseline')
    parser.add_argument('--baseline-preds', dest='baseline_preds_name', nargs='?', help='Name of baseline prediction file in <output_dirpath>/output/predictions; default baseline_lr_test_preds.txt', default='baseline_lr_test_preds.txt')
    parser.add_argument('--no-significance', dest='no_significance_test', action='store_true', help="Don't do a significance test over a baseline")
    parser.add_argument('--classifier', dest='classifier_type', nargs='?', help='lr svm ffn', default='')
    parser.add_argument('--name', dest='model_name', nargs='?', help='model name base, automatically appends experiment features and classifier, None just puts classifier and features', default=None)
    parser.add_argument('--data-dirpath', dest='data_dirpath', nargs='?', help='data dirpath; default /data/websci2020_tumblr_identity/icwsm2020_sample1k', default='/data/websci2020_tumblr_identity/icwsm2020_sample1k')
    parser.add_argument('--test-dirpath', dest='test_dirpath', nargs='?', help='test dirpath if provided; if None (default) then does a random 10% split', default=None)
    parser.add_argument('--output-dirpath', dest='output_dirpath', nargs='?', help='The output dirpath will be named ``output'' inside of this parent directory. Default is /projects/websci2020_tumblr_identity', default='/projects/websci2020_tumblr_identity')
    parser.add_argument('--categories', dest='categories', nargs='?', help='default all single categories + all together', default='all+all')
    parser.set_defaults(remove_zeros=False)
    parser.set_defaults(experiment1=False)
    parser.set_defaults(experiment2=False)
    parser.set_defaults(baseline=False)
    args = parser.parse_args()

    data_dirpath = args.data_dirpath
    output_dirpath = args.output_dirpath

    feature_tables_dir = os.path.join(data_dirpath, 'feature_tables')
    filenames = ['reblog_features.csv', 'nonreblog_features.csv', 'ranking_labels.csv']

    # Read in data
    print("Loading data...")

    instances, instance_labels = load_data(feature_tables_dir, filenames, os.path.join(data_dirpath, 'processed_data', 'instances.pkl'))
    #instances, instance_labels = load_data(feature_tables_dir, filenames)

    if args.test_dirpath:
        test_instances, test_instance_labels = load_data(os.path.join(args.test_dirpath, 'feature_tables'), filenames, os.path.join(args.test_dirpath, 'processed_data', 'instances.pkl'))
    else:
        test_instances = None
        test_instance_labels = None
    print(len(instances), len(instance_labels))

    # ### Create tag vocabulary
    tag_vocab = get_tag_vocab(instances, os.path.join(data_dirpath, 'processed_data', 'tag_vocab.pkl'))
    print(f'Size of tag vocab: {len(tag_vocab)}')
    print()

    identity_categories = ['age', 'ethnicity/nationality', 'fandoms', 'gender',
                           'interests', 'location', 'personality type', 'pronouns', 'relationship status', 
                            #'roleplay',
                           'sexual orientation', 'zodiac']

    # ### Create category label vocabulary
    category_vocabs = get_category_vocabs(instances, identity_categories, os.path.join(data_dirpath, 'processed_data', 'category_vocab.pkl'))

    # Classifier definitions
    classifiers = {
        'lr': linear_model.LogisticRegressionCV(cv=10, n_jobs=10, max_iter=10000, verbose=0),
        'svm': model_selection.GridSearchCV(svm.LinearSVC(dual=False, max_iter=10000, verbose=0), {'C': [.01, .1, 1, 10, 100], 'penalty': ['l2']}, n_jobs=10, cv=10, verbose=2),
        'ffn': neural_network.MLPClassifier(hidden_layer_sizes=(100, 32, 50), activation='relu', early_stopping=True, verbose=2)
    }
    
    # ### Post baseline
    print("Extracting post baseline features...")
    X_train, y_train, X_test, y_test, baseline_X, features_vectorizer = extract_features(['post_baseline'], instances, instance_labels, identity_categories, test_instances=None, test_instance_labels=None, extras=[tag_vocab])
    
    if test_instances:
        # Extract baseline features from provided test set, if there is one
        _, _, _, _, baseline_X_test, _ = extract_features(['post_baseline'], test_instances, test_instance_labels, identity_categories, extras=[tag_vocab])

    # Run baseline
    if args.baseline:

        clf = classifiers[args.classifier_type]

        print("Running post baseline...")
        if args.model_name is None:
            model_name = f'{args.classifier_type}_baseline'
        else:
            model_name = f'{args.model_name}_{args.classifier_type}'

        tqdm.write(f"\tTraining set #features: {X_train.shape[1]}, #instances: {X_train.shape[0]}")
        model, score, baseline_preds = run_model(model_name, clf, X_train, y_train, X_test, y_test, output_dirpath)
        print(f'\tBaseline score: {score: .4f}')

        # Save informative features
        if args.classifier_type != 'ffn':
            if args.classifier_type == 'svm':
                model = model.best_estimator_
            get_informative_features(features_vectorizer, model, model_name, output_dirpath, n=10000)

    # Run experiments
    else:
        # Load baseline predictions
        #baseline_preds = np.loadtxt(os.path.join(output_dirpath, 'output', 'predictions', 'all_no_roleplay_baseline_lr_test_preds.txt'))
        if not args.no_significance_test:
            baseline_preds = np.loadtxt(os.path.join(output_dirpath, 'output', 'predictions', args.baseline_preds_name))

        experiments = []
        if args.experiment1:
            experiments.append('experiment1')
        if args.experiment2:
            experiments.append('experiment2')

        if args.model_name is None:
            base_model_name = f'{args.classifier_type}_baseline'
        else:
            base_model_name = args.model_name

        if 'experiment1' in experiments:
            base_model_name = base_model_name + f'+exp1'
        if 'experiment2' in experiments:
            base_model_name = base_model_name + f'+exp2'

        if args.categories == 'all+all':
            selected_categories = ['all'] + identity_categories
        else:
            selected_categories = args.categories.split(',')

        print(f"Running {' '.join(experiments)}...")
        for category in tqdm(selected_categories, ncols=50):

            model_name = base_model_name + f'_{category}' + f'_{args.classifier_type}'

            if args.remove_zeros: model_name = model_name + '_filtered'

            tqdm.write(f"\n{category} {' '.join(experiments)}")

            test_initialization = None if not test_instances else copy.deepcopy(baseline_X_test)
            X_train, y_train, X_test, y_test, X, features_vectorizer = extract_features(
                experiments, 
                instances, 
                instance_labels, 
                identity_categories, 
                initialization=copy.deepcopy(baseline_X), 
                remove_zeros=args.remove_zeros, 
                categories=[category], 
                test_instances=test_instances,
                test_instance_labels=test_instance_labels,
                test_initialization=test_initialization,
                model_name=model_name, 
                output_dirpath=output_dirpath, 
                save=True, 
                extras=[tag_vocab, category_vocabs])
            tqdm.write(f"Number of total instances: {X_train.shape[0] + X_test.shape[0]}")
            tqdm.write(f"\tTraining set #features: {X_train.shape[1]}, #instances: {X_train.shape[0]}")

            clf = classifiers[args.classifier_type]
            model, score, preds = run_model(model_name, clf, X_train, y_train, X_test, y_test, output_dirpath)
            print(f'\n{model_name} score: {score: .4f}\n')

            # Significance test
            if not args.no_significance_test:
                test_result = run_mcnemar(baseline_preds, preds, y_test)
                tqdm.write(f"McNemar's p-value: {test_result.pvalue: .6f}")

            # Save informative features
            if args.classifier_type != 'ffn':
                if args.classifier_type == 'svm':
                    model = model.best_estimator_
                get_informative_features(features_vectorizer, model, model_name, output_dirpath, n=10000)


if __name__ == '__main__':
    main()
