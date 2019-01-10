import csv
import os
import codecs
from collections import defaultdict


def _str2list(in_str):
    return [el[1:-1] for el in in_str[1:-1].split(', ')]


def update_tag_counts(tag_counts, counted_ids, candidate): # for hashtags
#     candidate_tags = [tag.lower() for tag in eval(candidate['post_tags'])] # uses tokens provided in feature tables
    candidate_tags = [tag.lower() for tag in _str2list(candidate['post_tags'])] # uses tokens provided in feature tables
    followee_id = candidate['tumblog_id_followee']    
    for tag in candidate_tags:
        if not followee_id in counted_ids[tag]: # only counts the tag if user hasn't already used the tag
            tag_counts[tag] += 1
            counted_ids[tag].add(followee_id)


# Comparison space features
def _extract_features_post_baseline_candidate(candidate, incr):
    candidate_tags = [tag.lower() for tag in eval(candidate['post_tags'])]
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
        

def extract_features_post_baseline(reblog_candidate, nonreblog_candidate, label):
    ### Post baseline
    features = defaultdict(float) # {feat: count} for each instance
    # if randomly-generated label is 1, second candidate is reblog, so flip: -1 is whatever candidate should consider first
    if label == 1: 
        _extract_features_post_baseline_candidate(nonreblog_candidate, incr=-1)
        _extract_features_post_baseline_candidate(reblog_candidate, incr=1)
    else:
        _extract_features_post_baseline_candidate(reblog_candidate, incr=-1)
        _extract_features_post_baseline_candidate(nonreblog_candidate, incr=1)

    return features



def main():

    data_dirpath = '/usr2/mamille2/tumblr/data/sample1k'

    feature_tables_dir = os.path.join(data_dirpath, 'feature_tables')
    filenames = ['reblog_features.csv', 'nonreblog_features.csv', 'ranking_labels.csv']
    joined_filenames = [os.path.join(feature_tables_dir, filename) for filename in filenames]
    csv_readers = [csv.DictReader(x.replace('\0', '') for x in open(filename, 'r')) for filename in joined_filenames]

    # Read in data
    instances = []
    instance_labels = []
    for row in zip(*csv_readers):
        reblog_features = row[0]
        nonreblog_features = row[1]
        label = int(row[2]['ranking_label'])
        instance = (reblog_features, nonreblog_features) # reblog always first, nonreblog always second
        instances.append(instance)
        instance_labels.append(label)
        
    print(len(instances), len(instance_labels))


    # # Feature Extraction

    # ### Create tag vocabulary
    counted_ids = defaultdict(lambda: set()) # for each tag, a set of followees who used those tags
    tag_counts = defaultdict(int) # count of unique followees who used each tag
    for reblog_candidate, nonreblog_candidate in instances:
        update_tag_counts(tag_counts, counted_ids, reblog_candidate)
        update_tag_counts(tag_counts, counted_ids, nonreblog_candidate)

    tag_counts_filtered = {k:v for k,v in tag_counts.items() if v > 1} # at least 2 users used the tag
    tag_vocab = tag_counts_filtered.keys()
    print(len(tag_vocab))

    identity_categories = ['age', 'ethnicity/nationality', 'fandoms', 'gender',
                           'interests', 'location', 'personality type', 'pronouns', 'relationship status', 'roleplay',
                           'sexual orientation', 'weight', 'zodiac']
    len(identity_categories)


    # ### Count category label instances
    category_label_counts = defaultdict(lambda: defaultdict(int)) # {category: {value: count_of_unique_users}}
    # counted_ids = set()
    for category in identity_categories:
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


    # ### Create category label vocabulary
    category_vocabs = defaultdict(lambda: set())
    for identity_category in category_label_counts:
        category_labels_filtered_vocab = set([k for k,v in category_label_counts[identity_category].items() if v > 1]) # min 2 users using label
        category_vocabs[identity_category] = category_labels_filtered_vocab
        #print(identity_category, len(category_vocabs[identity_category]))
        #print(category_vocabs[identity_category])
        #print('-----------------')
        #print()
        
    # ### Experiment 1 - Identity framing, presence of variables

    # In[8]:


    def extract_features_experiment_1(reblog_candidate, nonreblog_candidate, label):
        # Baseline features
    #     features = defaultdict(float)
        features = extract_features_post_baseline(reblog_candidate, nonreblog_candidate, label)
        
        # Follower features
    #     for identity_category in identity_categories:
    #         identity_category_follower = eval(reblog_candidate[identity_category + '_terms_follower'])
    #         follower_presence = len(identity_category_follower) > 0
    #         if follower_presence:
    #             feat_tag = ('follower_cat=%s' % identity_category)
    #             features[feat_tag] += 1
                
        # Follower-followee comparison space features
        def _extract_features_experiment_1_candidate(candidate, incr):
            
            num_matches = 0
            num_mismatched_follower_presents = 0
            num_mismatched_followee_presents = 0
            
            for identity_category in identity_categories:
                identity_category_follower = eval(reblog_candidate[identity_category + '_terms_follower'])
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
                if (follower_presence and not followee_presence): # XOR
                    feat_tag = ('xor_cat=%s' % identity_category)
                    features[feat_tag] += incr
                    num_mismatched_follower_presents += 1
                elif (not follower_presence and followee_presence): # XOR
                    feat_tag = ('xor_cat=%s' % identity_category)
                    features[feat_tag] += incr
                    num_mismatched_followee_presents += 1
                    
            # Number of matches
            features['num_matches'] += num_matches * incr
            features['num_mismatched_follower_presents'] += num_mismatched_follower_presents * incr
            features['num_mismatched_followee_presents'] += num_mismatched_followee_presents * incr
                
        if label == 1:
            _extract_features_experiment_1_candidate(nonreblog_candidate, incr=-1)
            _extract_features_experiment_1_candidate(reblog_candidate, incr=1)
        else:
            _extract_features_experiment_1_candidate(reblog_candidate, incr=-1)
            _extract_features_experiment_1_candidate(nonreblog_candidate, incr=1)

        return features


    # ### Experiment 2 - Compatibility

    # In[15]:


    def extract_features_experiment_2(reblog_candidate, nonreblog_candidate, label):
        # Baseline features
    #     features = defaultdict(float)
    #     features = extract_features_post_baseline(reblog_candidate, nonreblog_candidate, label)
        features = extract_features_experiment_1(reblog_candidate, nonreblog_candidate, label)

        
        # Follower features
        for identity_category in identity_categories:
            identity_category_follower = [x.lower() for x in eval(reblog_candidate[identity_category + '_terms_follower'])]
            for identity_label in identity_category_follower:
                if identity_label in category_vocabs[identity_category]:
                    feat_tag = ('cat=%s,follower_lab=%s' % (identity_category, identity_label))
                    features[feat_tag] += 1
                
        # Comparison space features
        def _extract_features_experiment_2_candidate(candidate, incr):
            for identity_category in identity_categories:
                identity_category_follower = [x.lower() for x in eval(reblog_candidate[identity_category + '_terms_follower'])]
                identity_category_followee = [x.lower() for x in eval(reblog_candidate[identity_category + '_terms_followee'])]
                for identity_label_followee in identity_category_followee:
                    if identity_label_followee in category_vocabs[identity_category]:
                        feat_tag = ('cat=%s,followee_lab=%s' % (identity_category, identity_label_followee))
                        features[feat_tag] += incr
                        
                        # Compatibility features: explicit marking of follower and followee labels together
                        for identity_label_follower in identity_category_follower:
                            if identity_label_follower in category_vocabs[identity_category]:
                                feat_tag = ('cat=%s,follower_lab=%s,followee_lab=%s' % (identity_category,
                                                                                        identity_label_follower,
                                                                                        identity_label_followee))
                                features[feat_tag] += incr
                
                    
        if label == 1:
            _extract_features_experiment_2_candidate(nonreblog_candidate, incr=-1)
            _extract_features_experiment_2_candidate(reblog_candidate, incr=1)
        else:
            _extract_features_experiment_2_candidate(reblog_candidate, incr=-1)
            _extract_features_experiment_2_candidate(nonreblog_candidate, incr=1)

        return features


    # # Run models

    # In[9]:


    from sklearn import feature_extraction
    from sklearn import linear_model
    from sklearn import model_selection
    from sklearn import preprocessing
    from sklearn import svm
    import numpy as np


    # ### Post baseline

    # In[23]:


    X = []
    y = []
    for (reblog_candidate, nonreblog_candidate), label in zip(instances, instance_labels):
        X.append(extract_features_post_baseline(reblog_candidate, nonreblog_candidate, label))
        y.append(label)
        
    post_features_vectorizer = feature_extraction.DictVectorizer()
    post_features_scaler = preprocessing.StandardScaler(with_mean=False) # normalization standard scaler
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)
    X_train = post_features_vectorizer.fit_transform(X_train)
    X_train = post_features_scaler.fit_transform(X_train)
    X_test = post_features_vectorizer.transform(X_test)
    X_test = post_features_scaler.transform(X_test)

    baseline_model = linear_model.LogisticRegressionCV(cv=10).fit(X_train, y_train) # default 5 folds
    print(baseline_model.score(X_test, y_test))
    baseline_pred = baseline_model.predict(X_test)


    # ### Experiment 1 - Identity framing, presence of variables

    # In[ ]:


    X = []
    y = []
    for (reblog_candidate, nonreblog_candidate), label in zip(instances, instance_labels):
        X.append(extract_features_experiment_1(reblog_candidate, nonreblog_candidate, label))
        y.append(label)
        
    features_vectorizer_experiment_1 = feature_extraction.DictVectorizer()
    features_scaler_experiment_1 = preprocessing.StandardScaler(with_mean=False)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)
    X_train = features_vectorizer_experiment_1.fit_transform(X_train)
    X_train = features_scaler_experiment_1.fit_transform(X_train)
    X_test = features_vectorizer_experiment_1.transform(X_test)
    X_test = features_scaler_experiment_1.transform(X_test)

    experiment_1_model = linear_model.LogisticRegressionCV(cv=10, n_jobs=10, max_iter=1000, verbose=2).fit(X_train, y_train)
    print(experiment_1_model.score(X_test, y_test))
    experiment_1_pred = experiment_1_model.predict(X_test)

    # Save predictions
    np.savetxt(os.path.join(data_dirpath, 'results', 'baseline_exp1.txt'), experiment_1_pred)

    # Save classifier (with weights)
    with open(os.path.join(data_dirpath, 'models', 'lr_baseline_exp1.pkl'), 'wb') as f:
        pickle.dump(experiment_1_model)


    # ### Experiment 2 - Compatibility

    # In[18]:


    X = []
    y = []
    for (reblog_candidate, nonreblog_candidate), label in zip(instances, instance_labels):
        X.append(extract_features_experiment_2(reblog_candidate, nonreblog_candidate, label))
        y.append(label)
        
    features_vectorizer_experiment_2 = feature_extraction.DictVectorizer()
    features_scaler_experiment_2 = preprocessing.StandardScaler(with_mean=False)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)
    X_train = features_vectorizer_experiment_2.fit_transform(X_train)
    X_train = features_scaler_experiment_2.fit_transform(X_train)
    X_test = features_vectorizer_experiment_2.transform(X_test)
    X_test = features_scaler_experiment_2.transform(X_test)

    experiment_2_model = linear_model.LogisticRegressionCV(cv=10, max_iter=1000, n_jobs=5, verbose=2).fit(X_train, y_train)
    print(experiment_2_model.score(X_test, y_test))
    experiment_2_pred = experiment_2_model.predict(X_test)

    # Save predictions
    np.savetxt(os.path.join(data_dirpath, 'results', 'baseline_exp1_exp2.txt'), experiment_2_pred)


    # # McNemar's Test (Significance)

    # In[ ]:


    a = 0
    b = 0 # Baseline correct, experiment incorrect
    c = 0 # Baseline incorrect, experiment correct
    d = 0
    for b_pred, ex_pred, true in zip(baseline_pred, experiment_1_pred, y_test):
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
    print(table)


    # In[ ]:


    # Example of calculating the mcnemar test
    from statsmodels.stats.contingency_tables import mcnemar
    # calculate mcnemar test
    result = mcnemar(table, exact=False, correction=False)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
    else:
            print('Different proportions of errors (reject H0)')

if __name__ == '__main__':
    main()
