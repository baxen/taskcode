import argparse
import numpy as np
import pandas as pd

from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import ensemble, preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from select_algorithm import optimized_classifier
from IPython import embed

from taskcode import construct

def train(optimize=False, cv=10):
    '''
    Determines optimized classifier using select algorithm and then returns
    predict_probability array for the data frame
    '''
    df = construct.load_tasks(cache=True, interval='30m', categories=True)
    X, y = construct.to_array(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # If we tell this one to optimize, we do a quick narrowly focused optimization
    # to pick a classifier
    if optimize:
        n_examples, n_features = X_train.shape
        gbc_params = {"n_estimators": [100],
                      "max_features": np.linspace(np.sqrt(n_features) / 2, np.sqrt(n_features) * 2, 5).astype(int),
                      "max_depth": range(2, 5),
                      "min_samples_split": np.linspace(2, n_examples / 50, 10).astype(int),
                      "learning_rate": np.linspace(0.01, 0.41, 5)}
        classifier, score = optimized_classifier(X_train, y_train, ensemble.GradientBoostingClassifier(), gbc_params, n_iter=60)
    else:
        gbc = ensemble.GradientBoostingClassifier(init=None, learning_rate=0.3,
                                                  loss='deviance', max_depth=4, max_features=21,
                                                  max_leaf_nodes=None, min_samples_leaf=1,
                                                  min_samples_split=100, min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, presort='auto', random_state=None,
                                                  subsample=1.0, verbose=0, warm_start=False)
        classifier = make_pipeline(preprocessing.RobustScaler(), gbc)

    # Now use cross validation to measure f1/accuracy with a confidence interval

    scores = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=StratifiedShuffleSplit(y,n_iter=cv,test_size=0.5))
    # skf = StratifiedKFold(y=y,n_folds=cv,shuffle=False)
    # for train_i, test_i in skf:
    #     y_train, y_test = y[train_i], y[test_i]
    #     print len(y_train), len(y_test)
    #     #train_counts = y_train.label.groupby(y_train.label).count()
    #     #test_counts = y_test.label.groupby(y_test.label).count()
    #     #print train_counts, test_counts
    print scores
    print "F1 Weighted: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std()/np.sqrt(cv))
    classifier.fit(X_train, y_train)
    return classifier.predict_proba(X_test)

def initialize_prob_df():
    df_p = pd.DataFrame()
    prob_data = train(optimize = False, cv = 10)
    prob_series = []
    for prob in prob_data:
        prob_series.append(sorted(prob, reverse = True))
    df_p['prob_dist'] = prob_series
    return df_p

def task_threshold(prob_array, threshold):
    '''
    Takes an array of sorted probabilities and determines minimum number required for confidence
    threshold
    '''
    conf = 0
    index = 0
    while conf < threshold:
        conf += prob_array[index]
        index += 1
    return index

def task_suggestion_conf(threshold):
    df = initialize_prob_df()
    df['threshold_set_length'] = df['prob_dist'].apply(lambda x: task_threshold(x, threshold))
    #Calculate the total length of the set
    total_length = float(len(df))
    length_95_percent = int(0.95*total_length)
    #Calculate the length of the single task set and determine the percentage of threhold%
    #confidence sets that are only of length 1
    single_set_percentage = float(len(df[df['threshold_set_length'] == 1]))/total_length*100
    single_set_percentage = "{0:.1f}".format(single_set_percentage)
    average_set_length = df['threshold_set_length'].mean()
    average_set_length = "{0:.1f}".format(average_set_length)
    df = df.sort_values(by = 'threshold_set_length', ascending = True)
    df.index = range(len(df))
    df2 = df.iloc[0:length_95_percent, :]
    set_95_max = df2['threshold_set_length'].max()
    print str(single_set_percentage) + '% of the 99% confidence sets contain only a single task suggestion.'
    print 'On average, the 99% confidence set contains ' + str(average_set_length) + ' elements.'
    print '95% of the 99% confidence sets contain ' + str(set_95_max) + ' or fewer task suggestions.'
