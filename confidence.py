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
from operator import itemgetter

def train(optimize=False, cv=10):
    '''
    Determines optimized classifier using select algorithm and then returns
    predict_probability array for the data frame
    '''
    #df = construct.load_tasks(cache=True, interval='30m', categories=True, gps_reduce = 'derived', q = 0.01, dens = 0.2)
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
    return [classifier.predict_proba(X_test), y_test]

def initialize_prob_df():
    '''
    Initializes a probability distribution data frame using train()
    Returns a DataFrame with a series of probability arrays sorted in descending order
    '''
    lst = train(optimize = False, cv = 10)
    prob_data = lst[0]
    true_label = sorted(lst[1].unique())
    df_p = pd.DataFrame()
    prob_series = []
    for prob in prob_data:
        prob_dict = {}
        for j,item in enumerate(prob):
            prob_dict[item] = true_label[j]
        prob_series.append(prob_dict)
    df_p['prob_dist'] = prob_series
    return df_p

def sort_prob(df, threshold):
    '''
    Takes a dataframe of sorted probabilities and determines minimum number required for confidence
    threshold. Also determines if the 99% threshold includes the desired label
    '''
    keys = df.columns.tolist()
    def constructor(lst):
        columns = df.iloc[:, 1:9].columns.tolist()
        return [list(x) for x in zip(*sorted(zip(lst, columns), reverse = True, key = itemgetter(0)))]
    df['all_probs'] = map(list, df.iloc[:, 1:9].values)
    df['sorted'] = map(constructor, df['all_probs'].values)
    df['sorted_probs'] = df['sorted'].apply(lambda x: x[0])
    df['sorted_tasks'] = df['sorted'].apply(lambda x: x[1])
    def thresholder(lst, limit):
        conf = 0
        index = 0
        while conf < limit:
            conf += lst[index]
            index += 1
        return index
    df['limit_to_threshold'] = df['sorted_probs'].apply(lambda x: thresholder(x, threshold))
    tasks = df['sorted_tasks'].tolist()
    limits = df['limit_to_threshold'].tolist()
    task_set = []
    for i,item in enumerate(tasks):
        task_set.append(item[0:limits[i]])
    df['task_set'] = task_set
    def truthfulness(column1, column2):
        column2 = [int(x) for x in column2]
        if int(column1) in column2:
            return True
        else:
            return False
    df['goods'] = map(truthfulness, df.iloc[:, 0].values, df['task_set'].values)
    return df


def task_suggestion_conf(df):
    '''
    Calculates various confidence parameters associated with task suggestion
    '''
    #Calculate the total length of the set
    total_length = float(len(df))
    length_95_percent = int(0.95*total_length)
    #Calculate the length of the single task set and determine the percentage of threhold%
    #confidence sets that are only of length 1
    single_set_percentage = float(len(df[df['limit_to_threshold'] == 1]))/total_length*100
    single_set_percentage = "{0:.1f}".format(single_set_percentage)
    #Calculate the mean number of tasks suggested for 99% confidence
    average_set_length = df['limit_to_threshold'].mean()
    average_set_length = "{0:.1f}".format(average_set_length)
    #Calculate the maximum number of suggested tasks for 95% of all probability arrays
    df = df.sort_values(by = 'limit_to_threshold', ascending = True)
    df.index = range(len(df))
    df2 = df.iloc[0:length_95_percent, :]
    set_95_max = df2['limit_to_threshold'].max()
    #Determine how many of the 99% confidence sets contain the correct classification
    df_truth = df[df['goods'] == True]
    true_length = len(df_truth)
    true_percentage = float(true_length)/float(total_length)*100
    #Print the results
    print str(single_set_percentage) + '% of the 99% confidence sets contain only a single task suggestion.'
    print 'On average, the 99% confidence set contains ' + str(average_set_length) + ' elements.'
    print '95% of the 99% confidence sets contain ' + str(set_95_max) + ' or fewer task suggestions.'
    print str(true_percentage) + '% of the 99% confidence sets contain the correct task as a suggestion'
