import argparse
import numpy as np

from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import ensemble, preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from select_algorithm import optimized_classifier
from IPython import embed

from taskcode import construct
    
def train(optimize=False, cv=10):
    df = construct.load_tasks(cache=True, interval='30m', categories=True, gps_reduce='derived',q=0.01,dens=0.0)
    #df = construct.load_tasks(cache=True, interval='30m', categories=True)
    X, y = construct.to_array(df)

    # If we tell this one to optimize, we do a quick narrowly focused optimization
    # to pick a classifier
    if optimize:
        X_train, X_test, y_train, y_test = train_test_split(X, y)

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
    


def main():
    parser = argparse.ArgumentParser('Train GBC on dataset and test.')
    parser.add_argument('--optimize', action='store_true',
                        help='Run cross-validation to choose optimal paremeters. Otherwise use saved parameters.')
    parser.add_argument('--kfold', type=int, default=10,
                        help='Number of fold to use for cv measurement.')
    args = parser.parse_args()

    train(args.optimize, args.kfold)


if __name__ == "__main__":
    main()
