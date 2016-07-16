import scipy
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn import svm, neighbors, ensemble, preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split

from IPython import embed

# ----------------------------------------
# Algorithm Selection/Optimization
# ----------------------------------------

def optimized_classifier(X, y, classifier, distributions, scorer='f1_weighted', n_iter=30, cv=3):
    """
    Return best classifier and scores for X,y from a randomized search over parameters

    X             -- Features for each sample
    y             -- Class label for each sample
    classifier    -- An estimator class or pipeline from sklearn
    distributions -- The parameter distributions to search for that estimator
    scorer        -- Scoring function (e.g. accuracy or f1)
    n_iter        -- The number of random iterations to try
    """
    # Make a pipeline out of the classifier, to allow for feature scaling in the first step.

    # Add prefix to parameters to support use in pipeline
    class_name = classifier.__class__.__name__.lower()
    distributions = dict((class_name + "__" + key, val) for key, val in distributions.iteritems())

    # It is important to handle scaling here so we don't accidentally overfit some to the
    # test data by scaling using that information as well.
    classifier = make_pipeline(preprocessing.RobustScaler(), classifier)
    randomized_search = RandomizedSearchCV(
        classifier, param_distributions=distributions, n_iter=n_iter, scoring=scorer, cv=cv, n_jobs=1)
    randomized_search.fit(X, y)

    print randomized_search.best_estimator_
    print "Validation Score ({}): {:.2f}".format(scorer, randomized_search.best_score_)
    print ""
    return randomized_search.best_estimator_, randomized_search.best_score_


def main():
    df = pd.read_pickle('data/133156838395276.pkl')
    X = df.iloc[:,4:184].astype(float)
    X[np.isnan(X)]=0.0
    y = df.label.values.astype(int)
    print X.shape
    #X = np.random.normal(size=(1000,10))#np.asarray(df['feature_vector'])
    #y = np.random.choice(range(10),size=1000) #np.asarray(df['label'])

    # Convert to numpy arrays to use with learning algorithm
    X_train, X_test, y_train, y_test = train_test_split(X,y)# convert to numpy array and train/test split it

    # Now we test out a few algorithms
    # The goal is to select which algorithm will do best given reasonable
    # optimization, then we will do a more careful job in train.py
    algorithms = []

    # Throwing in a KNN first to provide a fast-to-calculate reference
    # Don't particularly expect it to be competitive.
    # knn_params = {'n_neighbors': np.logspace(.5, 2, 10).astype(int).tolist(),
    #               'weights': ['uniform', 'distance']}
    # algorithms.append(optimized_classifier(X_train, y_train, neighbors.KNeighborsClassifier(), knn_params, n_iter=20))
    # algorithms[-1][0].fit(X_train, y_train)

    n_examples, n_features = X_train.shape
    rfc_params = {"n_estimators": [100],
                  "criterion": ["gini", "entropy"],
                  "max_features": np.linspace(np.sqrt(n_features) / 2, np.sqrt(n_features) * 2, 5).astype(int),
                  "max_depth": range(3, 7),
                  "min_samples_split": np.linspace(2, n_examples / 50, 10).astype(int)}
    algorithms.append(optimized_classifier(X_train, y_train, ensemble.RandomForestClassifier(), rfc_params))

    gbc_params = {"n_estimators": [100],
                  "max_features": np.linspace(np.sqrt(n_features) / 2, np.sqrt(n_features) * 2, 5).astype(int),
                  "max_depth": range(3, 7),
                  "min_samples_split": np.linspace(2, n_examples / 50, 10).astype(int),
                  "learning_rate": np.linspace(0.1, 0.5, 5)}
    algorithms.append(optimized_classifier(X_train, y_train, ensemble.GradientBoostingClassifier(), gbc_params))

    # Decided to use LinearSVC here because it is the only one that runs quickly enough to
    # test on my laptop. Given access to more resource it would be reasonable to try a poly or rbf kernel.
    # svc_params = {'C': scipy.stats.expon(scale=100),
    #               'class_weight': ['balanced', None]}
    # algorithms.append(optimized_classifier(X_train, y_train, svm.LinearSVC(), svc_params))

    # Train the best of them on the full training set and test on the test set
    print "------------------------------------------------------------"
    print " Best Option"
    print "------------------------------------------------------------"

    classifier = max(algorithms, key=lambda x: x[1])[0]

    classifier.fit(X_train, y_train)

    print classifier
    print ""
    print "Results on Test Data"
    print "{:>11}: {:.2f}".format("f1", f1_score(y_test, classifier.predict(X_test)))
    print "{:>11}: {:.2f}".format("Accuracy", classifier.score(X_test, y_test))
    print ""
    print "Confusion Matrix:"
    cm = confusion_matrix(y_test, classifier.predict(X_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print cm

if __name__ == "__main__":
    main()
