'''
Plotting tools for gps/accel data visualizations.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from sklearn.learning_curve import learning_curve

def errorband(x,y,yerr,ax=None, **kwargs):
    '''
    Plot x vs y with a shaded band to represent yerr.

    Kwargs:
    ax    -- matplotlib.Axes instance to draw on

    Remaining args are passed to call to matplotlib.pyplot.plot
    '''
    if ax is None:
        ax = plt.gca()
    line = ax.plot(x,y, **kwargs)[0]
    band = ax.fill_between(x, y-yerr, y+yerr, facecolor=line.get_color(), alpha=0.5)


def learning_curve_band(estimator, X, y, ax=None, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1', n_jobs=3, cv=10):
    """
    Plot the training and testing score vs sample size for estimator
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, n_jobs=n_jobs, train_sizes=train_sizes, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)/np.sqrt(cv)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)/np.sqrt(cv)
    line1 = ax.plot(train_sizes, train_scores_mean, 'o-', label="Training")[0]
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.4, facecolor=line1.get_color())
    line2 = ax.plot(train_sizes, test_scores_mean, 'o-',label="Cross Validation")[0]
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.4, facecolor=line2.get_color())
    return train_scores_mean, test_scores_mean

def main():
    # Run a few examples

    # Simple Data
    x = np.linspace(0,4*np.pi,30)
    y = np.sin(x) + 0.05*np.random.randn(30)
    yerr = 0.25*np.random.rand(30) + 0.1
    
    fig, ax = plt.subplots()
    errorband(x,y,yerr,ax=ax, label='Example')
    plt.legend()
    plt.show()

    
    
