"""
Author: Corey Lynch
Date: 4/27/12
"""
import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.base import clone
from time import time
import matplotlib.pyplot as plt
#from botos3 import write_to_s3
import cPickle as pickle

# Define an error function here
def rmsle(preds,actuals):
    return np.sum((np.log(np.array(preds)+1) - np.log(np.array(actuals)+1))**2 / len(preds))

cost = rmsle

class ErrorCurves():
  def __init__(self,clf,X,y,model_name,n_runs=5,train_fraction_range=np.linspace(0.1, 0.9, 10),test_fraction=0.1):
    self.clf = clf
    self.X = X
    self.y = y
    self.model_name=model_name
    self.n_runs = n_runs
    self.train_fraction_range = train_fraction_range
    self.test_fraction=test_fraction

  def plot_and_save(self):
    n_datasets = self.train_fraction_range.shape[0]
    training_score = np.zeros((n_datasets, self.n_runs))
    test_score = np.zeros((n_datasets, self.n_runs))
    training_time = np.zeros((n_datasets, self.n_runs))

    n_samples = self.y.shape[0]
    n_features = self.X.shape[1]

    for i, train_fraction in enumerate(self.train_fraction_range):
        print "Train fraction: %0.2f" % train_fraction

        cv = ShuffleSplit(n_samples, n_iterations=self.n_runs, test_fraction=self.test_fraction, train_fraction=train_fraction)
        for j, (train, test) in enumerate(cv):
            cloned = clone(self.clf)
            t0 = time()
            cloned.fit(self.X[train], self.y[train])
            training_time[i, j] = time() - t0
            training_score[i, j] = cost(self.y[train],cloned.predict(self.X[train]))
            test_score[i, j] = cost(self.y[test],cloned.predict(self.X[test]))
    
    train_fraction_range = self.train_fraction_range
    mean_test_score = test_score.mean(axis=1)
    mean_train_score = training_score.mean(axis=1)

    fname = '%s_error_curves.pkl' % self.model_name
    with open(fname, 'wb') as f:
        pickle.dump(plot_vecs,f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(self.train_fraction_range,mean_train_score,'b',self.train_fraction_range,mean_test_score,'g')
    leg = ax.legend(('Train','Test'),'lower right')
    ax.set_xlabel('Fraction of the dataset used for training')
    ax.set_ylabel('Prediction accuracy') 
    ax.set_title('Learning Curves')
    #plt.show()
    # Save fig
    fname = '%s_error_curves.png' % self.model_name
    fig.savefig(fname)
