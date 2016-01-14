"""
Implements SVM of the user's design and specification
on the mice data. Imports training and test sets, and creates a 
validation set from the training data.

Created by: Ryan Gooch, January 13, 2016
"""
import numpy as np 

from inputs.miceinputs import miceloader
from sklearn import cross_validation, preprocessing, metrics
from micefuncs.miceFuncs import onehotcoder, spliteven, evenup

from sklearn import svm, cross_validation, preprocessing, metrics, decomposition
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
	rs = 2016 # random state for consistency

	# Usage is miceloader(trainmice = 14,random_state = 2016,
	#	datadir = 'MiceData/Feats File/')
	
	ML = miceloader(random_state = rs)

	X_train, X_test, y_train, y_test = ML.getdata()

	# Clean data, ensure no infs
	X_train[X_train == np.inf] = 0
	X_test[X_test == np.inf] = 0


	# Scale data
	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)

	# Create validation set using sklearn. This will also shuffle
	# features in training and test sets
	X_train, X_val, y_val, y_val = \
		cross_validation.train_test_split(X_train, y_train, \
			test_size=0.2, random_state=rs)