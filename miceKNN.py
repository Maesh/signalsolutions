"""
Implements KNeighbors of the user's design and specification
on the mice data. Imports training and test sets, and creates a 
validation set from the training data.

Created by: Ryan Gooch, Feb, 2016
"""
import numpy as np 

from inputs.readAAFeats import ReadFeats
from sklearn import cross_validation, preprocessing, metrics, neighbors
from micefuncs.miceFuncs import onehotcoder, spliteven, evenup

from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
	rs = 2016 # random state for consistency

	# Get mice data using custom class
	# Usage is miceloader(trainmice = 14,random_state = 2016,
	#	datadir = 'MiceData/Feats File/')
	r = ReadFeats(filepath = 'MiceData/FeatsFiles/')
	X, y = r.getdata()

	# Clean data, ensure no infs
	X[X == np.inf] = 0

	# X = contextfeats(X, time_step = 2)

	# Scale data
	X = preprocessing.scale(X)

	# Create validation set using sklearn. This will also shuffle
	# features in training and test sets
	X_train, X_test, y_train, y_test = \
		cross_validation.train_test_split(X, y, \
			test_size=0.2, random_state=rs)

	X_train, X_val, y_train, y_val = \
		cross_validation.train_test_split(X_train, y_train, \
			test_size=0.25, random_state=rs)

	# Train the classifier and fit to training data
	# Grid search for RBF
	# Cs = np.logspace(0, 4, 5)
	# gammas = np.logspace(-2,0,3)
	# classifier = GridSearchCV(estimator=svm.SVC(), \
	# 	param_grid=dict(class_weight=['balanced'],kernel=['linear']),
	# 	verbose=3,scoring='accuracy',n_jobs=-1)

	# classifier.fit(X_train, y_train)
	# print classifier.best_score_
	# print classifier.best_estimator_

	classifier = neighbors.KNeighborsClassifier(n_jobs = -1,
		algorithm = 'auto', weights = 'distance', n_neighbors = 10)
	classifier.fit(X_train,y_train)
	pred = classifier.predict(X_val)

	print(metrics.classification_report(y_val,pred))
	print('Confusion Matrix')
	print(metrics.confusion_matrix(y_val,pred))
	print('Accuracy is ' + metrics.accuracy_score(y_val,pred).astype(str))