"""
Implements a Neural Network of the user's design and specification
on the mice data. Imports training and test sets, and creates a 
validation set from the training data.

Created by: Ryan Gooch, January 13, 2016
"""
import numpy as np 

from inputs.miceinputs import miceloader
from micefuncs.keraswrapper import NN
from sklearn import cross_validation, preprocessing, metrics
from micefuncs.miceFuncs import onehotcoder, spliteven, evenup

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

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

	# clf = NN(inputShape = X_train.shape[1], layers = [64, 64], 
	#     dropout = [0.5, 0.5], loss='rmse', optimizer = 'adadelta', 
	#     init = 'glorot_normal', nb_epochs = 10, validation_split=0,
	#     activation='relu')

	# print('Training model...')
	# clf.fit(X_train,y_train)

	# print('Making predictions on test set...')
	# testpreds = clf.predict(X_test)

	# metrics.classification_report(y_test,testpreds)

	# Fake class equal probability with custom spliteven function
	# X_train, y_train = spliteven(X_train,y_train,bootstrap = False, size = 1)
	# Didn't work, it trained extremely well but it did not perform well at
	# all on validation set

	# We can however augment the classes by bootstrap resampling, with
	# the custom function evenup
	X_train, y_train = evenup(X_train,y_train)



	# neural network handles categorical variables in one hot manner. 
	# since we have 3 classes, we need three neurons, each corresponding
	# to a class. A given class will be fed as a 1 to a given neuron, and 
	# the others, 0. This custom function handles the conversion of 
	# labels to accommodate this.
	y_train_onehot = onehotcoder(y_train)

	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(32, input_dim=X_train.shape[1], init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform',input_dim=8))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, init='uniform',input_dim=64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3, init='uniform',input_dim=32))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	
	# Use mean absolute error as loss function since that is 
	# what kaggle uses
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	# Batch size = 100 seems to have stabilized it
	model.fit(X_train, y_train_onehot, nb_epoch=10, batch_size=128)
	predictions = model.predict(X_val,batch_size=128)

	# Convert resulting predictions back to labels 1, 2, and 3
	# Take max value in preds rows as classification
	pred = np.zeros((len(X_val)))
	# yint = np.zeros((len(X_val)))
	for row in np.arange(0,len(predictions)) :
		pred[row] = np.argmax(predictions[row]) + 1
		# yint[row] = np.argmax(y_test[row])

	print(metrics.classification_report(y_val,pred))
	
