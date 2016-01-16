"""
For the time being, this imports the 13 features identified
by Anuj as the best 13. It creates training and test sets
for given parameters. trainmice is the number of mice in 
training set, random_state ensures consistency by forcing
the same shuffling to occur based on a given seed, and
datadir ensures the correct directory is used.

Created by: Ryan Gooch, January 13, 2016
"""

import numpy as np
import numpy.random as npr
import scipy.io as sio

class miceloader :
	def __init__ (self, trainmice = 14, random_state = 2016, data_type = 'feats') :
		"""
		Initializes class. Can add functionality to allow user 
		to input list of mice later if more are added.
		"""
		
		if data_type == 'feats' :
			datadir = 'MiceData/Feats File/'
		elif data_type == 'raw' :
			datadir = 'MiceData//'
		else :
			raise ValueError('data_type argument must be either feats or raw')

		# List of all possible mice
		self.micelist = ['feats_UK5.mat','feats_UK14.mat','feats_UK17.mat',
			'feats_UK18.mat','feats_UK19.mat','feats_UK20.mat','feats_UK21.mat',
			'feats_UK22.mat','feats_UK26.mat','feats_UK27.mat','feats_UK29.mat',
			'feats_UK30.mat','feats_UK31.mat','feats_UK33.mat','feats_UK34.mat',
			'feats_UK35.mat','feats_UK36.mat','feats_UK37.mat','feats_UK38.mat',
			'feats_UK39.mat']
		self.trainmice = trainmice
		self.random_state = random_state
		self.datadir = datadir
		
	def getdata (self) :
		"""
		Compiles training and test sets and produces labels for validation
		"""

		npr.seed(self.random_state)
		permutes = npr.permutation(self.micelist)

		"""
		Mice files have specific format. They are dictionaries, where
		the key nrfv1 is the feature matrix, and newlbl is the labels
		"""

		# training and validation sets mice
		for i in range(self.trainmice) :
			if i == 0 : # first one, initialize training matrix
				tmp = sio.loadmat(self.datadir + permutes[i]) # get first mouse in
				X_train = tmp['nrfv1']
				y_train = tmp['newlbl']
			else :
				tmp = sio.loadmat(self.datadir + permutes[i]) # now get the rest
				X_train = np.vstack((X_train,tmp['nrfv1']))
				y_train = np.hstack((y_train,tmp['newlbl']))

		# testing set mice
		for i in range(self.trainmice, len(self.micelist)) :
			if i == self.trainmice : # first one, initialize training matrix
				tmp = sio.loadmat(self.datadir + permutes[i]) # get first mouse in
				X_test = tmp['nrfv1']
				y_test = tmp['newlbl']
			else :
				tmp = sio.loadmat(self.datadir + permutes[i]) # now get the rest
				X_test = np.vstack((X_test,tmp['nrfv1']))
				y_test = np.hstack((y_test,tmp['newlbl']))


		return X_train, X_test, y_train[0], y_test[0]

		def getrawdata (self) :