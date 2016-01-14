"""
Implements a Neural Network of the user's design and specification
on the mice data. Imports training and test sets, and creates a 
validation set from the training data.

Created by: Ryan Gooch, January 13, 2016
"""
import numpy as np 

from inputs.miceinputs import miceloader

if __name__ == '__main__':
	# Usage is miceloader(trainmice = 14,random_state = 2016,
	#	datadir = 'MiceData/Feats File/')
	ML = miceloader()

	X_train, X_test, y_train, y_test = ML.getdata()