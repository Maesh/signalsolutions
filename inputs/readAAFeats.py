from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io as sio

class ReadFeats:
	"""
	Class to read feature matrices generated by Anuj 
	and return NumPy array dataset
	where X = data and y = labels, the format required by scikit-learn

	Written by Ryan Gooch, Feb 2016
	"""
	def __init__(self, filepath='MiceData/FeatsFiles/',rs = 19683):
		# Does nothing right now, might move shuffling in here later
		self.rs = rs 
		self.filepath = filepath

	def getallfeatsfiles(self) :
		"""
		Get all feats files in path
		"""
		self.files = [f for f in listdir(self.filepath) if isfile(join(self.filepath, f))]

	def combinefeatsmats(self) :
		"""
		Uses all files found in getallfeatsfiles and combines the data in them
		into randomized NumPy array
		"""
		self.getallfeatsfiles()
		# initialize with appropriate number of features
		piezomat = np.empty((0,sio.loadmat(self.filepath+self.files[0])['nrfv1'].shape[1]))
		labels = np.empty((1,0))
		for f in self.files :
			data = sio.loadmat(self.filepath+f)
			piezomat = np.concatenate([piezomat,data['nrfv1']])
			labels = np.concatenate([labels,data['newlbl']],1)
		
		# Return X and y
		return piezomat, labels[0]