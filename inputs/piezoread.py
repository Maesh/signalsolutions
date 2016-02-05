"""
Handles import of Piezo files from mat format,
and preprocesses them to be mean zero and of
specified sampling rate (default 120)

Written by Ryan Gooch, Feb 2016
"""
import pandas as pd
import numpy as np 
import scipy.io as sio
from scipy.signal import resample

class PiezoReader :
	"""
	Reads piezo data from EDF and returns Piezo 
	signal
	"""

	def __init__(self, fs = 120) :
		# Set our sampling rate
		self.fs = fs

	def openmat (self, filepath, filename, savemat = False) :
		"""
		Open the edf for reading
		"""
		# Need to fake header. If we refactor this to actually
		# open the EDFs, then the header will be given
		# For now, we know the relevant values

		self.header = {}
		self.header['duration'] = 10
		self.header['samples'] = np.array([[4000,4000,4000,4000,400]])
		# EEG1, EEG2, EMG, Piezo, EDF Annotations
		print('Opening mat file')
		self.filename = filename
		self.filepath = filepath
		matdata = sio.loadmat(filepath+filename)
		self.rawpiezo = matdata['Piezo'][0]

		# Go ahead and process it
		print('Processing Piezo Data')
		self.processpiezo()
		
		print('Removing segments with scorer disagreement')
		self.labelspiezo()
		print('Done!')

		# If flagged, save the matrix
		if savemat == True :
			matdict = {}
			matdict['piezomat'] = self.piezomat
			matdict['labels'] = self.labels
			matdict['fs'] = self.fs
			savename = 'MiceData/' + filename.strip('.mat') + '.2scored.signal.matrix'
			# Using format '4' across the board to ensure files can be opened and
			# closed. 
			sio.savemat(savename, matdict, appendmat = True, format = '4')

		return self.piezomat, self.labels

		

	def processpiezo (self) :
		"""
		Processes the piezo file by resampling,
		removing bias, and aligning signal with
		segments that were scored by both observers
		"""
		#Sampling rate for edf
		fsold = self.header['samples'][0][3] / self.header['duration']

		# Resample to desired sampling rate, fs
		self.piezo = resample(self.rawpiezo, len(self.rawpiezo) * self.fs / fsold)
		self.piezo = self.piezo - np.mean(self.piezo)

	def labelspiezo(self) :
		"""
		Uses two sets of labels from EEG scorers to keep
		only those segments corresponding to agreedupon 
		scores 
		"""
		# First get file names
		lbls1name = self.filename.strip('.mat') + '.xls'
		lbls2name = self.filename.strip('.mat') + '_2.xls'

		# Import scores as dataframes
		lbls1 = pd.read_excel(self.filepath+lbls1name, header = None)
		lbls2 = pd.read_excel(self.filepath+lbls2name, header = None)

		# Concatenate into same dataframe and keep segments where equal
		concatted = pd.concat([lbls1[0],lbls2[0]],1)
		concatted.columns = ['scorer1','scorer2']
		scoredf = concatted[concatted['scorer1']==concatted['scorer2']]

		# scoredf is a dataframe with indices corresponding to the piezo
		# segments where there is agreement, and the identical labels in
		# each column

		# first reshape the piezo
		npr = np.reshape(self.piezo,[len(self.piezo)/(self.fs*4),self.fs*4])

		# this single function slices the reshaped piezo matrix such that
		# it retains only segments where doublescored
		self.piezomat = npr[scoredf.index]
		# as_matrix ensures indices are not saved since we need only labels
		self.labels = scoredf['scorer1'].as_matrix()
