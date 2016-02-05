"""
Handles import of Piezo files from mat format,
and preprocesses them to be mean zero and of
specified sampling rate (default 120)

Written by Ryan Gooch, Feb 2016
"""
import pandas as pd
import numpy as np 
from scipy.io import loadmat
from scipy.signal import resample

# Not pretty. Need to fix
# Right now I'm using this since no good
# edf reader seems to exist in python,
# and since most of the code for this problem
# lives in matlab anyway. Workable solution for now
octave.addpath(octave.genpath('./'))

class PiezoReader :
	"""
	Reads piezo data from EDF and returns Piezo 
	signal
	"""

	def __init__(self, fs = 120) :
		# Set our sampling rate
		self.fs = fs

	def openmat (self, filepath, filename) :
		"""
		Open the edf for reading
		"""
		print('Opening EDF')
		self.filename = filename
		self.filepath = filepath
		matdata = loadmat(filepath+filename)
		self.rawpiezo = matdata['Piezo'][0]

		# Go ahead and process it
		print('Processing Piezo Data')
		self.processpiezo()
		
		print('Removing segments with scorer disagreement')
		self.labelspiezo()
		print('Done!')
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
		lbls1name = self.filename.strip('.edf') + '.xls'
		lbls2name = self.filename.strip('.edf') + '_2.xls'

		# Import scores as dataframes
		lbls1 = pd.read_excel(self.filepath+lbls1name, header = None)
		lbls2 = pd.read_excel(self.filepath+lbls2name, header = None)

		# Concatenate into same dataframe and keep segments where equal
		concatted = pd.concat([lbls1[0],lbls2[0]])
		concatted.columns = ['scorer1','scorer2']
		scoredf = concatted[concatted['scorer1']==concatted['scorer2']]

		# scoredf is a dataframe with indices corresponding to the piezo
		# segments where there is agreement, and the identical labels in
		# each column

		# first reshape the piezo
		npr = np.reshape(self.piezo,[len(piezo)/(self.fs*4),self.fs*4])

		# this single function slices the reshaped piezo matrix such that
		# it retains only segments where doublescored
		self.piezomat = npr[scoredf.index]
		self.labels = scoredf['scorer1']
