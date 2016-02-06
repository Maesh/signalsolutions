"""
Takes a list of mice matrices and computes features.
The matrices are expected to be N x n, where N is the
number of epochs and n is the number of samples per 
epoch. The output is N x m, where m is the number of
features. 

May add functionality later to allow custom features
to be added, different features selected, filter
bounds, etc.

Written by Ryan Gooch, Feb 2016
"""

import numpy as np 
import scipy.io as sio
import scipy.signal
import miceFuncs as mF
import csv

class Featcalc :
	"""
	class that computes features given matrix of mice data
	"""
	def __init__(self,scale=True, context = 0):

		self.scale = scale
		if context < 0 :
			
	def openmat (self) :
		"""
		calculates the actual features for a given
		matrix. May refactor at a later date to allow
		bulk calculation but for now, segregate by 
		mouse and aggregate (and randomize) in 
		another script. Matrices must be of type
		UKXX.2scored.signal.matrix.mat for this class
		"""
		# load data. Stored as mat file to make compatible
		# with format used by other researchers. Might
		# refactor this to use a more universal format
		# and write code to change formats
		data = sio.loadmat(mat)
		self.fs = int(data['fs'][0])
		self.labels = data['labels']

		# Reshape into matrix. Columns represent 4 second samples
		pzr = data['piezomat']

		if self.scale == True :
			# Scale Data to mean zero, std dev = 1
			self.pzr_sc = preprocessing.scale(pzr)
		else :
			self.pzr_sc = pzr

	def calcfeats(self, mat) :
		self.openmat(mat)
		### Compute features
		# Power Spectrum Features

		# Power Spectrum
		# F = frequencies, Pxx = matrix of FFTs
		# Pxx = (480,fs/2+1)
		F,Pxx = scipy.signal.welch(self.pzr_sc.T,fs=self.fs,
			window='hamming',nperseg=120,noverlap=60)

		# So that columns are samples
		Pxx = Pxx.T.copy()

		# max_inx is frequency of max value in Spectrum
		# log_ratio is ratio of sleep pk to global pk

		max_inx = np.apply_along_axis(mF.index_max,0,Pxx)[0]
		log_ratio = np.apply_along_axis(mF.log_ratio,0,Pxx)

		# Autocorrelation
		pzac = np.apply_along_axis(mF.calc_AC,0,pzr_sc)

		# normalize AC to max lag = 1
		pzac = np.apply_along_axis(mF.norm_AC,0,pzac)

		# std_ac is the std dev of the AC
		# ac_pks_diff is the first derivative of the peak locations in AC

		std_ac = np.std(pzac,0)
		ac_pks_diff = np.apply_along_axis(mF.loc_peaks,0,pzac)

		# diff_pks_locs is the sum of the first derivative of peaks locations
		# pks_hts is the sum of the abs value of first derivative of peak
		#	heights

		diff_pks_locs, diff_pks_hts = np.apply_along_axis(mF.dpl,0,pzr_sc)

		# pkdns is the sum of the abs value of first derivative of the signal

		pkdns = np.apply_along_axis(mF.pkdns,0,pzr_sc)

		# log ratio of max value in signal to std dev. For scaled data should
		#	be just the log ratio of max value since std dev = 1

		ln_max_std = np.apply_along_axis(mF.ln_max_std,0,pzr_sc)


	def getlabels(self) :
		# Get the labels
		y = np.empty(len(std_ac)+1)
		iteration = 0
		csvfile = open('UK22.csv','rb')
		csvfile2 = open('UK22_2.csv','rb')
		rdr = csv.reader(csvfile,delimiter=',')
		rdr2 = csv.reader(csvfile2,delimiter=',')
		# Only pick the labels that are agreed upon by both
		# scorers
		for row, row2 in zip(rdr,rdr2) :
			if row[0] == row2[0] :
				y[iteration] = np.float(row[0])
				iteration += 1
		print iteration

		# Data dimension doesn't match label dimension. I need
		# to address this later and import the data with python,
		# but for now this "hack" isn't too egregious
		y = y[:-1].copy()

	def contextfeats(self, n=2) :
		"""
		Incorporate contextual features by adding
		to each sample feature values of the past
		n time steps
		"""
		feats = np.empty((len(std_ac),18))

		feats[:,0] = std_ac
		feats[:,1] = ac_pks_diff
		feats[:,2] = diff_pks_locs
		feats[:,3] = diff_pks_hts
		feats[:,4] = pkdns
		feats[:,5] = ln_max_std

		# Need to know to contextual features
		feats[1:,6] = std_ac[:-1]
		feats[1:,7] = ac_pks_diff[:-1]
		feats[1:,8] = diff_pks_locs[:-1]
		feats[1:,9] = diff_pks_hts[:-1]
		feats[1:,10] = pkdns[:-1]
		feats[1:,11] = ln_max_std[:-1]

		feats[0,6] = 0
		feats[0,7] = 0
		feats[0,8] = 0
		feats[0,9] = 0
		feats[0,10] = 0
		feats[0,11] = 0

		feats[2:,12] = std_ac[:-2]
		feats[2:,13] = ac_pks_diff[:-2]
		feats[2:,14] = diff_pks_locs[:-2]
		feats[2:,15] = diff_pks_hts[:-2]
		feats[2:,16] = pkdns[:-2]
		feats[2:,17] = ln_max_std[:-2]

		feats[:2,6] = 0
		feats[:2,7] = 0
		feats[:2,8] = 0
		feats[:2,9] = 0
		feats[:2,10] = 0
		feats[:2,11] = 0