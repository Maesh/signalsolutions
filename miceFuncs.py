import numpy as np 
import scipy.io as sio
import scipy.signal
import sklearn
import matplotlib.pyplot as plt

#### Functions for mice classification ####
# Ryan Gooch, October 9 2015 #

# index of max value
def index_max(a) :
	"""
	Finds the index of a max value for a vector
	"""
	return np.where(a == np.max(a))

def log_ratio(a) :
	"""
	Finds the log ratio of peak in sleep range to 
	max global range
	"""
	inpk = np.max(a[2:5])
	globpk = np.max(a)
	return np.log(inpk/globpk)

def loc_peaks(a) :
	"""
	Finds the local maxima in an input signal
	"""
	pklocs = scipy.signal.argrelmax(a)
	return np.std(np.diff(pklocs))

def pct_over(a) :
	"""
	Finds percent of AC over threshold of 0.1
	"""

def norm_AC(a) :
	"""
	Normalize signal to max value = 1
	"""
	return a/np.max(a)

def calc_AC(a) :
	"""
	Calculates the autocorrelation of a signal
	"""
	ac = scipy.signal.correlate(a,a,'full')
	return norm_AC(ac)

def dpl(a) :
	pks = scipy.signal.argrelmax(a)
	diff_pks_locs = np.sum(np.abs(np.diff(pks,n=1)))
	pkdns = np.sum(np.abs(np.diff(a[pks],n=1)))
	return diff_pks_locs, pkdns

def pkdns(a) :
	return np.sum(np.abs(np.diff(a,n=1)))

def ln_max_std(a) :
	return np.log(np.max(a)/np.std(a))