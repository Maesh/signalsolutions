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

def onehotcoder(a) :
	res = []
	for i in range(len(a)) :
		if a[i] == 1 :
			res.append([1,0,0])
		elif a[i] == 2 :
			res.append([0,1,0])
		elif a[i] == 3 :
			res.append([0,0,1])
	return np.array(res)

def spliteven(X, y, bootstrap = False, size = 0.67) :
	"""
	Evenly splits data so that each class is of
	equal frequency. Expectation is that y are 
	labels and X is mice data. for y: 1 is wake,
	2 is nrem, 3 is rem, so the limiting factor 
	is rem. This would be another candidate for
	refactoring, but for now it seems fine.

	bootstrap is boolean and if True, samples arrays
	with replacement. size is a value in range (0,1]
	that determines percentage of smallest behavior
	count to be kept. ie, if there are 30000 rem 
	segments, then size = 0.67 would mean 20000 behavior
	segments for each behavior are returned
	"""
	# Number of rem, wake, nrem segments
	remcount = y[y==3].shape[0]
	nremcount = y[y==2].shape[0]
	wakecount = y[y==1].shape[0]

	# number of segments in each class to keep
	segmentcount = int(remcount * size)

	# perform sampling by determining indices
	if bootstrap == True :
		reminx = np.random.choice(remcount,
			size = segmentcount, replace = True)
		nreminx = np.random.choice(nremcount,
			size = segmentcount, replace = True)
		wakeinx = np.random.choice(wakecount,
			size = segmentcount, replace = True)
	elif bootstrap == False :
		reminx = np.random.choice(remcount,
			size = segmentcount, replace = False)
		nreminx = np.random.choice(nremcount,
			size = segmentcount, replace = False)
		wakeinx = np.random.choice(wakecount,
			size = segmentcount, replace = False)
	else :
		raise ValueError('bootstrap argument must be either True or False')

	# New array for returned values
	newX = np.empty((segmentcount*3,X.shape[1]))
	newy = np.empty((segmentcount*3,))
	# drop in kept behavior segments
	newX[:segmentcount,:] 				= X[y[y==3]][reminx]
	newX[segmentcount:2*segmentcount,:]	= X[y[y==2]][nreminx]
	newX[2*segmentcount:,:] 			= X[y[y==1]][wakeinx]

	newy[:segmentcount] 				= y[y==3][reminx]
	newy[segmentcount:2*segmentcount]	= y[y==2][nreminx]
	newy[2*segmentcount:] 				= y[y==1][wakeinx]
	# return the new matrices
	return newX.astype(int), newy.astype(int) # make sure they're int

def evenup(X,y) :
	"""
	bootstrap resamples the minority classes in X to 
	provide equal representation in training
	"""
	# Number of rem, wake, nrem segments
	remcount = len(y[y==3])
	nremcount = len(y[y==2])
	wakecount = len(y[y==1])

	# the max of the three above will stay the same, the
	# rest of the classes will be augmented with 
	# bootstrap resamples
	segmentcount = np.max([remcount,nremcount,wakecount])
	# new training matrices
	newX = np.empty((segmentcount*3,X.shape[1]))
	newy = np.empty((segmentcount*3,))

	# select bootstrap sample indices
	reminx = np.random.choice(remcount,
			size = segmentcount, replace = True)
	nreminx = np.random.choice(nremcount,
		size = segmentcount, replace = True)
	wakeinx = np.random.choice(wakecount,
		size = segmentcount, replace = True)

	newX[:segmentcount] 				= X[y[y==3]][reminx]
	newX[segmentcount:2*segmentcount,:]	= X[y[y==2]][nreminx]
	newX[2*segmentcount:,:] 			= X[y[y==1]][wakeinx]

	newy[:segmentcount] 				= y[y==3][reminx]
	newy[segmentcount:2*segmentcount]	= y[y==2][nreminx]
	newy[2*segmentcount:] 				= y[y==1][wakeinx]
	# return the new matrices
	return newX.astype(int), newy.astype(int) # make sure they're int

def contextfeats(X, time_step = 2) :
	"""
	Returns a feature matrix containing contextual features
	for every time segment by expanding the dimensionality 
	of the feature matrix and concatenating previous time steps

	time_step argument allows user to define length of recursive
	feature inclusion
	"""

	# Build feature matrix
	# Want to test if contextual feature info makes a difference
	# So build feature matrix with 3x number of columns to features
	newX = np.zeros((X.shape[0],(time_step + 1)*X.shape[1]))
	newX[:,:X.shape[1]] = X
	
	for i in range(time_step - 1) :
		newX[(i + 1):, (i+1) * X.shape[1] : (i+2) * X.shape[1] ] = X[:-(i+1)]

	newX[time_step:,time_step * X.shape[1]:] = X[:-time_step]
	return newX