import numpy as np 
import scipy.io as sio
import scipy.signal
import sklearn
from sklearn import preprocessing, cross_validation, metrics, svm
import matplotlib.pyplot as plt
import miceFuncs as mF
import csv

mats = ['UK5_2scored_reshaped.mat',
		'UK14_2scored_reshaped.mat',
		'UK17_2scored_reshaped.mat',
		'UK21_2scored_reshaped.mat',
		'UK22_2scored_reshaped.mat',
		'UK24_2scored_reshaped.mat',
		'UK26_2scored_reshaped.mat',
		'UK27_2scored_reshaped.mat',
		'UK29_2scored_reshaped.mat',
		'UK30_2scored_reshaped.mat',
		'UK31_2scored_reshaped.mat',
		'UK33_2scored_reshaped.mat',
		'UK34_2scored_reshaped.mat',
		'UK39_2scored_reshaped.mat']

# Get data
i = 4
mat = mats[i]

# load data
data = sio.loadmat(mat)
fs = int(data['fs'][0])

# Piezo data array
piezo = data['Piezo'][0].copy()

# Reshape into matrix. Columns represent 4 second samples
pzr = np.reshape(piezo,(fs*4,len(piezo)/(fs*4)),'F')

# Scale Data
pzr_sc = preprocessing.scale(pzr)
# means = np.apply_along_axis(some_func,0,pr)

### Compute features
# Power Spectrum Features

# Power Spectrum
# F = frequencies, Pxx = matrix of FFTs
# Pxx = (480,fs/2+1)
F,Pxx = scipy.signal.welch(pzr_sc.T,fs=fs,
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

# Build feature matrix
# Want to test if contextual feature info makes a difference
# So build feature matrix with 3x number of columns to features
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

# now let's see if this works
rs = 2727
X = feats.copy()
X_train, X_test, y_train, y_test = \
	cross_validation.train_test_split(X, y, test_size=0.4, random_state=rs)

# # Linear SVM first, C = 1
# classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# classifier.score(X_test, y_test)
# # 89.988% classifier score, not bad. This is three class problem

# # Can't do this in multiclass
# # cv = cross_validation.StratifiedKFold(y_test, 5, shuffle=True, random_state=rs)
# # print cross_validation.cross_val_score(classifier, \
# # 	X_test, y_test, cv=cv, scoring='roc_auc')

# # Generate confusion matrix
# y_pred = classifier.predict(X_test)
# cm = metrics.confusion_matrix(y_test,y_pred)
# np.set_printoptions(precision=2)
# print 'Confusion Matrix'
# print(cm)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print 'Normalized Confusion matrix'
# print(cm_normalized)
# # 			Predicted label
# #				Wake	NREM	REM
# #		Wake	96.6%	3.43%	0.00%
# #
# # True	NREM	10.2%	89.8%	0.00%
# #
# #		REM		46.2%	50.5%	3.38%
# #
# # Pretty terrible score on the REM here

# # Increase C?
# classifier = svm.SVC(kernel='linear', C=100, degree=2).fit(X_train, y_train)
# classifier.score(X_test, y_test)
# y_pred = classifier.predict(X_test)
# cm = metrics.confusion_matrix(y_test,y_pred)
# np.set_printoptions(precision=2)
# print 'Confusion Matrix'
# print(cm)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print 'Normalized Confusion matrix'
# print(cm_normalized)

# # Class weight?
# classifier = svm.SVC(kernel='linear', C=100, class_weight='auto').fit(X_train, y_train)
# classifier.score(X_test, y_test)
# y_pred = classifier.predict(X_test)
# cm = metrics.confusion_matrix(y_test,y_pred)
# np.set_printoptions(precision=2)
# print 'Confusion Matrix'
# print(cm)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print 'Normalized Confusion matrix'
# print(cm_normalized)

# 			Predicted label
#				Wake	NREM	REM
#		Wake	83%		1%		16%
#
# True	NREM	7%		70%		23%
#
#		REM		8%		11%		81%
#
# HOLY FUCK BOYS THIS IS AMAZING
# What would that REM PPV be?
# Well still terrible. 14.5 %. But still! This feels like progress!
# The automatic weights for REM are probably just too agressive.
# This can probably be tweaked to something better though. I hope...

# Need to continue this. 
#	1. Grid search on the three class problem
#		a. Find optimal class weights C
#	2. Two class problem of classifying REM and NREM
#	3. Gaussian kernel for both

# classifier = svm.SVC(kernel='rbf', \
# 	class_weight='auto').fit(X_train, y_train)
# classifier.score(X_test, y_test)
# y_pred = classifier.predict(X_test)
# cm = metrics.confusion_matrix(y_test,y_pred)
# np.set_printoptions(precision=2)
# print 'Confusion Matrix'
# print(cm)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print 'Normalized Confusion matrix'
# print(cm_normalized)

# # only one gamma produced a decent result, gamma = 10^-3
# scores = []
# for i in np.logspace(-7,3,11) :
# 	classifier = svm.SVC(kernel='rbf', gamma=i).fit(X_train, y_train)
# 	y_pred = classifier.predict(X_test)
# 	cm = metrics.confusion_matrix(y_test,y_pred)
# 	if np.sum(cm[:-1,-1]) > 0 :
# 		scores.append(cm[-1,-1].astype('float')/np.sum(cm[:-1,-1]))
# 	else :
# 		scores.append(0)

##############################
# Try two class problem

slp_train_inxs = np.where(y_train > 1)
slp_test_inxs = np.where(y_test > 1)

X_train_sleep = X_train[slp_train_inxs].copy()
X_test_sleep = X_test[slp_test_inxs].copy()
y_train_sleep = y_train[slp_train_inxs].copy()
y_test_sleep = y_test[slp_test_inxs].copy()

# Class weight?
cw = dict()
cw[2] = .3
cw[3] = .7
classifier = svm.SVC(kernel='linear',\
	class_weight=cw).fit(X_train_sleep, y_train_sleep)
classifier.score(X_test_sleep, y_test_sleep)
y_pred = classifier.predict(X_test_sleep)
cm = metrics.confusion_matrix(y_test_sleep,y_pred)
np.set_printoptions(precision=2)
print 'Confusion Matrix'
print(cm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print 'Normalized Confusion matrix'
print(cm_normalized)

# 				Predicted label
#				NREM	REM
#
#
# True	NREM	7%		70%		
#
# 		REM		8%		11%		
#

