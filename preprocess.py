import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def binarize_greater(feature_name, k, df, base):

	data = df[feature_name].to_numpy()

	# generate limits
	max_value = base * round(np.amax(data)/base)
	min_value = base * round(np.amin(data)/base)
	step = base * round(int((max_value - min_value) * k)/base)
	limits = list(range(min_value, max_value, step))

	# limits per feature
	str1 = ','.join(str(e) for e in limits)
	print("%s_limits = %s" % (feature_name, str1))

	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)
	subfeatures_names = []
	
	for limit in limits:

		subfeature = np.array(data)
		subfeature[subfeature <= limit] = 0
		subfeature[subfeature > limit] = 1

		df.insert(index, "%s > %d" % (feature_name, limit), subfeature, True)
		subfeatures_names.append("%s > %d" % (feature_name, limit))
		index+=1

	return df, subfeatures_names, limits

def binarize_manual(feature_name, df, min_value, max_value, step):

	data = df[feature_name].to_numpy()

	# generate limits
	limits = list(range(min_value, max_value, step))

	# limits per feature
	str1 = ','.join(str(e) for e in limits)
	print("%s_limits = %s" % (feature_name, str1))

	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)
	subfeatures_names = []
	
	for limit in limits:

		subfeature = np.array(data)
		subfeature[subfeature <= limit] = 0
		subfeature[subfeature > limit] = 1

		df.insert(index, "%s > %d" % (feature_name, limit), subfeature, True)
		subfeatures_names.append("%s > %d" % (feature_name, limit))
		index+=1

	return df, subfeatures_names, limits

def binarize_limits(feature_name, train_df, test_df, limits):

	train_data = train_df[feature_name].to_numpy()
	test_data = test_df[feature_name].to_numpy()

	# limits per feature
	str1 = ','.join(str(e) for e in limits)
	print("%s_limits = %s" % (feature_name, str1))

	index = train_df.columns.get_loc(feature_name)
	train_df.drop(feature_name, axis=1, inplace=True)
	test_df.drop(feature_name, axis=1, inplace=True)
	subfeatures_names = []
	
	for limit in limits:

		subfeature_train = np.array(train_data)
		subfeature_test = np.array(test_data)

		subfeature_train[subfeature_train < limit] = 0
		subfeature_train[subfeature_train >= limit] = 1

		subfeature_test[subfeature_test < limit] = 0
		subfeature_test[subfeature_test >= limit] = 1

		
		if isinstance(limit, int):
			train_df.insert(index, "%s >= %d" % (feature_name, limit), subfeature_train, True)
			test_df.insert(index, "%s >= %d" % (feature_name, limit), subfeature_test, True)
			subfeatures_names.append("%s >= %d" % (feature_name, limit))
		else:
			train_df.insert(index, "%s >= %.4f" % (feature_name, limit), subfeature_train, True)
			test_df.insert(index, "%s >= %.4f" % (feature_name, limit), subfeature_test, True)
			subfeatures_names.append("%s >= %.4f" % (feature_name, limit))
		index+=1

	return train_df, test_df, subfeatures_names, limits

def binarize_interval(feature_name, k, df):

	data = df[feature_name].to_numpy()

	# generate limits
	max_value = np.amax(data)
	min_value = np.amin(data)
	step = int((max_value - min_value) * k)
	limits = list(range(min_value + int(max_value*0.01), max_value - int(max_value*0.01), step))

	# limits per feature
	str1 = ','.join(str(e) for e in limits)
	print("%s_limits(interval) = %s" % (feature_name, str1))

	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)
	subfeatures_names = []

	for i,limit in enumerate(limits):

		subfeature = np.array(data)

		if i == 0:

			# first interval
			subfeature[subfeature <= limit] = 1
			subfeature[subfeature > limit] = 0
			df.insert(index, "%s <= %d" % (feature_name, limit), subfeature, True)
			subfeatures_names.append("%s <= %d" % (feature_name, limit))

		else:

			subfeature[(subfeature <= limits[i-1]) | (subfeature > limit)] = 0
			subfeature[(subfeature <= limit) & (subfeature > limits[i-1])] = 1
			df.insert(index, "%d < %s <= %d" % (limits[i-1], feature_name, limit), subfeature, True)
			subfeatures_names.append("%d < %s <= %d" % (limits[i-1], feature_name, limit))

		index+=1

	# last interval
	subfeature = np.array(data)
	subfeature[subfeature <= limits[-1]] = 0
	subfeature[subfeature > limits[-1]] = 1
	df.insert(index, "%d < %s" % (limits[-1], feature_name), subfeature, True)
	subfeatures_names.append("%d < %s" % (limits[-1], feature_name))

	return df, subfeatures_names

def binarize_category(feature_name, df):

	data = df[feature_name].to_numpy()
	subfeatures = {}
	n = len(data)

	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)
	subfeatures_names = []

	for i,value in enumerate(data):

		if value not in subfeatures.keys():

			subfeatures[value] = np.zeros((n,), dtype=int)
			subfeatures[value][i] = 1

		else:

			subfeatures[value][i] = 1

	# insert subfeatures
	for key in subfeatures.keys():

		df.insert(index, key, subfeatures[key], True)
		subfeatures_names.append(key)
		index+=1

	return df, subfeatures_names

# class1_name are 0, class2_name are 1
def binarize_sex(feature_name, class1_name, class2_name, df):

	data = df[feature_name].to_numpy()

	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)

	df.insert(index, class1_name, 1 - data, True)
	df.insert(index+1, class2_name, data, True)

	return df, [class1_name, class2_name]

def binarize(feature_name, limits, df):

	data = df[feature_name].to_numpy()
	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)

	for limit in limits:

		subfeature = np.array(data)
		subfeature[subfeature <= limit] = 0
		subfeature[subfeature > limit] = 1

		if isinstance(limit, int):
			df.insert(index, "%s >= %d" % (feature_name, limit), subfeature, True)
		else:
			df.insert(index, "%s >= %.4f" % (feature_name, limit), subfeature, True)
		index+=1

	return df

def sec2time(seconds):

	hours = int(seconds/3600)
	minutes = int((seconds%3600)/60)
	secs = int(seconds%60)

	return '%dh %dmin %dsec' % (hours, minutes, secs)

def riskslim_cv(n_fold, rm, X, y):

	cv = StratifiedKFold(n_splits=n_fold)
	classifier = rm

	accs = []
	build_times = []
	optimality_gaps = []

	for i, (train, test) in enumerate(cv.split(X, y)):

		classifier.threshold = 0.5
		classifier.fit(X[train], y[train])
		y_pred = classifier.predict(X[test])
		accs.append(accuracy_score(y[test], y_pred))
		build_times.append(classifier.model_info['solver_time'])
		optimality_gaps.append(classifier.model_info['optimality_gap'])

	accs = np.array(accs)
	build_times = np.array(build_times)
	optimality_gaps = np.array(optimality_gaps)

	return accs, build_times, optimality_gaps

if __name__ == "__main__":

	os.chdir('..')
	path = os.getcwd() + '/risk-slim/examples/data/' + 'heart.csv'

	df  = pd.read_csv(path, float_precision='round_trip')
	X = df.iloc[:, 0:-1].values
	y = df.iloc[:,-1].values
	y[y == -1] = 0

	# binarizing features
	df, sex_features = binarize_sex('sex', 'female', 'male', df)

	# moving target to beginning
	df.drop('target', axis=1, inplace=True)
	df.insert(0, "target", y, True)

	# saving processed data
	df.to_csv('risk_slim/hrt.csv', sep=',', index=False,header=True)
