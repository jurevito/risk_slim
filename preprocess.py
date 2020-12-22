import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

def binarize_limits(feature_name, train_df, test_df, limits):

	train_data = train_df[feature_name].to_numpy()
	test_data = test_df[feature_name].to_numpy()

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
			bin_feature = ("%s >= %f" % (feature_name, limit)).rstrip('0').rstrip('.')
			train_df.insert(index, bin_feature, subfeature_train, True)
			test_df.insert(index, bin_feature, subfeature_test, True)
			subfeatures_names.append(bin_feature)
		index+=1

	return train_df, test_df, subfeatures_names

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

def find_treshold_index(tresholds, my_treshold):

    index = 0
    smallest_d = 1
    for i,t in enumerate(tresholds):

        d = abs(my_treshold - t)

        if d < smallest_d:
            smallest_d = d
            index = i

    return index

def stump_selection(C, train_df):

	X_labels = train_df.columns[1:]
	y_label = train_df.columns[0]
	X = train_df[X_labels]
	y = train_df[y_label]

	selector = SelectFromModel(LogisticRegression(solver='liblinear', C=C, penalty='l1', random_state=0))
	selector.fit(X, y)
	selected_features = list(X_labels[selector.get_support()])
	selected_features.insert(0, y_label)
	removed_features = np.setdiff1d(X_labels, selected_features)
	print("Removed stumps (%d - %d = %d):\n" % (len(X_labels),len(removed_features), len(X_labels) - len(removed_features)))

	return selected_features

def fix_names(names, selected_features):
	return list(set(names) & set(selected_features))


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
