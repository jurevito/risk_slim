import os
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

from interpret.glassbox import ExplainableBoostingClassifier

def ebm_binarization(df_train, df_test, n_features, type='all', feature_names=None):

	headers = df_train.columns[1:].values
	names = []
	feature_dict = {}

	if type == 'all':
		names = headers

	elif type == 'exclude':
		names = list(set(headers) - set(feature_names))

	elif type == 'include':
		names = feature_names

	else:
		print('ERROR: Wrong binarization "Type" value.')


	# train EBM model
	X_labels = df_train.columns[1:]
	y_label = df_train.columns[0]

	X_train = df_train[X_labels]
	y_train = df_train[y_label]

	ebm = ExplainableBoostingClassifier(random_state=0)
	ebm.fit(X_train, y_train)
	ebm_global = ebm.explain_global(name='EBM')

	# search limits for each feature
	for i,feature in enumerate(X_train.columns.values):
		if feature in names:
			
			graph = ebm_global.data(i)
			step = round(len(graph['scores'])/10)+1

			jumps = []
			limits = []

			# find limits
			for k in range(0,len(graph['scores'])-step,round(step/2)):

				#print('k = %d, k+step = %d' % (k, k+step))
				jump_value = abs(graph['scores'][k] - graph['scores'][k+step])
				limit_value = graph['names'][k] if abs(graph['scores'][k]) > abs(graph['scores'][k+step]) else graph['names'][k+step]
				jumps.append(jump_value)
				limits.append(limit_value)

			jumps, limits = zip(*sorted(zip(jumps, limits)))

			# binarize feature
			df_train, df_test, bin_features  = binarize_limits(feature, df_train, df_test, list(limits)[-n_features:])	
			feature_dict[feature] = bin_features


	return df_train, df_test, feature_dict

def auto_selection(max_features, df_train, df_test, feature_dict=None):

	X_labels = df_train.columns[1:]
	y_label = df_train.columns[0]
	X = df_train[X_labels]
	y = df_train[y_label]

	n = 100
	C = 4.0

	while n > max_features:

		selector = SelectFromModel(LogisticRegression(solver='liblinear', C=C, penalty='l1', random_state=0))
		selector.fit(X, y)
		selected_features = list(X_labels[selector.get_support()])
		selected_features.insert(0, y_label)
		removed_features = np.setdiff1d(X_labels, selected_features)

		n = len(selected_features)
		C = C*0.7
	
	
	print("Removed stumps (%d - %d = %d):\n" % (len(X_labels),len(removed_features), len(X_labels) - len(removed_features)))
	df_train = df_train[selected_features]
	df_test = df_test[selected_features]

	# fix operation constraints
	for key in feature_dict.keys():
		feature_dict[key] = fix_names(feature_dict[key], selected_features)

	return df_train, df_test, feature_dict

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

def find_treshold_index(tresholds, my_treshold):

    index = 0
    smallest_d = 1
    for i,t in enumerate(tresholds):

        d = abs(my_treshold - t)

        if d < smallest_d and t > my_treshold:
            smallest_d = d
            index = i

    return index

def stump_selection(C, train_df, weighted):

	X_labels = train_df.columns[1:]
	y_label = train_df.columns[0]
	X = train_df[X_labels]
	y = train_df[y_label]

	if weighted:
		selector = SelectFromModel(LogisticRegression(solver='liblinear', C=C, penalty='l1', random_state=0, class_weight='balanced'))
	else:
		selector = SelectFromModel(LogisticRegression(solver='liblinear', C=C, penalty='l1', random_state=0))

	selector.fit(X, y)
	selected_features = list(X_labels[selector.get_support()])
	selected_features.insert(0, y_label)
	removed_features = np.setdiff1d(X_labels, selected_features)
	print("Removed stumps (%d - %d = %d):\n" % (len(X_labels),len(removed_features), len(X_labels) - len(removed_features)))

	return selected_features

def fix_names(names, selected_features):
	return list(set(names) & set(selected_features))

def print_cv_results(results):

    print('CV accuracy = %.3f' % np.array(results['accuracy']).mean())

    if len(results['recall_1']) != 0:

    	print('CV build time = %s' % sec2time(np.array(results['build_times']).mean()))
    	print('CV optimality = %.3f' % np.array(results['optimality_gaps']).mean())
    	table = PrettyTable()
    	table.field_names = ['metrics','1', '0']
    	table.add_row(['recall', '%.3f' % np.array(results['recall_1']).mean(), '%.3f' % np.array(results['recall_0']).mean()])
    	table.add_row(['precision', '%.3f' % np.array(results['precision_1']).mean(), '%.3f' % np.array(results['precision_0']).mean()])
    	table.add_row(['f1', '%.3f' % np.array(results['f1_1']).mean(), '%.3f' % np.array(results['f1_0']).mean()])
    	print(table)

    elif len(results['f1_macro']) != 0:

    	table = PrettyTable()
    	table.field_names = ['metrics','value']
    	table.add_row(['macro f1', '%.3f' % np.array(results['f1_macro']).mean()])
    	table.add_row(['micro f1', '%.3f' % np.array(results['f1_micro']).mean()])
    	print(table)

def binarize_sex(feature_name, class1_name, class2_name, df_train, df_test):

	# train
	data_train = df_train[feature_name].to_numpy()

	index = df_train.columns.get_loc(feature_name)
	df_train.drop(feature_name, axis=1, inplace=True)

	df_train.insert(index, class1_name, 1 - data_train, True)
	df_train.insert(index+1, class2_name, data_train, True)

	# test
	data_test = df_test[feature_name].to_numpy()

	index = df_test.columns.get_loc(feature_name)
	df_test.drop(feature_name, axis=1, inplace=True)

	df_test.insert(index, class1_name, 1 - data_test, True)
	df_test.insert(index+1, class2_name, data_test, True)

	return df_train, df_test, [class1_name, class2_name]

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