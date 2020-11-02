import os
import numpy as np
import pandas as pd

def binarize_real_value(feature_name, n, df):

	ages = df[feature_name].to_numpy()
	target = df['target'].to_numpy()
	ages_sorted = np.unique(np.sort(ages))
	best_limit = 0
	best_gini = 1.0
	
	# find split
	for limit in ages_sorted[:-1]:

		group1 = ages[ages > limit]
		group2 = ages[ages <= limit]

		target_group1 = target[ages > limit]
		target_group2 = target[ages <= limit]
		
		p1 = len(group1[target_group1 == 1])/len(group1)
		p2 = len(group1[target_group1 == 0])/len(group1)
		gini1 = 1 - pow(p1,2) - pow(p2,2)

		p3 = len(group2[target_group2 == 1])/len(group2)
		p4 = len(group2[target_group2 == 0])/len(group2)
		gini2 = 1 - pow(p3,2) - pow(p4,2)

		gini = (len(group1)/len(ages)) * gini1 + (len(group2)/len(ages)) * gini2

		if best_gini > gini:
			best_limit = limit
			best_gini = gini

	print("best limit = %d" % best_limit)

	# binarize feature
	subfeature1 = np.array(ages)
	subfeature2 = np.array(ages)
	
	subfeature1[subfeature1 <= best_limit] = 1
	subfeature1[subfeature1 > best_limit] = 0

	subfeature2[subfeature2 <= best_limit] = 0
	subfeature2[subfeature2 > best_limit] = 1

	# edit dataframe
	index = df.columns.get_loc(feature_name)
	df.insert(index, "%s <= %d" % (feature_name, best_limit), subfeature1, True)
	df.insert(index+1, "%s > %d" % (feature_name, best_limit), subfeature2, True)
	df.drop(feature_name, axis=1, inplace=True)

	return df

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

		df.insert(index, "%s > %d" % (feature_name, limit), subfeature, True)
		index+=1

	return df



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
