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

def binarize_real(feature_name, k, df):

	data = df[feature_name].to_numpy()

	# generate limits
	max_value = np.amax(data)
	min_value = np.amin(data)
	step = int((max_value - min_value) * k)
	limits = list(range(min_value + int(max_value*0.01), max_value - int(max_value*0.01), step))

	# limits per feature
	str1 = ','.join(str(e) for e in limits)
	print("%s_limits = %s" % (feature_name, str1))

	index = df.columns.get_loc(feature_name)
	df.drop(feature_name, axis=1, inplace=True)
	
	for limit in limits:

		subfeature = np.array(data)
		subfeature[subfeature <= limit] = 0
		subfeature[subfeature > limit] = 1

		df.insert(index, "%s > %d" % (feature_name, limit), subfeature, True)
		index+=1

	return df


os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + 'heart.csv'

df  = pd.read_csv(path, float_precision='round_trip')
X = df.iloc[:, 0:-1].values
y = df.iloc[:,-1].values
y[y == -1] = 0

# binarizing features
df = binarize_real('age', 0.2, df)
df = binarize_real('trestbps', 0.15, df)
df = binarize_real('chol', 0.08, df)
df = binarize_real('thalach', 0.15, df)

# moving target to beginning
df.drop('target', axis=1, inplace=True)
df.insert(0, "target", y, True)

# saving processed data
df.to_csv('risk_slim/hrt.csv', sep=',', index=False,header=True)
