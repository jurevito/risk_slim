import os
import numpy as np
import pandas as pd
from riskmodel import RiskModel
import matplotlib.pyplot as plt
import seaborn as sns

from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

from preprocess import binarize_greater, binarize_interval, binarize_category, binarize_sex, binarize, binarize_manual, binarize_limits, sec2time, riskslim_cv
from prettytable import PrettyTable

# setup variables
output_file = open('result.txt', 'w+')
file = 'breast'
test_size = 0.2
n_folds = 5
max_runtime = 2.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file + '.csv'
df  = pd.read_csv(path, float_precision='round_trip')

# removing highly coorelated features
df = df.drop(['radius_worst', 'radius_mean','area_worst','perimeter_mean','area_mean'], axis=1)
df = df.drop(['texture_worst','concave points_mean','radius_se'], axis=1)
df = df.drop(['compactness_mean','compactness_worst','concavity_mean'], axis=1)

# split data
df = shuffle(df, random_state=1)
train_df, test_df = train_test_split(df, test_size=test_size, random_state=0)

# binarizing train and test set
train_df, test_df, texture_mean, texture_mean_limits = binarize_limits('texture_mean', train_df, test_df, [15, 21.5])
train_df, test_df, smoothness_mean, smoothness_mean_limits = binarize_limits('smoothness_mean', train_df, test_df, [0.880, 0.903])
train_df, test_df, symmetry_mean, symmetry_mean_limits = binarize_limits('symmetry_mean', train_df, test_df, [0.148, 0.153])
train_df, test_df, fractal_dimension_mean, fractal_dimension_mean_limits = binarize_limits('fractal_dimension_mean', train_df, test_df, [0.558, 0.572])
train_df, test_df, texture_se, texture_se_limits = binarize_limits('texture_se', train_df, test_df, [0.82])
train_df, test_df, perimeter_se, perimeter_se_limits = binarize_limits('perimeter_se', train_df, test_df, [4.12, 1.5])
train_df, test_df, area_se, area_se_limits = binarize_limits('area_se', train_df, test_df, [17])
train_df, test_df, smoothness_se, smoothness_se_limits = binarize_limits('smoothness_se', train_df, test_df, [0.01, 0.004])
train_df, test_df, compactness_se, compactness_se_limits = binarize_limits('compactness_se', train_df, test_df, [0.011, 0.041])
train_df, test_df, concavity_se, concavity_se_limits = binarize_limits('concavity_se', train_df, test_df, [0.0105])
train_df, test_df, concave_points_se, concave_points_se_limits = binarize_limits('concave points_se', train_df, test_df, [0.0095])
train_df, test_df, symmetry_se, symmetry_se_limits = binarize_limits('symmetry_se', train_df, test_df, [0.014])
train_df, test_df, fractal_dimension_se, fractal_dimension_se_limits = binarize_limits('fractal_dimension_se', train_df, test_df, [0.003, 0.0052])
train_df, test_df, perimeter_worst, perimeter_worst_limits = binarize_limits('perimeter_worst', train_df, test_df, [98, 117, 103])
train_df, test_df, smoothness_worst, smoothness_worst_limits = binarize_limits('smoothness_worst', train_df, test_df, [0.135, 0.14])
train_df, test_df, concavity_worst, concavity_worst_limits = binarize_limits('concavity_worst', train_df, test_df, [0.19, 0.21, 0.37])
train_df, test_df, concave_points_worst, concave_points_worst_limits = binarize_limits('concave points_worst', train_df, test_df, [0.114, 0.15])
train_df, test_df, symmetry_worst, symmetry_worst_limits = binarize_limits('symmetry_worst', train_df, test_df, [0.33, 0.265])
train_df, test_df, fractal_dimension_worst, fractal_dimension_worst_limits = binarize_limits('fractal_dimension_worst', train_df, test_df, [0.093])

print('number of features = %d' % len(train_df.columns))

X_labels = train_df.columns[1:]
y_label = train_df.columns[0]
X = train_df[X_labels]
y = train_df[y_label]

# stump selection
stump_select = SelectFromModel(LogisticRegression(solver='liblinear', C=0.55, penalty='l1'))
stump_select.fit(X, y)
selected_features = list(X_labels[stump_select.get_support()])
selected_features.insert(0, 'diagnosis')
removed_features = np.setdiff1d(X_labels,selected_features)
print(removed_features)
print("removed features = %d - %d = %d" % (len(X_labels),len(removed_features), len(X_labels) - len(removed_features)))

train_df = train_df[selected_features]
test_df = test_df[selected_features]

texture_mean = list(set(texture_mean) & set(selected_features))
smoothness_mean = list(set(smoothness_mean) & set(selected_features))
symmetry_mean = list(set(symmetry_mean) & set(selected_features))
fractal_dimension_mean = list(set(fractal_dimension_mean) & set(selected_features))
texture_se = list(set(texture_se) & set(selected_features))
perimeter_se = list(set(perimeter_se) & set(selected_features))
area_se = list(set(area_se) & set(selected_features))
smoothness_se = list(set(smoothness_se) & set(selected_features))
compactness_se = list(set(compactness_se) & set(selected_features))
concavity_se = list(set(concavity_se) & set(selected_features))
concave_points_se = list(set(concave_points_se) & set(selected_features))
symmetry_se = list(set(symmetry_se) & set(selected_features))
fractal_dimension_se = list(set(fractal_dimension_se) & set(selected_features))
perimeter_worst = list(set(perimeter_worst) & set(selected_features))
smoothness_worst = list(set(smoothness_worst) & set(selected_features))
concavity_worst = list(set(concavity_worst) & set(selected_features))
concave_points_worst = list(set(concave_points_worst) & set(selected_features))
symmetry_worst = list(set(symmetry_worst) & set(selected_features))
fractal_dimension_worst = list(set(fractal_dimension_worst) & set(selected_features))

# saving processed data
train_df.to_csv('risk_slim/train_data.csv', sep=',', index=False,header=True)
test_df.to_csv('risk_slim/test_data.csv', sep=',', index=False,header=True)

params = {
    'max_coefficient' : 6,                    # value of largest/smallest coefficient
    'max_L0_value' : 8,                       # maximum model size (set as float(inf))
    'max_offset' : 50,                        # maximum value of offset parameter (optional)
    'c0_value' : 1e-6,                        # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    'w_pos' : 1.00                            # relative weight on examples with y = +1; w_neg = 1.00 (optional)
}

settings = {

    'c0_value': params['c0_value'],
    'w_pos': params['w_pos'],

    # LCPA Settings
    'max_runtime': max_runtime,                                # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # print CPLEX progress on screen
    'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')

    # LCPA Improvements
    'round_flag': True,                                # round continuous solutions with SeqRd
    'polish_flag': True,                               # polish integer feasible solutions with DCD
    'chained_updates_flag': True,                      # use chained updates
    'add_cuts_at_heuristic_solutions': True,           # add cuts at integer feasible solutions found using polishing/rounding

    # Initialization
    'initialization_flag': True,                       # use initialization procedure
    'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
    'init_max_coefficient_gap': 0.49,

    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}

# operation constraints
op_constraints = {
    'texture_mean' : texture_mean,
    'smoothness_mean' : smoothness_mean,
    'symmetry_mean' : symmetry_mean,
    'fractal_dimension_mean' : fractal_dimension_mean,
    'texture_se' : texture_se,
    'perimeter_se' : perimeter_se,
    'area_se' : area_se,
    'smoothness_se' : smoothness_se,
    'compactness_se' : compactness_se,
    'concavity_se' : concavity_se,
    'concave_points_se' : concave_points_se,
    'symmetry_se' : symmetry_se,
    'fractal_dimension_se' : fractal_dimension_se,
    'perimeter_worst' : perimeter_worst,
    'smoothness_worst' : smoothness_worst,
    'concavity_worst' : concavity_worst,
    'concave_points_worst' : concave_points_worst,
    'symmetry_worst' : symmetry_worst,
    'fractal_dimension_worst' : fractal_dimension_worst,
}

# preparing data
df_train  = pd.read_csv('risk_slim/train_data.csv', float_precision='round_trip')
df_test  = pd.read_csv('risk_slim/test_data.csv', float_precision='round_trip')

X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:,0].values

# cross validating
rm = RiskModel(data_headers=df_train.columns.values, params=params, settings=settings, op_constraints=op_constraints)
cv_result, build_times, opt_gaps = riskslim_cv(n_folds,rm, X_train, y_train)

# fitting model
rm.fit(X_train,y_train)
y_pred = rm.predict(X_test)

# print metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))
print("optimality_gap = %.3f" % rm.model_info['optimality_gap'])
print(sec2time(rm.model_info['solver_time']))

# roc auc
y_roc_pred = rm.decision_function(X_test)
fpr_risk, tpr_risk, treshold_risk = roc_curve(y_test, y_roc_pred)
auc_risk = auc(fpr_risk, tpr_risk)

# saving results and model info
table1 = PrettyTable(["Parameter","Value"])
table1.add_row(["Accuracy", "%0.2f" % accuracy_score(y_test, y_pred)])
table1.add_row(["AUC", "%0.2f" % auc_risk])
table1.add_row(["CV-%d" % n_folds ,"%0.2f (+/- %0.2f)" % (cv_result.mean(), cv_result.std()*2)])
table1.add_row(["Avg. Run Time", "%.0f (+/- %.0f)" % (build_times.mean(), build_times.std()*2)])
table1.add_row(["Test Run Time", round(rm.model_info['solver_time'])])
table1.add_row(["Run Hours", sec2time(rm.model_info['solver_time'])])
table1.add_row(["Max Time", max_runtime])
table1.add_row(["Max Features", params['max_L0_value']])
table1.add_row(["Total Stumps", len(df_train.columns)])
table1.add_row(["Avg. Optimality Gap", "%.3f (+/- %.3f)" % (opt_gaps.mean(), opt_gaps.std()*2)])
table1.add_row(["Optimality Gap", round(rm.model_info['optimality_gap'],3)])

output_file.write(str(table1))
output_file.close()

# plotting roc curve
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(fpr_risk, tpr_risk, linestyle='-', label='Risk Slim (auc = %0.2f)' % auc_risk)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

