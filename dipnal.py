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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

from preprocess import binarize_limits, sec2time, riskslim_cv, find_treshold_index, stump_selection, fix_names, print_cv_results, binarize_sex
from prettytable import PrettyTable

from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier
import time

# setup variables
output_file = open('result.txt', 'w+')
file = 'breast.csv'
test_size = 0.2
n_folds = 5
max_runtime = 3600.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file
df  = pd.read_csv(path, float_precision='round_trip')

# removing highly coorelated features
df = df.drop(['radius_worst', 'radius_mean','area_worst','perimeter_mean','area_mean'], axis=1)
df = df.drop(['texture_worst','concave points_mean','radius_se'], axis=1)
df = df.drop(['compactness_mean','compactness_worst','concavity_mean'], axis=1)

# split data
df = shuffle(df, random_state=1)
df_train, df_test = train_test_split(df, test_size=test_size, random_state=0, stratify=df['diagnosis'])

# binarizing train and test set
df_train, df_test, texture_mean = binarize_limits('texture_mean', df_train, df_test, [15, 22])
df_train, df_test, smoothness_mean = binarize_limits('smoothness_mean', df_train, df_test, [0.085, 0.116])
df_train, df_test, symmetry_mean = binarize_limits('symmetry_mean', df_train, df_test, [0.15])
df_train, df_test, fractal_dimension_mean = binarize_limits('fractal_dimension_mean', df_train, df_test, [0.053, 0.073])
df_train, df_test, texture_se = binarize_limits('texture_se', df_train, df_test, [2.3])
df_train, df_test, perimeter_se = binarize_limits('perimeter_se', df_train, df_test, [1.4, 3, 4.1])
df_train, df_test, area_se = binarize_limits('area_se', df_train, df_test, [19, 36])
df_train, df_test, smoothness_se = binarize_limits('smoothness_se', df_train, df_test, [0.022, 0.003])
df_train, df_test, compactness_se = binarize_limits('compactness_se', df_train, df_test, [0.045])
df_train, df_test, concavity_se = binarize_limits('concavity_se', df_train, df_test, [0.01, 0.17])
df_train, df_test, concave_points_se = binarize_limits('concave points_se', df_train, df_test, [0.021])
df_train, df_test, symmetry_se = binarize_limits('symmetry_se', df_train, df_test, [0.045, 0.011])
df_train, df_test, fractal_dimension_se = binarize_limits('fractal_dimension_se', df_train, df_test, [0.013, 0.0047])
df_train, df_test, perimeter_worst = binarize_limits('perimeter_worst', df_train, df_test, [100, 115])
df_train, df_test, smoothness_worst = binarize_limits('smoothness_worst', df_train, df_test, [0.18, 0.14])
df_train, df_test, concavity_worst = binarize_limits('concavity_worst', df_train, df_test, [0.19, 0.27])
df_train, df_test, concave_points_worst = binarize_limits('concave points_worst', df_train, df_test, [0.14, 0.15])
df_train, df_test, symmetry_worst = binarize_limits('symmetry_worst', df_train, df_test, [0.2, 0.37])
df_train, df_test, fractal_dimension_worst = binarize_limits('fractal_dimension_worst', df_train, df_test, [0.09, 0.1])

print('1. n_features = %d' % len(df_train.columns))

# binary valued feature selection
selected_features = stump_selection(0.8, df_train)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('2. n_features = %d' % len(df_train.columns))

texture_mean = fix_names(texture_mean, selected_features)
smoothness_mean = fix_names(smoothness_mean, selected_features)
symmetry_mean = fix_names(symmetry_mean, selected_features)
fractal_dimension_mean = fix_names(fractal_dimension_mean, selected_features)
texture_se = fix_names(texture_se, selected_features)
perimeter_se = fix_names(perimeter_se, selected_features)
area_se = fix_names(area_se, selected_features)
smoothness_se = fix_names(smoothness_se, selected_features)
compactness_se = fix_names(compactness_se, selected_features)
concavity_se = fix_names(concavity_se, selected_features)
concave_points_se = fix_names(concave_points_se, selected_features)
symmetry_se = fix_names(symmetry_se, selected_features)
fractal_dimension_se = fix_names(fractal_dimension_se, selected_features)
perimeter_worst = fix_names(perimeter_worst, selected_features)
smoothness_worst = fix_names(smoothness_worst, selected_features)
concavity_worst = fix_names(concavity_worst, selected_features)
concave_points_worst = fix_names(concave_points_worst, selected_features)
symmetry_worst = fix_names(symmetry_worst, selected_features)
fractal_dimension_worst = fix_names(fractal_dimension_worst, selected_features)

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
    'texture_mean': texture_mean,
    'smoothness_mean': smoothness_mean,
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean,
    'texture_se': texture_se,
    'perimeter_se': perimeter_se,
    'area_se': area_se,
    'smoothness_se': smoothness_se,
    'compactness_se': compactness_se,
    'concavity_se': concavity_se,
    'concave_points_se': concave_points_se,
    'symmetry_se': symmetry_se,
    'fractal_dimension_se': fractal_dimension_se,
    'perimeter_worst': perimeter_worst,
    'smoothness_worst': smoothness_worst,
    'concavity_worst': concavity_worst,
    'concave_points_worst': concave_points_worst,
    'symmetry_worst': symmetry_worst,
    'fractal_dimension_worst': fractal_dimension_worst,
}

# preparing data
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
data_headers = df_train.columns


rm = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# cross validating
kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 0)
results = {
    'accuracy': [],
    'build_times': [],
    'optimality_gaps': [],
    'recall_1': [],
    'recall_0': [],
    'precision_1': [],
    'precision_0': [],
    'f1_1': [],
    'f1_0': [],
}

for train_index, valid_index in kf.split(X_train, y_train):

    X_train_cv = X_train[train_index]
    y_train_cv = y_train[train_index]

    X_valid_cv = X_train[valid_index]
    y_valid_cv = y_train[valid_index]

    rm.fit(X_train_cv, y_train_cv)
    y_pred = rm.predict(X_valid_cv)

    results['accuracy'].append(accuracy_score(y_valid_cv, y_pred))
    results['recall_1'].append(recall_score(y_valid_cv, y_pred, pos_label=1))
    results['recall_0'].append(recall_score(y_valid_cv, y_pred, pos_label=0))
    results['precision_1'].append(precision_score(y_valid_cv, y_pred, pos_label=1))
    results['precision_0'].append(precision_score(y_valid_cv, y_pred, pos_label=0))
    results['f1_1'].append(f1_score(y_valid_cv, y_pred, pos_label=1))
    results['f1_0'].append(f1_score(y_valid_cv, y_pred, pos_label=0))

    results['build_times'].append(rm.model_info['solver_time'])
    results['optimality_gaps'].append(rm.model_info['optimality_gap'])

# fitting model
rm.fit(X_train,y_train)

# print cv results
print(results['accuracy'])
print_cv_results(results)

# printing metrics
print('Testing results:')
y_pred = rm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))
print("optimality_gap = %.3f" % rm.model_info['optimality_gap'])
print(sec2time(rm.model_info['solver_time']))

# roc auc
y_roc_pred = rm.predict_proba(X_test)
fpr_risk, tpr_risk, treshold_risk = roc_curve(y_test, y_roc_pred)
auc_risk = auc(fpr_risk, tpr_risk)
op_index = find_treshold_index(treshold_risk, 0.5)

# saving results and model info
cv_result = np.array(results['accuracy'])
build_times = np.array(results['build_times'])
opt_gaps = np.array(results['optimality_gaps'])

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
plt.plot([fpr_risk[op_index]], [tpr_risk[op_index]], marker='o', color='cyan')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
