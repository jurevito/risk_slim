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
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

from preprocess import binarize_limits, sec2time, find_treshold_index, stump_selection, fix_names, print_cv_results, binarize_sex
from prettytable import PrettyTable

from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier
import time

# setup variables
output_file = open('result.txt', 'w+')
file = 'disease_bin_imputed.h5'
test_size = 0.2
n_folds = 5
max_runtime = 10.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file
df_train = pd.read_hdf(path, 'train')
df_test = pd.read_hdf(path, 'test')

# move outcome at beginning
outcome_values = df_train['class'].values
df_train = df_train.drop(['class'], axis=1)
df_train.insert(0, 'class', outcome_values, True)
outcome_values = df_test['class'].values
df_test = df_test.drop(['class'], axis=1)
df_test.insert(0, 'class', outcome_values, True)

# remove highly coorelated features
df_train = df_train.drop(['X207','X249','X245','X246','X068','X248','X075','X211','X083','X089','X114','X055','X076','X124','X085','X052','X094','X053','X237','X111','X185','X088','X185','X090','X073','X212','X090','X072','X076'], axis=1)
df_test = df_test.drop(['X207','X249','X245','X246','X068','X248','X075','X211','X083','X089','X114','X055','X076','X124','X085','X052','X094','X053','X237','X111','X185','X088','X185','X090','X073','X212','X090','X072','X076'], axis=1)

# feature selection
selected_features = stump_selection(0.03, df_train, True)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('1. n_features = %d' % len(df_train.columns))

# binarizing train and test set
df_train, df_test, X009 = binarize_limits('X009', df_train, df_test, [-2.5])
df_train, df_test, X024 = binarize_limits('X024', df_train, df_test, [0.3, 0.1])
df_train, df_test, X040 = binarize_limits('X040', df_train, df_test, [0.06, 0.3])
df_train, df_test, X041 = binarize_limits('X041', df_train, df_test, [0, -0.1])
df_train, df_test, X056 = binarize_limits('X056', df_train, df_test, [-0.1, 2.4])
df_train, df_test, X065 = binarize_limits('X065', df_train, df_test, [-0.2, -0.05])
df_train, df_test, X109 = binarize_limits('X109', df_train, df_test, [-0.03, -0.165])
df_train, df_test, X110 = binarize_limits('X110', df_train, df_test, [-0.09, -0.12])
df_train, df_test, X113 = binarize_limits('X113', df_train, df_test, [-0.06, 0.3])
df_train, df_test, X144 = binarize_limits('X144', df_train, df_test, [0.1, -0.05])
df_train, df_test, X149 = binarize_limits('X149', df_train, df_test, [-0.12, 0.12])
df_train, df_test, X159 = binarize_limits('X159', df_train, df_test, [-0.17, 0.035])
df_train, df_test, X162 = binarize_limits('X162', df_train, df_test, [-0.28, -0.06])
df_train, df_test, X163 = binarize_limits('X163', df_train, df_test, [0.33, 0.37])
df_train, df_test, X170 = binarize_limits('X170', df_train, df_test, [-0.18, -0.04, 0.23])
df_train, df_test, X171 = binarize_limits('X171', df_train, df_test, [0.1, -0.24])
df_train, df_test, X187 = binarize_limits('X187', df_train, df_test, [-0.23, 0.26, 0.35])
df_train, df_test, X204 = binarize_limits('X204', df_train, df_test, [-0.29, -0.19, 0.04])
df_train, df_test, X215 = binarize_limits('X215', df_train, df_test, [-0.2, 0.13, 0.16])
df_train, df_test, X225 = binarize_limits('X225', df_train, df_test, [-0.1, 0.05, -0.29])
df_train, df_test, X234 = binarize_limits('X234', df_train, df_test, [-0.17, 0.01])
df_train, df_test, X244 = binarize_limits('X244', df_train, df_test, [0, -0.24])
df_train, df_test, X270 = binarize_limits('X270', df_train, df_test, [0.4, 0.16])

print('2. n_features = %d' % len(df_train.columns))

# binary valued feature selection
selected_features = stump_selection(0.015, df_train, True)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('3. n_features = %d' % len(df_train.columns))

X009 = fix_names(X009, selected_features)
X024 = fix_names(X024, selected_features)
X040 = fix_names(X040, selected_features)
X041 = fix_names(X041, selected_features)
X056 = fix_names(X056, selected_features)
X065 = fix_names(X065, selected_features)
X109 = fix_names(X109, selected_features)
X110 = fix_names(X110, selected_features)
X113 = fix_names(X113, selected_features)
X144 = fix_names(X144, selected_features)
X149 = fix_names(X149, selected_features)
X159 = fix_names(X159, selected_features)
X162 = fix_names(X162, selected_features)
X163 = fix_names(X163, selected_features)
X170 = fix_names(X170, selected_features)
X171 = fix_names(X171, selected_features)
X187 = fix_names(X187, selected_features)
X204 = fix_names(X204, selected_features)
X215 = fix_names(X215, selected_features)
X225 = fix_names(X225, selected_features)
X234 = fix_names(X234, selected_features)
X244 = fix_names(X244, selected_features)
X270 = fix_names(X270, selected_features)

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
    'X009': X009,
    'X024': X024,
    'X040': X040,
    'X041': X041,
    'X056': X056,
    'X065': X065,
    'X109': X109,
    'X110': X110,
    'X113': X113,
    'X144': X144,
    'X149': X149,
    'X159': X159,
    'X162': X162,
    'X163': X163,
    'X170': X170,
    'X171': X171,
    'X187': X187,
    'X204': X204,
    'X215': X215,
    'X225': X225,
    'X234': X234,
    'X244': X244,
    'X270': X270,
}

# preparing data
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
data_headers = df_train.columns

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
rm = RiskModel(data_headers=data_headers, params=params, settings=settings, sample_weights=sample_weights, op_constraints=op_constraints)

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

    rm.sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_cv)
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
rm.sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
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

# make result.h5 file save predictions and probabilites
"""df_results = pd.DataFrame()
df_results.insert(0, 'risk_pred', y_pred)
df_results.insert(0, 'risk_prob', y_roc_pred)
print(df_results)
df_results.to_hdf('results.h5', key='disease_bin', mode='w')"""

