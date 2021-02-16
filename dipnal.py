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

from preprocess import binarize_limits, sec2time, find_treshold_index, stump_selection, fix_names, print_cv_results, binarize_sex, impute_dataframes
from prettytable import PrettyTable

from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier
import time

# setup variables
output_file = open('result.txt', 'w+')
file = 'Disease-MUL.hd5'
test_size = 0.2
n_folds = 5
max_runtime = 1.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file
hdf  = pd.HDFStore(path, mode='r')
df = hdf.get('/Xy')
with_s = [x for x in df.columns.values if x.startswith('S')]
df = df.drop(with_s, axis=1)

percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
removed_features = list(missing_value_df.loc[missing_value_df['percent_missing'] >= 99.98, 'column_name'])
print('total removed = %d (%.2f%%)' % (len(removed_features), (len(removed_features) / len(df.columns)*100)))
df = df.drop(removed_features, axis=1)

# category to int
LE = LabelEncoder()
df['class'] = LE.fit_transform(df['class'])

# split data
df = shuffle(df, random_state=1)
df_train, df_test = train_test_split(df, test_size=test_size, random_state=0, stratify=df['class'])

# data imputation
df_train, df_test = impute_dataframes(df_train, df_test)

# remove highly coorelated features
df_train = df_train.drop(['X207','X088','X075','X083','X073','X245','X052','X211','X087','X076','X226','X246','X022','X095','X111','X055','X159','X049','X212','X270','X085','X053'], axis=1)
df_test = df_test.drop(['X207','X088','X075','X083','X073','X245','X052','X211','X087','X076','X226','X246','X022','X095','X111','X055','X159','X049','X212','X270','X085','X053'], axis=1)

# move class to beginning
outcome_values = df_train['class'].values
df_train = df_train.drop(['class'], axis=1)
df_train.insert(0, 'class', outcome_values, True)
outcome_values = df_test['class'].values
df_test = df_test.drop(['class'], axis=1)
df_test.insert(0, 'class', outcome_values, True)

# real valued feature selection
selected_features = stump_selection(0.05, df_train, False)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print(df_train)

# binarizing train and test set
df_train, df_test, X029 = binarize_limits('X029', df_train, df_test, [0.13, -0.1, 0.24, 0.02])
df_train, df_test, X031 = binarize_limits('X031', df_train, df_test, [0.2, -0.17, -0.2])
df_train, df_test, X056 = binarize_limits('X056', df_train, df_test, [-0.2, 0.18, 0.2, 0])
df_train, df_test, X090 = binarize_limits('X090', df_train, df_test, [0.02, 0.1, -0.25, -0.42])
df_train, df_test, X103 = binarize_limits('X103', df_train, df_test, [0.7, -0.16, -0.03, 0.1])
df_train, df_test, X106 = binarize_limits('X106', df_train, df_test, [-0.1, 0.01, 0.3, -0.05])
df_train, df_test, X109 = binarize_limits('X109', df_train, df_test, [0.1, 0, -0.16])
df_train, df_test, X115 = binarize_limits('X115', df_train, df_test, [0.15, 0.2])
df_train, df_test, X141 = binarize_limits('X141', df_train, df_test, [-0.22, 0.19, 0.3])
df_train, df_test, X145 = binarize_limits('X145', df_train, df_test, [-0.1, 0, 0.15, 0.19])
df_train, df_test, X158 = binarize_limits('X158', df_train, df_test, [-0.22])
df_train, df_test, X160 = binarize_limits('X160', df_train, df_test, [0.12, -0.16, 0])
df_train, df_test, X162 = binarize_limits('X162', df_train, df_test, [0.2, 0.15])
df_train, df_test, X184 = binarize_limits('X184', df_train, df_test, [-0.15, 0.02, 0.1, -0.18, 0.3])
df_train, df_test, X186 = binarize_limits('X186', df_train, df_test, [-0.18, 0.1])
df_train, df_test, X192 = binarize_limits('X192', df_train, df_test, [0.14, -0.14, -0.3])
df_train, df_test, X193 = binarize_limits('X193', df_train, df_test, [-0.18, 0.17, -0.25, 0.13])
df_train, df_test, X197 = binarize_limits('X197', df_train, df_test, [-0.35, -0.5])
df_train, df_test, X217 = binarize_limits('X217', df_train, df_test, [-0.04, -0.1, -0.38])
df_train, df_test, X221 = binarize_limits('X221', df_train, df_test, [0.24, 0, 0.38])
df_train, df_test, X269 = binarize_limits('X269', df_train, df_test, [0.3, 0.24, -0.4])
df_train, df_test, X277 = binarize_limits('X277', df_train, df_test, [-0.3, 0.13, -0.24, 0])

print('1. n_features = %d' % len(df_train.columns))

# binary valued feature selection
selected_features = stump_selection(0.0002, df_train, False)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('2. n_features = %d' % len(df_train.columns))

X029 = fix_names(X029, selected_features)
X031 = fix_names(X031, selected_features)
X056 = fix_names(X056, selected_features)
X090 = fix_names(X090, selected_features)
X103 = fix_names(X103, selected_features)
X106 = fix_names(X106, selected_features)
X109 = fix_names(X109, selected_features)
X115 = fix_names(X115, selected_features)
X141 = fix_names(X141, selected_features)
X145 = fix_names(X145, selected_features)
X158 = fix_names(X158, selected_features)
X160 = fix_names(X160, selected_features)
X162 = fix_names(X162, selected_features)
X184 = fix_names(X184, selected_features)
X186 = fix_names(X186, selected_features)
X192 = fix_names(X192, selected_features)
X193 = fix_names(X193, selected_features)
X197 = fix_names(X197, selected_features)
X217 = fix_names(X217, selected_features)
X221 = fix_names(X221, selected_features)
X269 = fix_names(X269, selected_features)
X277 = fix_names(X277, selected_features)

params = {
    'max_coefficient' : 6,                    # value of largest/smallest coefficient
    'max_L0_value' : 5,                       # maximum model size (set as float(inf))
    'max_offset' : 50,                        # maximum value of offset parameter (optional)
    'c0_value' : 1e-5,                        # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
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
    'X029': X029,
    'X031': X031,
    'X056': X056,
    'X090': X090,
    'X103': X103,
    'X106': X106,
    'X109': X109,
    'X115': X115,
    'X141': X141,
    'X145': X145,
    'X158': X158,
    'X160': X160,
    'X162': X162,
    'X184': X184,
    'X186': X186,
    'X192': X192,
    'X193': X193,
    'X197': X197,
    'X217': X217,
    'X221': X221,
    'X269': X269,
    'X277': X277,
}


# preparing data
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
data_headers = df_train.columns

model = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)
rm = OneVsRestClassifier(estimator=model)

# cross validating
"""kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 0)
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
    results['optimality_gaps'].append(rm.model_info['optimality_gap'])"""

# fitting model
rm.fit(X_train,y_train)

# print cv results
#print(results['accuracy'])
#print_cv_results(results)

# printing metrics
print('Testing results:')
y_pred = rm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))
print("optimality_gap = %.3f" % rm.model_info['optimality_gap'])
print(sec2time(rm.model_info['solver_time']))

# roc auc
"""y_roc_pred = rm.predict_proba(X_test)
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
plt.show()"""