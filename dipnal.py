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

from preprocess import binarize_limits, sec2time, riskslim_cv, find_treshold_index, stump_selection, fix_names, print_cv_results
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

# move outcome at beginning
outcome_values = df['class'].values
df = df.drop(['class'], axis=1)
df.insert(0, 'class', outcome_values, True)

# category to int
LE = LabelEncoder()
df['class'] = LE.fit_transform(df['class'])

# show missing value percentage
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True)
removed_features = list(missing_value_df.loc[missing_value_df['percent_missing'] >= 98.00, 'column_name'])
print('total removed = %d (%.2f%%)' % (len(removed_features), (len(removed_features) / len(df.columns)*100)))
df = df.drop(removed_features, axis=1)

# split data
df = shuffle(df, random_state=1)
df_train, df_test = train_test_split(df, test_size=test_size, random_state=0, stratify=df['class'])

# data imputation
tmp1 = df_train
tmp2 = df_test

imputer = KNNImputer(n_neighbors=2)
df_train = pd.DataFrame(imputer.fit_transform(df_train))
df_test = pd.DataFrame(imputer.transform(df_test))

df_train.columns = tmp1.columns
df_train.index = tmp1.index
df_test.columns = tmp2.columns
df_test.index = tmp2.index

# remove highly coorelated features
df_train = df_train.drop(['X212','X094','X109','X213','X273','X096','X095','X138','X121','X107'], axis=1)
df_test = df_test.drop(['X212','X094','X109','X213','X273','X096','X095','X138','X121','X107'], axis=1)

corr_lower = 0.85
corr = df_train.corr().abs()
s = corr.unstack()
so = s.sort_values(kind="quicksort")
so = so[(so > corr_lower) & (so < 1.0)]
print(so)

# real valued feature selection
selected_features = stump_selection(0.03, df_train)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

# binarizing train and test set
df_train, df_test, X028 = binarize_limits('X028', df_train, df_test, [-0.01])
df_train, df_test, X051 = binarize_limits('X051', df_train, df_test, [-0.05, 0.045])
df_train, df_test, X103 = binarize_limits('X103', df_train, df_test, [-0.178, 0.1])
df_train, df_test, X111 = binarize_limits('X111', df_train, df_test, [-0.23, 0.12])
df_train, df_test, X118 = binarize_limits('X118', df_train, df_test, [-0.086, 0.051])
df_train, df_test, X127 = binarize_limits('X127', df_train, df_test, [0.115, -0.061])
df_train, df_test, X144 = binarize_limits('X144', df_train, df_test, [-0.06, -0.16, 0.025])
df_train, df_test, X146 = binarize_limits('X146', df_train, df_test, [-0.23, 0.13])
df_train, df_test, X162 = binarize_limits('X162', df_train, df_test, [-0.2, 0.04])
df_train, df_test, X172 = binarize_limits('X172', df_train, df_test, [0.17, 0.05])
df_train, df_test, X173 = binarize_limits('X173', df_train, df_test, [0.06, -0.055])
df_train, df_test, X174 = binarize_limits('X174', df_train, df_test, [-0.1, 0.04])
df_train, df_test, X175 = binarize_limits('X175', df_train, df_test, [-0.31, 0.16])
df_train, df_test, X180 = binarize_limits('X180', df_train, df_test, [0.135, 0.2, -0.12])
df_train, df_test, X201 = binarize_limits('X201', df_train, df_test, [0.23, -0.08])
df_train, df_test, X210 = binarize_limits('X210', df_train, df_test, [0, 0.124, -0.12])
df_train, df_test, X215 = binarize_limits('X215', df_train, df_test, [0.065, -0.03, 0.23])
df_train, df_test, X221 = binarize_limits('X221', df_train, df_test, [0.05, -0.23])
df_train, df_test, X247 = binarize_limits('X247', df_train, df_test, [-0.05, 0, 0.02])
df_train, df_test, X266 = binarize_limits('X266', df_train, df_test, [-0.05, 0.065, 0])
df_train, df_test, X271 = binarize_limits('X271', df_train, df_test, [0.05, -0.02, -0.11])
df_train, df_test, X272 = binarize_limits('X272', df_train, df_test, [0.1, -0.1])
df_train, df_test, X277 = binarize_limits('X277', df_train, df_test, [-0.33])
df_train, df_test, X278 = binarize_limits('X278', df_train, df_test, [0.5])

print('1. n_features = %d' % len(df_train.columns))

# binary valued feature selection
selected_features = stump_selection(0.005, df_train)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('2. n_features = %d' % len(df_train.columns))

X028 = fix_names(X028, selected_features)
X051 = fix_names(X051, selected_features)
X103 = fix_names(X103, selected_features)
X111 = fix_names(X111, selected_features)
X118 = fix_names(X118, selected_features)
X127 = fix_names(X127, selected_features)
X144 = fix_names(X144, selected_features)
X146 = fix_names(X146, selected_features)
X162 = fix_names(X162, selected_features)
X172 = fix_names(X172, selected_features)
X173 = fix_names(X173, selected_features)
X174 = fix_names(X174, selected_features)
X175 = fix_names(X175, selected_features)
X180 = fix_names(X180, selected_features)
X201 = fix_names(X201, selected_features)
X210 = fix_names(X210, selected_features)
X215 = fix_names(X215, selected_features)
X221 = fix_names(X221, selected_features)
X247 = fix_names(X247, selected_features)
X266 = fix_names(X266, selected_features)
X271 = fix_names(X271, selected_features)
X272 = fix_names(X272, selected_features)
X277 = fix_names(X277, selected_features)
X278 = fix_names(X278, selected_features)

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
    'X028': X028,
    'X051': X051,
    'X103': X103,
    'X111': X111,
    'X118': X118,
    'X127': X127,
    'X144': X144,
    'X146': X146,
    'X162': X162,
    'X172': X172,
    'X173': X173,
    'X174': X174,
    'X175': X175,
    'X180': X180,
    'X201': X201,
    'X210': X210,
    'X215': X215,
    'X221': X221,
    'X247': X247,
    'X266': X266,
    'X271': X271,
    'X272': X272,
    'X277': X277,
#   'X278': X278,
}

# preparing data
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
data_headers = df_train.columns


model = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)
rm = OneVsRestClassifier(model)

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
    'f1_macro': [],
    'f1_micro': [],
}

for train_index, valid_index in kf.split(X_train, y_train):

    X_train_cv = X_train[train_index]
    y_train_cv = y_train[train_index]

    X_valid_cv = X_train[valid_index]
    y_valid_cv = y_train[valid_index]

    rm.fit(X_train_cv, y_train_cv)
    y_pred = rm.predict(X_valid_cv)

    results['accuracy'].append(accuracy_score(y_valid_cv, y_pred))

    n_classes = len(list(set(y_pred)))
    if n_classes < 3:
        results['recall_1'].append(recall_score(y_valid_cv, y_pred, pos_label=1))
        results['recall_0'].append(recall_score(y_valid_cv, y_pred, pos_label=0))
        results['precision_1'].append(precision_score(y_valid_cv, y_pred, pos_label=1))
        results['precision_0'].append(precision_score(y_valid_cv, y_pred, pos_label=0))
        results['f1_1'].append(f1_score(y_valid_cv, y_pred, pos_label=1))
        results['f1_0'].append(f1_score(y_valid_cv, y_pred, pos_label=0))
        results['build_times'].append(rm.model_info['solver_time'])
        results['optimality_gaps'].append(rm.model_info['optimality_gap'])
    else:
        results['f1_macro'].append(f1_score(y_valid_cv, y_pred, average='macro'))
        results['f1_micro'].append(f1_score(y_valid_cv, y_pred, average='micro'))

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