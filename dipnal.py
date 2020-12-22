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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

from preprocess import binarize_limits, sec2time, riskslim_cv, find_treshold_index, stump_selection, fix_names
from prettytable import PrettyTable

from imblearn.combine import SMOTEENN

# setup variables
output_file = open('result.txt', 'w+')
file = 'Groups_knn.h5'
test_size = 0.1
n_folds = 5
max_runtime = 3600.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file
df_train = pd.read_hdf(path, 'train')
df_test = pd.read_hdf(path, 'test')

# move outcome at beginning
outcome_train = df_train['class'].values
outcome_test = df_test['class'].values
df_train = df_train.drop(['class'], axis=1)
df_test = df_test.drop(['class'], axis=1)
df_train.insert(0, 'class', outcome_train, True)
df_test.insert(0, 'class', outcome_test, True)

# class imbalance
zeros_class = (df_train['class'] == 0).astype(int).sum(axis=0)
ones_class = (df_train['class'] == 1).astype(int).sum(axis=0)
print('zeros = %d, ones = %d (%.2f%%)' % (zeros_class, ones_class, (zeros_class/len(df_train['class']))*100))

# remove highly coorelated features
df_train = df_train.drop(['X152','X294','X235','X237','X104','X076','X065','X037'], axis=1)
df_test = df_test.drop(['X152','X294','X235','X237','X104','X076','X065','X037'], axis=1)

# real valued feature selection
selected_features = stump_selection(0.04, df_train)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print(selected_features)

print('number of features = %d' % len(df_train.columns))

# binarizing train and test set
df_train, df_test, X009 = binarize_limits('X009', df_train, df_test, [-0.3, -0.2, 0.01])
df_train, df_test, X021 = binarize_limits('X021', df_train, df_test, [0.18, -0.02, 0.08])
df_train, df_test, X039 = binarize_limits('X039', df_train, df_test, [-0.085, 0.05])
df_train, df_test, X044 = binarize_limits('X044', df_train, df_test, [-0.04, -0.07])
df_train, df_test, X046 = binarize_limits('X046', df_train, df_test, [0.18, -0.2, -0.065])
df_train, df_test, X057 = binarize_limits('X057', df_train, df_test, [0, 0.075, 0.16])
df_train, df_test, X071 = binarize_limits('X071', df_train, df_test, [-0.25, -0.05, -0.27])
df_train, df_test, X081 = binarize_limits('X081', df_train, df_test, [-0.15, -0.1, 0.06])
df_train, df_test, X085 = binarize_limits('X085', df_train, df_test, [-0.1, -0.05, 0])
df_train, df_test, X096 = binarize_limits('X093', df_train, df_test, [0.23, 0.12])
df_train, df_test, X096 = binarize_limits('X096', df_train, df_test, [-0.22, -0.03])
df_train, df_test, X136 = binarize_limits('X136', df_train, df_test, [0.3, 0.24])
df_train, df_test, X183 = binarize_limits('X183', df_train, df_test, [0.07, -0.18, -0.2])
df_train, df_test, X200 = binarize_limits('X200', df_train, df_test, [-0.1, 0.2])
df_train, df_test, X207 = binarize_limits('X207', df_train, df_test, [0.02])
df_train, df_test, X211 = binarize_limits('X211', df_train, df_test, [-0.24, 0.05])
df_train, df_test, X217 = binarize_limits('X217', df_train, df_test, [-0.19, 0.1])
df_train, df_test, X221 = binarize_limits('X221', df_train, df_test, [-0.12, 0.14])
df_train, df_test, X228 = binarize_limits('X228', df_train, df_test, [-0.14, -0.12, 0.1])
df_train, df_test, X267 = binarize_limits('X267', df_train, df_test, [-0.11, 0.24, 0])
df_train, df_test, X280 = binarize_limits('X280', df_train, df_test, [0.38])
df_train, df_test, X283 = binarize_limits('X283', df_train, df_test, [0.1, 0.4])
df_train, df_test, X283 = binarize_limits('X289', df_train, df_test, [0.1, 0.4])
df_train, df_test, X295 = binarize_limits('X295', df_train, df_test, [-0.3, 0.3])
df_train, df_test, X299 = binarize_limits('X299', df_train, df_test, [0.25])
df_train, df_test, X307 = binarize_limits('X307', df_train, df_test, [-0.02, -0.1, -0.06])
df_train, df_test, X308 = binarize_limits('X308', df_train, df_test, [-0.1, 0.075])
#df_train, df_test, X317 = binarize_limits('X317', df_train, df_test, [])

print('number of binary features before = %d' % len(df_train.columns))

# binary valued feature selection
selected_features = stump_selection(0.0012, df_train) # 0.001 -> 30 stumps
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('number of binary features after = %d' % len(df_train.columns))

# fix feature names
X009 = fix_names(X009, selected_features)
X021 = fix_names(X021, selected_features)
X039 = fix_names(X039, selected_features)
X044 = fix_names(X044, selected_features)
X046 = fix_names(X046, selected_features)
X057 = fix_names(X057, selected_features)
X071 = fix_names(X071, selected_features)
X081 = fix_names(X081, selected_features)
X085 = fix_names(X085, selected_features)
X096 = fix_names(X096, selected_features)
X136 = fix_names(X136, selected_features)
X183 = fix_names(X183, selected_features)
X200 = fix_names(X200, selected_features)
X207 = fix_names(X207, selected_features)
X211 = fix_names(X211, selected_features)
X217 = fix_names(X217, selected_features)
X221 = fix_names(X221, selected_features)
X228 = fix_names(X228, selected_features)
X267 = fix_names(X267, selected_features)
X280 = fix_names(X280, selected_features)
X283 = fix_names(X283, selected_features)
X295 = fix_names(X295, selected_features)
X299 = fix_names(X299, selected_features)
X307 = fix_names(X307, selected_features)
X308 = fix_names(X308, selected_features)
#X317 = fix_names(X317, selected_features)


# split into test and validation
df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=0, stratify=df_test['class'])

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
    'X021': X021,
    'X039': X039,
    'X044': X044,
    'X046': X046,
    'X057': X057,
    'X071': X071,
    'X081': X081,
    'X085': X085,
    'X096': X096,
    'X136': X136,
    'X183': X183,
    'X200': X200,
    'X207': X207,
    'X211': X211,
    'X217': X217,
    'X221': X221,
    'X228': X228,
    'X267': X267,
    'X280': X280,
    'X283': X283,
    'X295': X295,
    'X299': X299,
    'X307': X307,
    'X308': X308,
    #'X317': X317,
}

# preparing data
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_valid = df_valid.iloc[:,1:].values
y_valid = df_valid.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
data_headers = df_train.columns

# cross validating
rm = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)
cv_result, build_times, opt_gaps = riskslim_cv(n_folds,rm, X_train, y_train)

# fitting model
rm.fit(X_train,y_train)

# validation metrics
print('validating:')
y_pred = rm.predict(X_valid)
print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_valid, y_pred))
print("optimality_gap = %.3f" % rm.model_info['optimality_gap'])
print(sec2time(rm.model_info['solver_time']))

# testing metrics
print('Testing:')
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

