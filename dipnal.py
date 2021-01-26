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

from preprocess import binarize_limits, sec2time, riskslim_cv, find_treshold_index, stump_selection, fix_names, print_cv_results, binarize_sex, ebm_binarization, auto_selection, auto_select
from prettytable import PrettyTable

from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier
import time

def OnevsRest_riskslim(df_train, df_test, params, settings):

    predictions = np.zeros((1,len(df_test.index)))

    for target_class in range(len(set(df_train['class'].values))):
        print('class is %d' % target_class)

        df_train_bin = df_train.copy()
        df_test_bin = df_test.copy()
        print(df_train_bin)
        #df_train_bin = df_train
        #df_test_bin = df_test

        #print('1. number of ones in class (%d) = %d' % (target_class, len(df_train_bin[df_train_bin['class'] == target_class])))

        df_train_bin['class'].values[df_train_bin['class'].values == target_class] = -1
        df_train_bin['class'].values[(df_train_bin['class'].values != target_class) & (df_train_bin['class'].values != -1)] = 0
        df_train_bin['class'].values[df_train_bin['class'].values == -1] = 1
        df_test_bin['class'].values[df_test_bin['class'].values == target_class] = -1
        df_test_bin['class'].values[(df_test_bin['class'].values != target_class) & (df_test_bin['class'].values != -1)] = 0
        df_test_bin['class'].values[df_test_bin['class'].values == -1] = 1

        #print('2. number of ones in class (%d) = %d' % (target_class, len(df_train_bin[df_train_bin['class'] == 1])))

        selected_features = auto_select(30, df_train_bin)
        df_train_bin = df_train_bin[selected_features]
        df_test_bin = df_test_bin[selected_features]

        df_train_bin, df_test_bin, feature_dict = ebm_binarization(df_train_bin, df_test_bin, 2, type='exclude', feature_names=['X278'])
        df_train_bin, df_test_bin, feature_dict = auto_selection(18, df_train_bin, df_test_bin, feature_dict)
        op_constraints = feature_dict

        X_train = df_train_bin.iloc[:,1:].values
        y_train = df_train_bin.iloc[:,0].values
        X_test = df_test_bin.iloc[:,1:].values
        y_test = df_test_bin.iloc[:,0].values
        data_headers = df_train_bin.columns

        rm = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)
        rm.fit(X_train,y_train)
        y_pred = rm.predict_proba(X_test)
        predictions = np.concatenate((predictions, np.array([y_pred])), axis=0)


    y_pred = np.zeros(len(df_test_bin))
    for i in range(len(y_pred)):
        arr = predictions[:,i]
        y_pred[i] = np.where(arr == np.amax(arr))[0][0] - 1

    print(predictions)

    return y_pred

# setup variables
output_file = open('result.txt', 'w+')
file = 'Disease-MUL.hd5'
test_size = 0.2
n_folds = 5
max_runtime = 10.0
is_multiclass = True

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


params = {
    'max_coefficient' : 6,                    # value of largest/smallest coefficient
    'max_L0_value' : 3,                       # maximum model size (set as float(inf))
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
"""op_constraints = {
    'age_features': age_features,
    'trestbps_features': trestbps_features,
    'chol_features': chol_features,
    'thalach_features': thalach_features,
    'oldpeak_features': oldpeak_features,
    'sex_features': sex_features,
}"""

# preparing data
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
data_headers = df_train.columns


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
    'f1_macro': [],
    'f1_micro': [],
}

for train_index, valid_index in kf.split(X_train, y_train):

    X_train_cv = X_train[train_index]
    y_train_cv = y_train[train_index]

    X_valid_cv = X_train[valid_index]
    y_valid_cv = y_train[valid_index]

    y_pred = OnevsRest_riskslim(df_train[train_index], df_train[valid_index], params, settings)

    results['accuracy'].append(accuracy_score(y_valid_cv, y_pred))
    results['f1_macro'].append(f1_score(y_valid_cv, y_pred, average='macro'))
    results['f1_micro'].append(f1_score(y_valid_cv, y_pred, average='micro'))

    if not is_multiclass:

        results['recall_1'].append(recall_score(y_valid_cv, y_pred, pos_label=1))
        results['recall_0'].append(recall_score(y_valid_cv, y_pred, pos_label=0))
        results['precision_1'].append(precision_score(y_valid_cv, y_pred, pos_label=1))
        results['precision_0'].append(precision_score(y_valid_cv, y_pred, pos_label=0))
        results['f1_1'].append(f1_score(y_valid_cv, y_pred, pos_label=1))
        results['f1_0'].append(f1_score(y_valid_cv, y_pred, pos_label=0))

        results['build_times'].append(rm.model_info['solver_time'])
        results['optimality_gaps'].append(rm.model_info['optimality_gap'])"""


# print cv results
"""print(results['accuracy'])
print_cv_results(results)"""

# printing metrics
print('Testing results:')
y_pred = OnevsRest_riskslim(df_train, df_test, params, settings)

print(y_pred)
print(y_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))
"""print("optimality_gap = %.3f" % rm.model_info['optimality_gap'])
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
plt.show()"""
