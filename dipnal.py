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
file = 'diabetes.csv'
test_size = 0.2
n_folds = 5
max_runtime = 3600.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file
df  = pd.read_csv(path, float_precision='round_trip')

# move outcome at beginning
outcome_values = df['Outcome'].values
df = df.drop(['Outcome'], axis=1)
df.insert(0, 'Outcome', outcome_values, True)

# 0 to nan
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, np.nan)

# split data
df = shuffle(df, random_state=1)
df_train, df_test = train_test_split(df, test_size=test_size, random_state=0, stratify=df['Outcome'])

# data imputation
tmp1 = df_train
tmp2 = df_test
imputer = KNNImputer(n_neighbors=2, weights="uniform")
df_train = pd.DataFrame(imputer.fit_transform(df_train))
df_test = pd.DataFrame(imputer.transform(df_test))

df_train.columns = tmp1.columns
df_train.index = tmp1.index
df_test.columns = tmp2.columns
df_test.index = tmp2.index

# binarizing train and test set
df_train, df_test, Pregnancies = binarize_limits('Pregnancies', df_train, df_test, [7, 13])
df_train, df_test, Glucose = binarize_limits('Glucose', df_train, df_test, [155, 166, 121, 130])
df_train, df_test, BloodPressure = binarize_limits('BloodPressure', df_train, df_test, [55, 92, 100])
df_train, df_test, SkinThickness = binarize_limits('SkinThickness', df_train, df_test, [15, 22, 32])
df_train, df_test, Insulin = binarize_limits('Insulin', df_train, df_test, [80, 126, 330])
df_train, df_test, BMI = binarize_limits('BMI', df_train, df_test, [25, 45])
df_train, df_test, DiabetesPedigreeFunction = binarize_limits('DiabetesPedigreeFunction', df_train, df_test, [1.35, 0.19])
df_train, df_test, Age = binarize_limits('Age', df_train, df_test, [63, 42, 23, 53])

print('1. n_features = %d' % len(df_train.columns))

# binary valued feature selection
selected_features = stump_selection(0.8, df_train)
df_train = df_train[selected_features]
df_test = df_test[selected_features]

print('2. n_features = %d' % len(df_train.columns))

Pregnancies = fix_names(Pregnancies, selected_features)
Glucose = fix_names(Glucose, selected_features)
BloodPressure = fix_names(BloodPressure, selected_features)
SkinThickness = fix_names(SkinThickness, selected_features)
Insulin = fix_names(Insulin, selected_features)
BMI = fix_names(BMI, selected_features)
DiabetesPedigreeFunction = fix_names(DiabetesPedigreeFunction, selected_features)
Age = fix_names(Age, selected_features)

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
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age,
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
