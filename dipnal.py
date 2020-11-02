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

from preprocess import binarize_greater, binarize_interval, binarize_category, binarize_sex, binarize
from prettytable import PrettyTable

# setup variables
output_file = open('result.txt', 'w+')
file = 'diabetes'
test_size = 0.2
n_folds = 3
max_runtime = 3.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + file + '.csv'
df  = pd.read_csv(path, float_precision='round_trip')

# taget variable
y = df.iloc[:,-1].values
y[y == -1] = 0
df.drop('Outcome', axis=1, inplace=True)
df.insert(0, 'Outcome', y, True)

# 0 to nan
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, np.nan)

# split data
df = shuffle(df, random_state=1)
train_df, test_df = train_test_split(df, test_size=test_size, random_state=0)

# imputate train set
imputer = KNNImputer(n_neighbors=4, weights="uniform")
imputed_train = pd.DataFrame(imputer.fit_transform(train_df.values))
imputed_train.columns = train_df.columns
imputed_train.index = train_df.index

# imputate test set
imputed_test = pd.DataFrame(imputer.transform(test_df.values))
imputed_test.columns = test_df.columns
imputed_test.index = test_df.index

# binarizing train set
imputed_train, pregnancies_features, pregnancies_limits = binarize_greater('Pregnancies', 0.15, imputed_train, 2)
imputed_train, glucose_features, glucose_limits = binarize_greater('Glucose', 0.08, imputed_train, 15)
imputed_train, blood_pressure_features, blood_pressure_limits = binarize_greater('BloodPressure', 0.15, imputed_train, 5)
imputed_train, skin_thickness_features, skin_thickness_limits = binarize_greater('SkinThickness', 0.2, imputed_train, 5)
imputed_train, insulin_features, insulin_limits = binarize_greater('Insulin', 0.1, imputed_train, 10)
imputed_train, bmi_features, bmi_limits = binarize_greater('BMI', 0.1, imputed_train, 4)
imputed_train, age_features, age_limits = binarize_greater('Age', 0.15, imputed_train, 5)

# binarizing test set
imputed_test = binarize('Pregnancies', pregnancies_limits, imputed_test)
imputed_test = binarize('Glucose', glucose_limits, imputed_test)
imputed_test = binarize('BloodPressure', blood_pressure_limits, imputed_test)
imputed_test = binarize('SkinThickness', skin_thickness_limits, imputed_test)
imputed_test = binarize('Insulin', insulin_limits, imputed_test)
imputed_test = binarize('BMI', bmi_limits, imputed_test)
imputed_test = binarize('Age', age_limits, imputed_test)

# saving processed data
imputed_train.to_csv('risk_slim/train_data.csv', sep=',', index=False,header=True)
imputed_test.to_csv('risk_slim/test_data.csv', sep=',', index=False,header=True)

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
    'pregnancies_features' : pregnancies_features,
    'glucose_features' : glucose_features,
    'blood_pressure_features' : blood_pressure_features,
    'skin_thickness_features' : skin_thickness_features,
    'insulin_features' : insulin_features,
    'bmi_features' : bmi_features,
    'age_features' : age_features,
}

# preparing data
df_train  = pd.read_csv('risk_slim/train_data.csv', float_precision='round_trip')
df_test  = pd.read_csv('risk_slim/test_data.csv', float_precision='round_trip')

X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:,0].values

rm = RiskModel(data_headers=df_train.columns.values, params=params, settings=settings, op_constraints=op_constraints)

# fitting model
rm.fit(X_train,y_train)
y_pred = rm.predict(X_test)

# print metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))

# roc auc
y_roc_pred = rm.decision_function(X_test)
fpr_risk, tpr_risk, treshold_risk = roc_curve(y_test, y_roc_pred)
auc_risk = auc(fpr_risk, tpr_risk)

# saving results and model info
table1 = PrettyTable(["Parameter","Value"])
table1.add_row(["Accuracy", "%0.2f" % accuracy_score(y_test, y_pred)])
table1.add_row(["AUC", "%0.2f" % auc_risk])
table1.add_row(["Bin. Method", "all greater, sex split"])
table1.add_row(["Run Time", round(rm.model_info['solver_time'],1)])
table1.add_row(["Max Time", max_runtime])
table1.add_row(["Max Features", params['max_L0_value']])
table1.add_row(["Optimality Gap", round(rm.model_info['optimality_gap'],3)])

output_file.write("\n\n!--- MODEL INFO ---!\n")
output_file.write(str(table1))
output_file.close()

# plotting roc curve
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(fpr_risk, tpr_risk, linestyle='-', label='Risk Slim (auc = %0.2f)' % auc_risk)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

