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

from preprocess import binarize_greater, binarize_interval, binarize_category, binarize_sex
from prettytable import PrettyTable

# setup variables
file = 'risk_slim/hrt.csv'
output_file = open('result.txt', 'w+')
test_size = 0.2
n_folds = 3
max_runtime = 1.0

os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + 'diabetes.csv'

# data preprocessing
df  = pd.read_csv(path, float_precision='round_trip')

X = df.iloc[:, 0:-1].values
y = df.iloc[:,-1].values
y[y == -1] = 0

# 0 to nan
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, np.nan)

# imputate nan values
imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputed_df = pd.DataFrame(imputer.fit_transform(df.values))
imputed_df.columns = df.columns
imputed_df.index = df.index

# binarizing features
imputed_df, pregnancies_features = binarize_greater('Pregnancies', 0.3, imputed_df, 2)
imputed_df, glucose_features = binarize_greater('Glucose', 0.1, imputed_df, 10)
imputed_df, blood_pressure_features = binarize_greater('BloodPressure', 0.2, imputed_df, 5)
imputed_df, skin_thickness_features = binarize_greater('SkinThickness', 0.2, imputed_df, 5)
imputed_df, insulin_features = binarize_greater('Insulin', 0.15, imputed_df, 10)
imputed_df, bmi_features = binarize_greater('BMI', 0.15, imputed_df, 5)
imputed_df, age_features = binarize_greater('Age', 0.15, imputed_df, 5)

imputed_df['Outcome'] = imputed_df['Outcome'].astype(int)
imputed_df.drop('Outcome', axis=1, inplace=True)
imputed_df.insert(0, "Outcome", y, True)

# saving processed data
imputed_df.to_csv('risk_slim/hrt.csv', sep=',', index=False,header=True)

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
df_in  = pd.read_csv(file, float_precision='round_trip')
X = df_in.iloc[:, 1:].values
y = df_in.iloc[:,0].values

X, y = shuffle(X, y, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

rm = RiskModel(data_headers=df_in.columns.values, params=params, settings=settings, op_constraints=op_constraints)

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

# cross validation
# scores_risk = cross_val_score(rm, X_train, y_train, scoring="accuracy", cv=n_folds)
# print("Risk Slim Cross Validation: %0.2f (+/- %0.2f)" % (scores_risk.mean(), scores_risk.std() * 2))

