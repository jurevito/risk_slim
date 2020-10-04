import os
import numpy as np
import pandas as pd
from riskmodel import RiskModel

from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# setup variables
os.chdir('..')
file = 'breastcancer'
path = os.getcwd() + '/risk-slim/examples/data/' + file + '_data.csv'
test_size = 0.2

# read data
df_in  = pd.read_csv(path, float_precision='round_trip')
X = df_in.iloc[:, 1:].values
y = df_in.iloc[:,0].values
X, y = shuffle(X, y, random_state=1)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

params = {
    'max_coefficient' : 5,
    'max_L0_value' : 5,
    'max_offset' : 50,
    'c0_value' : 1e-6,
    'w_pos' : 1.00
}

settings = {

    'c0_value': params['c0_value'],
    'w_pos': params['w_pos'],

    # LCPA Settings
    'max_runtime': 30.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # print CPLEX progress on screen
    'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')

    # LCPA Improvements
    'round_flag': True,                                # round continuous solutions with SeqRd
    'polish_flag': True,                               # polish integer feasible solutions with DCD
    'chained_updates_flag': True,                      # use chained updates
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding

    # Initialization
    'initialization_flag': True,                       # use initialization procedure
    'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
    'init_max_coefficient_gap': 0.49,

    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}

# train model and make prediction
rm = RiskModel(data_headers=df_in.columns.values, params=params, settings=settings)
rm.fit_transform(X_train,y_train)
y_pred = rm.predict(X_test)

# print metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.2f" % accuracy_score(y_test, y_pred))