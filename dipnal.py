import os
import numpy as np
import pandas as pd
from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa

# my own methods
from diptools import build_model, split_data
from riskmodel import RiskModel
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
"""
# data
os.chdir('..')
data_name = "breastcancer"                                    # name of the data
data_dir = os.getcwd() + '/risk-slim/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'          # csv file for the dataset
sample_weights_csv_file = None                              # csv file of sample weights for the dataset (optional)

# problem parameters
max_coefficient = 5                                         # value of largest/smallest coefficient
max_L0_value = 5                                            # maximum model size (set as float(inf))
max_offset = 50                                             # maximum value of offset parameter (optional)
c0_value = 1e-6                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
w_pos = 1.00                                                # relative weight on examples with y = +1; w_neg = 1.00 (optional)

# split data in train and test
split_data(data_csv_file,5)

# load data from disk
data = load_data_from_csv(dataset_csv_file = 'train_file.csv', sample_weights_csv_file = sample_weights_csv_file)
test_data = load_data_from_csv(dataset_csv_file = 'test_file.csv', sample_weights_csv_file = sample_weights_csv_file)

# create coefficient set and set the value of the offset parameter
coef_set = CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0)
conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
max_offset = min(max_offset, conservative_offset)
coef_set['(Intercept)'].ub = max_offset
coef_set['(Intercept)'].lb = -max_offset



constraints = {
    'L0_min': 0,
    'L0_max': max_L0_value,
    'coef_set':coef_set,
}


# major settings (see riskslim_ex_02_complete for full set of options)
settings = {
    # Problem Parameters
    'c0_value': c0_value,
    'w_pos': w_pos,
    #
    # LCPA Settings
    'max_runtime': 30.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # print CPLEX progress on screen
    'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')
    #
    # LCPA Improvements
    'round_flag': True,                                # round continuous solutions with SeqRd
    'polish_flag': True,                               # polish integer feasible solutions with DCD
    'chained_updates_flag': True,                      # use chained updates
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
    #
    # Initialization
    'initialization_flag': True,                       # use initialization procedure
    'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
    'init_max_coefficient_gap': 0.49,
    #
    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}
#print("--------TRAIN MODEL USING LATTICE CPA--------")
# train model using lattice_cpa
model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)

print("--------PRINT MODEL INFO WITH KEY RESULTS--------")

rm = build_model(model_info['solution'], data)
#print(rm.predict_single([3,7,7,4,4,9,4,8,1]))

print_model(model_info['solution'], data)
rm.test_model(data)
rm.test_model(test_data) """

# variables
testSize = 0.2
os.chdir('..')
path = os.getcwd() + '/risk-slim/examples/data/' + 'mammo_data.csv'


# read and prepocess data
df_in  = pd.read_csv(path, float_precision='round_trip')
X = df_in.iloc[:, :-1].values
y = df_in.iloc[:,0].values
X, y = shuffle(X, y, random_state=1)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=0)

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
    #
    # LCPA Settings
    'max_runtime': 30.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # print CPLEX progress on screen
    'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')
    #
    # LCPA Improvements
    'round_flag': True,                                # round continuous solutions with SeqRd
    'polish_flag': True,                               # polish integer feasible solutions with DCD
    'chained_updates_flag': True,                      # use chained updates
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
    #
    # Initialization
    'initialization_flag': True,                       # use initialization procedure
    'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
    'init_max_coefficient_gap': 0.49,
    #
    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}


rm = RiskModel(data_headers=df_in.columns.values, params=params, settings=settings)
rm.fit_transform(X_train,y_train)
y_pred = rm.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))