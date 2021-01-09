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

output_file = open('result.txt', 'w+')
file = 'Groups_knn2.h5'
test_size = 0.2
n_folds = 5
max_runtime = 3600.0

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
df_train = df_train.drop(['X152','X237','X294','X236','X076','X065','X105','X071','X085'], axis=1)
df_test = df_test.drop(['X152','X237','X294','X236','X076','X065','X105','X071','X085'], axis=1)


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

# !---- Building 9 Models ---- !
# (0) model
df_train_0 = df_train.copy()
df_test_0 = df_test.copy()

target_class = 0
df_train_0['class'].values[df_train_0['class'].values == target_class] = -1
df_train_0['class'].values[(df_train_0['class'].values != target_class) & (df_train_0['class'].values != -1)] = 0
df_train_0['class'].values[df_train_0['class'].values == -1] = 1
df_test_0['class'].values[df_test_0['class'].values == target_class] = -1
df_test_0['class'].values[(df_test_0['class'].values != target_class) & (df_test_0['class'].values != -1)] = 0
df_test_0['class'].values[df_test_0['class'].values == -1] = 1

selected_features = stump_selection(0.01, df_train_0)
df_train_0 = df_train_0[selected_features]
df_test_0 = df_test_0[selected_features]

df_train_0, df_test_0, X018 = binarize_limits('X018', df_train_0, df_test_0, [0.43])
df_train_0, df_test_0, X021 = binarize_limits('X021', df_train_0, df_test_0, [0.36, 0.2])
df_train_0, df_test_0, X046 = binarize_limits('X046', df_train_0, df_test_0, [0.4])
df_train_0, df_test_0, X063 = binarize_limits('X063', df_train_0, df_test_0, [0.1, 0.2])
df_train_0, df_test_0, X208 = binarize_limits('X208', df_train_0, df_test_0, [0.3])
df_train_0, df_test_0, X213 = binarize_limits('X213', df_train_0, df_test_0, [0.22, -0.22])
df_train_0, df_test_0, X248 = binarize_limits('X248', df_train_0, df_test_0, [-0.1, 0.07])
df_train_0, df_test_0, X267 = binarize_limits('X267', df_train_0, df_test_0, [0.23])
df_train_0, df_test_0, X280 = binarize_limits('X280', df_train_0, df_test_0, [0.75, 0.38])
df_train_0, df_test_0, X281 = binarize_limits('X281', df_train_0, df_test_0, [0.3, 0])

selected_features = stump_selection(0.1, df_train_0)
df_train_0 = df_train_0[selected_features]
df_test_0 = df_test_0[selected_features]

X018 = fix_names(X018, selected_features)
X021 = fix_names(X021, selected_features)
X046 = fix_names(X046, selected_features)
X063 = fix_names(X063, selected_features)
X208 = fix_names(X208, selected_features)
X213 = fix_names(X213, selected_features)
X248 = fix_names(X248, selected_features)
X267 = fix_names(X267, selected_features)
X280 = fix_names(X280, selected_features)
X281 = fix_names(X281, selected_features)

op_constraints = {
    'X018': X018,
    'X021': X021,
    'X046': X046,
    'X063': X063,
    'X208': X208,
    'X213': X213,
    'X248': X248,
    'X267': X267,
    'X280': X280,
    'X281': X281,
}

data_headers = df_train_0.columns
rm0 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (1) model
df_train_1 = df_train.copy()
df_test_1 = df_test.copy()

target_class = 1
df_train_1['class'].values[df_train_1['class'].values == target_class] = -1
df_train_1['class'].values[(df_train_1['class'].values != target_class) & (df_train_1['class'].values != -1)] = 0
df_train_1['class'].values[df_train_1['class'].values == -1] = 1
df_test_1['class'].values[df_test_1['class'].values == target_class] = -1
df_test_1['class'].values[(df_test_1['class'].values != target_class) & (df_test_1['class'].values != -1)] = 0
df_test_1['class'].values[df_test_1['class'].values == -1] = 1

selected_features = stump_selection(0.013, df_train_1)
df_train_1 = df_train_1[selected_features]
df_test_1 = df_test_1[selected_features]

df_train_1, df_test_1, X018 = binarize_limits('X018', df_train_1, df_test_1, [0.43, -0.1])
df_train_1, df_test_1, X046 = binarize_limits('X046', df_train_1, df_test_1, [0.23, 0.4])
df_train_1, df_test_1, X074 = binarize_limits('X074', df_train_1, df_test_1, [-0.18, 0.16])
df_train_1, df_test_1, X210 = binarize_limits('X210', df_train_1, df_test_1, [0.26])
df_train_1, df_test_1, X267 = binarize_limits('X267', df_train_1, df_test_1, [0.23])
df_train_1, df_test_1, X269 = binarize_limits('X269', df_train_1, df_test_1, [-0.16, -0.1])
df_train_1, df_test_1, X280 = binarize_limits('X280', df_train_1, df_test_1, [-0.06, 0.425])
df_train_1, df_test_1, X281 = binarize_limits('X281', df_train_1, df_test_1, [0, 3.5])
df_train_1, df_test_1, X317 = binarize_limits('X317', df_train_1, df_test_1, [0.5])

"""selected_features = stump_selection(0.2, df_train_1)
df_train_1 = df_train_1[selected_features]
df_test_1 = df_test_1[selected_features]"""

X018 = fix_names(X018, selected_features)
X046 = fix_names(X046, selected_features)
X074 = fix_names(X074, selected_features)
X210 = fix_names(X210, selected_features)
X267 = fix_names(X267, selected_features)
X269 = fix_names(X269, selected_features)
X280 = fix_names(X280, selected_features)
X281 = fix_names(X281, selected_features)
X317 = fix_names(X317, selected_features)

op_constraints = {
    'X018': X018,
    'X046': X046,
    'X074': X074,
    'X210': X210,
    'X267': X267,
    'X269': X269,
    'X280': X280,
    'X281': X281,
    'X317': X317,
}

data_headers = df_train_1.columns
rm1 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (2) model
df_train_2 = df_train.copy()
df_test_2 = df_test.copy()

target_class = 2
df_train_2['class'].values[df_train_2['class'].values == target_class] = -1
df_train_2['class'].values[(df_train_2['class'].values != target_class) & (df_train_2['class'].values != -1)] = 0
df_train_2['class'].values[df_train_2['class'].values == -1] = 1
df_test_2['class'].values[df_test_2['class'].values == target_class] = -1
df_test_2['class'].values[(df_test_2['class'].values != target_class) & (df_test_2['class'].values != -1)] = 0
df_test_2['class'].values[df_test_2['class'].values == -1] = 1

selected_features = stump_selection(0.0008, df_train_2)
df_train_2 = df_train_2[selected_features]
df_test_2 = df_test_2[selected_features]

df_train_2, df_test_2, X009 = binarize_limits('X009', df_train_2, df_test_2, [0.05])
df_train_2, df_test_2, X021 = binarize_limits('X021', df_train_2, df_test_2, [-0.17, -0.06])
df_train_2, df_test_2, X046 = binarize_limits('X046', df_train_2, df_test_2, [0.02, 0.22])
df_train_2, df_test_2, X136 = binarize_limits('X136', df_train_2, df_test_2, [0.32, 0.5])
df_train_2, df_test_2, X221 = binarize_limits('X221', df_train_2, df_test_2, [0.03, 0.22])
df_train_2, df_test_2, X228 = binarize_limits('X228', df_train_2, df_test_2, [-0.1, 0.14])
df_train_2, df_test_2, X267 = binarize_limits('X267', df_train_2, df_test_2, [0.2, -0.11])
df_train_2, df_test_2, X295 = binarize_limits('X295', df_train_2, df_test_2, [-0.36])
df_train_2, df_test_2, X317 = binarize_limits('X317', df_train_2, df_test_2, [0.5])

selected_features = stump_selection(0.1, df_train_2)
df_train_2 = df_train_2[selected_features]
df_test_2 = df_test_2[selected_features]

X009 = fix_names(X009, selected_features)
X021 = fix_names(X021, selected_features)
X046 = fix_names(X046, selected_features)
X136 = fix_names(X136, selected_features)
X221 = fix_names(X221, selected_features)
X228 = fix_names(X228, selected_features)
X267 = fix_names(X267, selected_features)
X295 = fix_names(X295, selected_features)
X317 = fix_names(X317, selected_features)

op_constraints = {
    'X009': X009,
    'X021': X021,
    'X046': X046,
    'X136': X136,
    'X221': X221,
    'X228': X228,
    'X267': X267,
    'X295': X295,
    'X317': X317,
}


data_headers = df_train_2.columns
rm2 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (3) model
df_train_3 = df_train.copy()
df_test_3 = df_test.copy()

target_class = 3
df_train_3['class'].values[df_train_3['class'].values == target_class] = -1
df_train_3['class'].values[(df_train_3['class'].values != target_class) & (df_train_3['class'].values != -1)] = 0
df_train_3['class'].values[df_train_3['class'].values == -1] = 1
df_test_3['class'].values[df_test_3['class'].values == target_class] = -1
df_test_3['class'].values[(df_test_3['class'].values != target_class) & (df_test_3['class'].values != -1)] = 0
df_test_3['class'].values[df_test_3['class'].values == -1] = 1

selected_features = stump_selection(0.0008, df_train_3)
df_train_3 = df_train_3[selected_features]
df_test_3 = df_test_3[selected_features]

df_train_3, df_test_3, X009 = binarize_limits('X009', df_train_3, df_test_3, [-0.3, 0.02])
df_train_3, df_test_3, X021 = binarize_limits('X021', df_train_3, df_test_3, [-0.12, 0.2])
df_train_3, df_test_3, X046 = binarize_limits('X046', df_train_3, df_test_3, [-0.21, -0.19])
df_train_3, df_test_3, X221 = binarize_limits('X221', df_train_3, df_test_3, [-0.12, 0.22])
df_train_3, df_test_3, X267 = binarize_limits('X267', df_train_3, df_test_3, [0.24, -0.02])
df_train_3, df_test_3, X283 = binarize_limits('X283', df_train_3, df_test_3, [0.1, 0.4])
df_train_3, df_test_3, X295 = binarize_limits('X295', df_train_3, df_test_3, [-0.32, 0.5])
df_train_3, df_test_3, X307 = binarize_limits('X307', df_train_3, df_test_3, [0, 0.22])
df_train_3, df_test_3, X308 = binarize_limits('X308', df_train_3, df_test_3, [0.23, 0.46])

selected_features = stump_selection(0.2, df_train_3)
df_train_3 = df_train_3[selected_features]
df_test_3 = df_test_3[selected_features]

X009 = fix_names(X009, selected_features)
X021 = fix_names(X021, selected_features)
X046 = fix_names(X046, selected_features)
X221 = fix_names(X221, selected_features)
X267 = fix_names(X267, selected_features)
X283 = fix_names(X283, selected_features)
X295 = fix_names(X295, selected_features)
X307 = fix_names(X307, selected_features)
X308 = fix_names(X308, selected_features)


op_constraints = {
    'X009': X009,
    'X021': X021,
    'X046': X046,
    'X221': X221,
    'X267': X267,
    'X283': X283,
    'X295': X295,
    'X307': X307,
    'X308': X308,
}

data_headers = df_train_3.columns
rm3 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (4) model
df_train_4 = df_train.copy()
df_test_4 = df_test.copy()

target_class = 4
df_train_4['class'].values[df_train_4['class'].values == target_class] = -1
df_train_4['class'].values[(df_train_4['class'].values != target_class) & (df_train_4['class'].values != -1)] = 0
df_train_4['class'].values[df_train_4['class'].values == -1] = 1
df_test_4['class'].values[df_test_4['class'].values == target_class] = -1
df_test_4['class'].values[(df_test_4['class'].values != target_class) & (df_test_4['class'].values != -1)] = 0
df_test_4['class'].values[df_test_4['class'].values == -1] = 1

selected_features = stump_selection(0.0025, df_train_4)
df_train_4 = df_train_4[selected_features]
df_test_4 = df_test_4[selected_features]

df_train_4, df_test_4, X021 = binarize_limits('X021', df_train_4, df_test_4, [-0.13, -0.57])
df_train_4, df_test_4, X041 = binarize_limits('X041', df_train_4, df_test_4, [0.4])
df_train_4, df_test_4, X045 = binarize_limits('X045', df_train_4, df_test_4, [0.28, -0.05])
df_train_4, df_test_4, X069 = binarize_limits('X069', df_train_4, df_test_4, [0.28, 0.32])
df_train_4, df_test_4, X136 = binarize_limits('X136', df_train_4, df_test_4, [-0.004])
df_train_4, df_test_4, X147 = binarize_limits('X147', df_train_4, df_test_4, [-0.008])
df_train_4, df_test_4, X214 = binarize_limits('X214', df_train_4, df_test_4, [0.05, 0.24])
df_train_4, df_test_4, X307 = binarize_limits('X307', df_train_4, df_test_4, [0.3, -0.11])
df_train_4, df_test_4, X317 = binarize_limits('X317', df_train_4, df_test_4, [0.5])

selected_features = stump_selection(0.2, df_train_4)
df_train_4 = df_train_4[selected_features]
df_test_4 = df_test_4[selected_features]

X021 = fix_names(X021, selected_features)
X041 = fix_names(X041, selected_features)
X045 = fix_names(X045, selected_features)
X069 = fix_names(X069, selected_features)
X136 = fix_names(X136, selected_features)
X147 = fix_names(X147, selected_features)
X214 = fix_names(X214, selected_features)
X307 = fix_names(X307, selected_features)
X317 = fix_names(X317, selected_features)

op_constraints = {
    'X021': X021,
    'X041': X041,
    'X045': X045,
    'X069': X069,
    'X136': X136,
    'X147': X147,
    'X214': X214,
    'X307': X307,
    'X317': X317,
}

data_headers = df_train_4.columns
rm4 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (5) model
df_train_5 = df_train.copy()
df_test_5 = df_test.copy()

target_class = 5
df_train_5['class'].values[df_train_5['class'].values == target_class] = -1
df_train_5['class'].values[(df_train_5['class'].values != target_class) & (df_train_5['class'].values != -1)] = 0
df_train_5['class'].values[df_train_5['class'].values == -1] = 1
df_test_5['class'].values[df_test_5['class'].values == target_class] = -1
df_test_5['class'].values[(df_test_5['class'].values != target_class) & (df_test_5['class'].values != -1)] = 0
df_test_5['class'].values[df_test_5['class'].values == -1] = 1

selected_features = stump_selection(0.0015, df_train_5)
df_train_5 = df_train_5[selected_features]
df_test_5 = df_test_5[selected_features]

df_train_5, df_test_5, X039 = binarize_limits('X039', df_train_5, df_test_5, [0.31, 0.16])
df_train_5, df_test_5, X043 = binarize_limits('X043', df_train_5, df_test_5, [0.34, 0.2])
df_train_5, df_test_5, X045 = binarize_limits('X045', df_train_5, df_test_5, [-0.1, 0, 0.27])
df_train_5, df_test_5, X147 = binarize_limits('X147', df_train_5, df_test_5, [0.08, 0, 0.04])
df_train_5, df_test_5, X217 = binarize_limits('X217', df_train_5, df_test_5, [-0.17, -0.28, -0.21])
df_train_5, df_test_5, X268 = binarize_limits('X268', df_train_5, df_test_5, [0.15, 0.05, 0.08, -0.22])
df_train_5, df_test_5, X271 = binarize_limits('X271', df_train_5, df_test_5, [0.3, 0.47, -0.045])
df_train_5, df_test_5, X295 = binarize_limits('X295', df_train_5, df_test_5, [-0.35, 0.3])
df_train_5, df_test_5, X308 = binarize_limits('X308', df_train_5, df_test_5, [-0.12, -0.17])

selected_features = stump_selection(0.1, df_train_5)
df_train_5 = df_train_5[selected_features]
df_test_5 = df_test_5[selected_features]

X039 = fix_names(X039, selected_features)
X043 = fix_names(X043, selected_features)
X045 = fix_names(X045, selected_features)
X147 = fix_names(X147, selected_features)
X217 = fix_names(X217, selected_features)
X268 = fix_names(X268, selected_features)
X271 = fix_names(X271, selected_features)
X295 = fix_names(X295, selected_features)
X308 = fix_names(X308, selected_features)

op_constraints = {
    'X039': X039,
    'X043': X043,
    'X045': X045,
    'X147': X147,
    'X217': X217,
    'X268': X268,
    'X271': X271,
    'X295': X295,
    'X308': X308,
}

data_headers = df_train_5.columns
rm5 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (6) model
df_train_6 = df_train.copy()
df_test_6 = df_test.copy()

target_class = 6
df_train_6['class'].values[df_train_6['class'].values == target_class] = -1
df_train_6['class'].values[(df_train_6['class'].values != target_class) & (df_train_6['class'].values != -1)] = 0
df_train_6['class'].values[df_train_6['class'].values == -1] = 1
df_test_6['class'].values[df_test_6['class'].values == target_class] = -1
df_test_6['class'].values[(df_test_6['class'].values != target_class) & (df_test_6['class'].values != -1)] = 0
df_test_6['class'].values[df_test_6['class'].values == -1] = 1

selected_features = stump_selection(0.002, df_train_6)
df_train_6 = df_train_6[selected_features]
df_test_6 = df_test_6[selected_features]

df_train_6, df_test_6, X008 = binarize_limits('X008', df_train_6, df_test_6, [0.028])
df_train_6, df_test_6, X021 = binarize_limits('X021', df_train_6, df_test_6, [0, -0.56])
df_train_6, df_test_6, X041 = binarize_limits('X041', df_train_6, df_test_6, [0.24, -0.13])
df_train_6, df_test_6, X088 = binarize_limits('X088', df_train_6, df_test_6, [-0.2, -0.1, 0.11])
df_train_6, df_test_6, X172 = binarize_limits('X172', df_train_6, df_test_6, [0.42, 0.2, 0.055])
df_train_6, df_test_6, X215 = binarize_limits('X215', df_train_6, df_test_6, [-0.15, 0])
df_train_6, df_test_6, X217 = binarize_limits('X217', df_train_6, df_test_6, [0.19, -0.29])
df_train_6, df_test_6, X221 = binarize_limits('X221', df_train_6, df_test_6, [-0.35, -0.17])
df_train_6, df_test_6, X289 = binarize_limits('X289', df_train_6, df_test_6, [0.31])
df_train_6, df_test_6, X317 = binarize_limits('X317', df_train_6, df_test_6, [0.5])

selected_features = stump_selection(0.1, df_train_6)
df_train_6 = df_train_6[selected_features]
df_test_6 = df_test_6[selected_features]

X008 = fix_names(X008, selected_features)
X021 = fix_names(X021, selected_features)
X041 = fix_names(X041, selected_features)
X088 = fix_names(X088, selected_features)
X172 = fix_names(X172, selected_features)
X215 = fix_names(X215, selected_features)
X217 = fix_names(X217, selected_features)
X221 = fix_names(X221, selected_features)
X289 = fix_names(X289, selected_features)
X317 = fix_names(X317, selected_features)

op_constraints = {
    'X008': X008,
    'X021': X021,
    'X041': X041,
    'X088': X088,
    'X172': X172,
    'X215': X215,
    'X217': X217,
    'X221': X221,
    'X289': X289,
    'X317': X317,
}

data_headers = df_train_6.columns
rm6 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (7) model
df_train_7 = df_train.copy()
df_test_7 = df_test.copy()

target_class = 7
df_train_7['class'].values[df_train_7['class'].values == target_class] = -1
df_train_7['class'].values[(df_train_7['class'].values != target_class) & (df_train_7['class'].values != -1)] = 0
df_train_7['class'].values[df_train_7['class'].values == -1] = 1
df_test_7['class'].values[df_test_7['class'].values == target_class] = -1
df_test_7['class'].values[(df_test_7['class'].values != target_class) & (df_test_7['class'].values != -1)] = 0
df_test_7['class'].values[df_test_7['class'].values == -1] = 1

selected_features = stump_selection(0.003, df_train_7)
df_train_7 = df_train_7[selected_features]
df_test_7 = df_test_7[selected_features]

df_train_7, df_test_7, X037 = binarize_limits('X037', df_train_7, df_test_7, [0.32])
df_train_7, df_test_7, X057 = binarize_limits('X057', df_train_7, df_test_7, [-0.27, -0.05, -0.21])
df_train_7, df_test_7, X066 = binarize_limits('X066', df_train_7, df_test_7, [-0.06, -0.15])
df_train_7, df_test_7, X172 = binarize_limits('X172', df_train_7, df_test_7, [0.05, 0.01, -0.11])
df_train_7, df_test_7, X215 = binarize_limits('X215', df_train_7, df_test_7, [0.07, -0.2])
df_train_7, df_test_7, X216 = binarize_limits('X216', df_train_7, df_test_7, [0.19, 0.03])
df_train_7, df_test_7, X251 = binarize_limits('X251', df_train_7, df_test_7, [0.38, 0.19])
df_train_7, df_test_7, X267 = binarize_limits('X267', df_train_7, df_test_7, [-0.18, 0.22, 0.54])
df_train_7, df_test_7, X302 = binarize_limits('X302', df_train_7, df_test_7, [-0.22, -0.13, -0.09])

selected_features = stump_selection(0.1, df_train_7)
df_train_7 = df_train_7[selected_features]
df_test_7 = df_test_7[selected_features]

X037 = fix_names(X037, selected_features)
X057 = fix_names(X057, selected_features)
X066 = fix_names(X066, selected_features)
X172 = fix_names(X172, selected_features)
X215 = fix_names(X215, selected_features)
X216 = fix_names(X216, selected_features)
X251 = fix_names(X251, selected_features)
X267 = fix_names(X267, selected_features)
X302 = fix_names(X302, selected_features)

op_constraints = {
    'X037': X037,
    'X057': X057,
    'X066': X066,
    'X172': X172,
    'X215': X215,
    'X216': X216,
    'X251': X251,
    'X267': X267,
    'X302': X302,
}

data_headers = df_train_7.columns
rm7 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# (8) model
df_train_8 = df_train.copy()
df_test_8 = df_test.copy()

target_class = 8
df_train_8['class'].values[df_train_8['class'].values == target_class] = -1
df_train_8['class'].values[(df_train_8['class'].values != target_class) & (df_train_8['class'].values != -1)] = 0
df_train_8['class'].values[df_train_8['class'].values == -1] = 1
df_test_8['class'].values[df_test_8['class'].values == target_class] = -1
df_test_8['class'].values[(df_test_8['class'].values != target_class) & (df_test_8['class'].values != -1)] = 0
df_test_8['class'].values[df_test_8['class'].values == -1] = 1

selected_features = stump_selection(0.001, df_train_8)
df_train_8 = df_train_8[selected_features]
df_test_8 = df_test_8[selected_features]

df_train_8, df_test_8, X021 = binarize_limits('X021', df_train_8, df_test_8, [-0.55, -0.05])
df_train_8, df_test_8, X037 = binarize_limits('X037', df_train_8, df_test_8, [0, 0.13, 0.28])
df_train_8, df_test_8, X039 = binarize_limits('X039', df_train_8, df_test_8, [0.3, 0.15])
df_train_8, df_test_8, X119 = binarize_limits('X119', df_train_8, df_test_8, [-0.6])
df_train_8, df_test_8, X217 = binarize_limits('X217', df_train_8, df_test_8, [0.2, 0.28, 0.17])
df_train_8, df_test_8, X221 = binarize_limits('X221', df_train_8, df_test_8, [0.2])
df_train_8, df_test_8, X267 = binarize_limits('X267', df_train_8, df_test_8, [-0.15, 0.2, 0.53])
df_train_8, df_test_8, X268 = binarize_limits('X268', df_train_8, df_test_8, [0.2, 0.31])
df_train_8, df_test_8, X308 = binarize_limits('X308', df_train_8, df_test_8, [-0.05, 0.02, 0.4])

#selected_features = stump_selection(0.1, df_train_8)
#df_train_8 = df_train_8[selected_features]
#df_test_8 = df_test_8[selected_features]

X021 = fix_names(X021, selected_features)
X037 = fix_names(X037, selected_features)
X039 = fix_names(X039, selected_features)
X119 = fix_names(X119, selected_features)
X217 = fix_names(X217, selected_features)
X221 = fix_names(X221, selected_features)
X267 = fix_names(X267, selected_features)
X268 = fix_names(X268, selected_features)
X308 = fix_names(X308, selected_features)

op_constraints = {
    'X021': X021,
    'X037': X037,
    'X039': X039,
    'X119': X119,
    'X217': X217,
    'X221': X221,
    'X267': X267,
    'X268': X268,
    'X308': X308,
}

data_headers = df_train_8.columns
rm8 = RiskModel(data_headers=data_headers, params=params, settings=settings, op_constraints=op_constraints)

# !--------------------------- !
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

# cross validating
kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 1)
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
count = 0
for train_index, valid_index in kf.split(X_train, y_train):

    count += 1
    print("!--- %d fold of CV ---!" % count)

    # cv model 0
    X_train_0 = df_train_0.iloc[:,1:].values
    y_train_0 = df_train_0.iloc[:,0].values
    X_train_cv_0 = X_train_0[train_index]
    y_train_cv_0 = y_train_0[train_index]
    X_valid_cv_0 = X_train_0[valid_index]
    y_valid_cv_0 = y_train_0[valid_index]
    rm0.fit(X_train_cv_0, y_train_cv_0)
    y_pred_0 = rm0.predict_proba(X_valid_cv_0)

    # cv model 1
    X_train_1 = df_train_1.iloc[:,1:].values
    y_train_1 = df_train_1.iloc[:,0].values
    X_train_cv_1 = X_train_1[train_index]
    y_train_cv_1 = y_train_1[train_index]
    X_valid_cv_1 = X_train_1[valid_index]
    y_valid_cv_1 = y_train_1[valid_index]
    rm1.fit(X_train_cv_1, y_train_cv_1)
    y_pred_1 = rm1.predict_proba(X_valid_cv_1)

    # cv model 2
    X_train_2 = df_train_2.iloc[:,1:].values
    y_train_2 = df_train_2.iloc[:,0].values
    X_train_cv_2 = X_train_2[train_index]
    y_train_cv_2 = y_train_2[train_index]
    X_valid_cv_2 = X_train_2[valid_index]
    y_valid_cv_2 = y_train_2[valid_index]
    rm2.fit(X_train_cv_2, y_train_cv_2)
    y_pred_2 = rm2.predict_proba(X_valid_cv_2)

    # cv model 3
    X_train_3 = df_train_3.iloc[:,1:].values
    y_train_3 = df_train_3.iloc[:,0].values
    X_train_cv_3 = X_train_3[train_index]
    y_train_cv_3 = y_train_3[train_index]
    X_valid_cv_3 = X_train_3[valid_index]
    y_valid_cv_3 = y_train_3[valid_index]
    rm3.fit(X_train_cv_3, y_train_cv_3)
    y_pred_3 = rm3.predict_proba(X_valid_cv_3)

    # cv model 4
    X_train_4 = df_train_4.iloc[:,1:].values
    y_train_4 = df_train_4.iloc[:,0].values
    X_train_cv_4 = X_train_4[train_index]
    y_train_cv_4 = y_train_4[train_index]
    X_valid_cv_4 = X_train_4[valid_index]
    y_valid_cv_4 = y_train_4[valid_index]
    rm4.fit(X_train_cv_4, y_train_cv_4)
    y_pred_4 = rm4.predict_proba(X_valid_cv_4)

    # cv model 5
    X_train_5 = df_train_5.iloc[:,1:].values
    y_train_5 = df_train_5.iloc[:,0].values
    X_train_cv_5 = X_train_5[train_index]
    y_train_cv_5 = y_train_5[train_index]
    X_valid_cv_5 = X_train_5[valid_index]
    y_valid_cv_5 = y_train_5[valid_index]
    rm5.fit(X_train_cv_5, y_train_cv_5)
    y_pred_5 = rm5.predict_proba(X_valid_cv_5)

    # cv model 6
    X_train_6 = df_train_6.iloc[:,1:].values
    y_train_6 = df_train_6.iloc[:,0].values
    X_train_cv_6 = X_train_6[train_index]
    y_train_cv_6 = y_train_6[train_index]
    X_valid_cv_6 = X_train_6[valid_index]
    y_valid_cv_6 = y_train_6[valid_index]
    rm6.fit(X_train_cv_6, y_train_cv_6)
    y_pred_6 = rm6.predict_proba(X_valid_cv_6)

    # cv model 7
    X_train_7 = df_train_7.iloc[:,1:].values
    y_train_7 = df_train_7.iloc[:,0].values
    X_train_cv_7 = X_train_7[train_index]
    y_train_cv_7 = y_train_7[train_index]
    X_valid_cv_7 = X_train_7[valid_index]
    y_valid_cv_7 = y_train_7[valid_index]
    rm7.fit(X_train_cv_7, y_train_cv_7)
    y_pred_7 = rm7.predict_proba(X_valid_cv_7)

    # cv model 8
    X_train_8 = df_train_8.iloc[:,1:].values
    y_train_8 = df_train_8.iloc[:,0].values
    X_train_cv_8 = X_train_8[train_index]
    y_train_cv_8 = y_train_8[train_index]
    X_valid_cv_8 = X_train_8[valid_index]
    y_valid_cv_8 = y_train_8[valid_index]
    rm8.fit(X_train_cv_8, y_train_cv_8)
    y_pred_8 = rm8.predict_proba(X_valid_cv_8)

    # get true predictions
    predictions = np.stack((y_pred_0, y_pred_1))
    predictions = np.concatenate((predictions, np.array([y_pred_2])), axis=0)
    predictions = np.concatenate((predictions, np.array([y_pred_3])), axis=0)
    predictions = np.concatenate((predictions, np.array([y_pred_4])), axis=0)
    predictions = np.concatenate((predictions, np.array([y_pred_5])), axis=0)
    predictions = np.concatenate((predictions, np.array([y_pred_6])), axis=0)
    predictions = np.concatenate((predictions, np.array([y_pred_7])), axis=0)
    predictions = np.concatenate((predictions, np.array([y_pred_8])), axis=0)
    y_pred = np.zeros(len(y_pred_0))

    for i in range(len(y_pred)):
        arr = predictions[:,i]
        y_pred[i] = np.where(arr == np.amax(arr))[0][0]

    y_valid_cv = y_train[valid_index]
    results['accuracy'].append(accuracy_score(y_valid_cv, y_pred))

    n_classes = len(list(set(y_pred)))
    if n_classes < 3 and not multiclass:
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

# print cv results
print(results['accuracy'])
print_cv_results(results)

#!--- fitting on whole set ---!
# model 0
X_train_0 = df_train_0.iloc[:,1:].values
y_train_0 = df_train_0.iloc[:,0].values
X_test_0 = df_test_0.iloc[:,1:].values
y_test_0 = df_test_0.iloc[:,0].values
rm0.fit(X_train_0, y_train_0)
y_pred_0 = rm0.predict_proba(X_test_0)

# model 1
X_train_1 = df_train_1.iloc[:,1:].values
y_train_1 = df_train_1.iloc[:,0].values
X_test_1 = df_test_1.iloc[:,1:].values
y_test_1 = df_test_1.iloc[:,0].values
rm1.fit(X_train_1, y_train_1)
y_pred_1 = rm1.predict_proba(X_test_1)

# model 2
X_train_2 = df_train_2.iloc[:,1:].values
y_train_2 = df_train_2.iloc[:,0].values
X_test_2 = df_test_2.iloc[:,1:].values
y_test_2 = df_test_2.iloc[:,0].values
rm2.fit(X_train_2, y_train_2)
y_pred_2 = rm2.predict_proba(X_test_2)

# model 3
X_train_3 = df_train_3.iloc[:,1:].values
y_train_3 = df_train_3.iloc[:,0].values
X_test_3 = df_test_3.iloc[:,1:].values
y_test_3 = df_test_3.iloc[:,0].values
rm3.fit(X_train_3, y_train_3)
y_pred_3 = rm3.predict_proba(X_test_3)

# model 4
X_train_4 = df_train_4.iloc[:,1:].values
y_train_4 = df_train_4.iloc[:,0].values
X_test_4 = df_test_4.iloc[:,1:].values
y_test_4 = df_test_4.iloc[:,0].values
rm4.fit(X_train_4, y_train_4)
y_pred_4 = rm4.predict_proba(X_test_4)

# model 5
X_train_5 = df_train_5.iloc[:,1:].values
y_train_5 = df_train_5.iloc[:,0].values
X_test_5 = df_test_5.iloc[:,1:].values
y_test_5 = df_test_5.iloc[:,0].values
rm5.fit(X_train_5, y_train_5)
y_pred_5 = rm5.predict_proba(X_test_5)

# model 6
X_train_6 = df_train_6.iloc[:,1:].values
y_train_6 = df_train_6.iloc[:,0].values
X_test_6 = df_test_6.iloc[:,1:].values
y_test_6 = df_test_6.iloc[:,0].values
rm6.fit(X_train_6, y_train_6)
y_pred_6 = rm6.predict_proba(X_test_6)

# model 7
X_train_7 = df_train_7.iloc[:,1:].values
y_train_7 = df_train_7.iloc[:,0].values
X_test_7 = df_test_7.iloc[:,1:].values
y_test_7 = df_test_7.iloc[:,0].values
rm7.fit(X_train_7, y_train_7)
y_pred_7 = rm7.predict_proba(X_test_7)

# model 8
X_train_8 = df_train_8.iloc[:,1:].values
y_train_8 = df_train_8.iloc[:,0].values
X_test_8 = df_test_8.iloc[:,1:].values
y_test_8 = df_test_8.iloc[:,0].values
rm8.fit(X_train_8, y_train_8)
y_pred_8 = rm8.predict_proba(X_test_8)

# get true predictions
predictions = np.stack((y_pred_0, y_pred_1))
predictions = np.concatenate((predictions, np.array([y_pred_2])), axis=0)
predictions = np.concatenate((predictions, np.array([y_pred_3])), axis=0)
predictions = np.concatenate((predictions, np.array([y_pred_4])), axis=0)
predictions = np.concatenate((predictions, np.array([y_pred_5])), axis=0)
predictions = np.concatenate((predictions, np.array([y_pred_6])), axis=0)
predictions = np.concatenate((predictions, np.array([y_pred_7])), axis=0)
predictions = np.concatenate((predictions, np.array([y_pred_8])), axis=0)
y_pred = np.zeros(len(y_pred_0))

for i in range(len(y_pred)):
    arr = predictions[:,i]
    y_pred[i] = np.where(arr == np.amax(arr))[0][0]

print(np.around(predictions,2))
print(y_pred)
print(y_test)

# printing metrics
print('Testing results:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy = %.3f" % accuracy_score(y_test, y_pred))
