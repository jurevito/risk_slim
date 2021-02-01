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

output_file = open('result.txt', 'w+')
file = 'groups_imputed.h5'
test_size = 0.2
n_folds = 5
max_runtime = 300.0

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

# remove highly correlated features
df_train = df_train.drop(['X154','X074','X236','X235','X317'], axis=1)
df_test = df_test.drop(['X154','X074','X236','X235','X317'], axis=1)


params = {
    'max_coefficient' : 6,                    # value of largest/smallest coefficient
    'max_L0_value' : 5,                       # maximum model size (set as float(inf))
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

selected_features = stump_selection(0.045, df_train_0, False)
df_train_0 = df_train_0[selected_features]
df_test_0 = df_test_0[selected_features]

df_train_0, df_test_0, X021 = binarize_limits('X021', df_train_0, df_test_0, [-0.5, -0.27, -0.12, 0.13])
df_train_0, df_test_0, X111 = binarize_limits('X111', df_train_0, df_test_0, [-0.18, 0.18, 0.08])
df_train_0, df_test_0, X119 = binarize_limits('X119', df_train_0, df_test_0, [-0.615, -0.27])
df_train_0, df_test_0, X208 = binarize_limits('X208', df_train_0, df_test_0, [0.26, 0.2, 0])
df_train_0, df_test_0, X217 = binarize_limits('X217', df_train_0, df_test_0, [-0.27, -0.02, 0.39])
df_train_0, df_test_0, X220 = binarize_limits('X220', df_train_0, df_test_0, [-0.16, 0.23, 0])
df_train_0, df_test_0, X234 = binarize_limits('X234', df_train_0, df_test_0, [0.25, 0.21])
df_train_0, df_test_0, X245 = binarize_limits('X245', df_train_0, df_test_0, [0.4, 0.29])
df_train_0, df_test_0, X267 = binarize_limits('X267', df_train_0, df_test_0, [0.42, -0.2])
df_train_0, df_test_0, X308 = binarize_limits('X308', df_train_0, df_test_0, [0.3, 0])
df_train_0, df_test_0, X316 = binarize_limits('X316', df_train_0, df_test_0, [-0.58, -0.5, -0.12])


selected_features = stump_selection(2.0, df_train_0, False)
df_train_0 = df_train_0[selected_features]
df_test_0 = df_test_0[selected_features]

X021 = fix_names(X021, selected_features)
X111 = fix_names(X111, selected_features)
X119 = fix_names(X119, selected_features)
X208 = fix_names(X208, selected_features)
X217 = fix_names(X217, selected_features)
X220 = fix_names(X220, selected_features)
X234 = fix_names(X234, selected_features)
X245 = fix_names(X245, selected_features)
X267 = fix_names(X267, selected_features)
X308 = fix_names(X308, selected_features)
X316 = fix_names(X316, selected_features)


op_constraints = {
    'X021': X021,
    'X111': X111,
    'X119': X119,
    'X208': X208,
    'X217': X217,
    'X220': X220,
    'X234': X234,
    'X245': X245,
    'X267': X267,
    'X308': X308,
    'X316': X316,
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

selected_features = stump_selection(0.0025, df_train_1, False)
df_train_1 = df_train_1[selected_features]
df_test_1 = df_test_1[selected_features]

df_train_1, df_test_1, X018 = binarize_limits('X018', df_train_1, df_test_1, [-0.056, 0.2, 0.25, -0.12])
df_train_1, df_test_1, X027 = binarize_limits('X027', df_train_1, df_test_1, [-0.22, 0.23])
df_train_1, df_test_1, X069 = binarize_limits('X069', df_train_1, df_test_1, [0.08, 0.25, 0.35])
df_train_1, df_test_1, X111 = binarize_limits('X111', df_train_1, df_test_1, [0.47, 0.26, -0.18])
df_train_1, df_test_1, X117 = binarize_limits('X117', df_train_1, df_test_1, [0.35, -0.195])
df_train_1, df_test_1, X210 = binarize_limits('X210', df_train_1, df_test_1, [-0.065, -0.055, 0.172, 0.206])
df_train_1, df_test_1, X245 = binarize_limits('X245', df_train_1, df_test_1, [0.5, 0.19, -0.03, -0.125])
df_train_1, df_test_1, X269 = binarize_limits('X269', df_train_1, df_test_1, [0.05, -0.13, -0.18])
df_train_1, df_test_1, X295 = binarize_limits('X295', df_train_1, df_test_1, [0.46])
df_train_1, df_test_1, X308 = binarize_limits('X308', df_train_1, df_test_1, [-0.14, 0.035])

selected_features = stump_selection(0.005, df_train_1, False)
df_train_1 = df_train_1[selected_features]
df_test_1 = df_test_1[selected_features]

X018 = fix_names(X018, selected_features)
X027 = fix_names(X027, selected_features)
X069 = fix_names(X069, selected_features)
X111 = fix_names(X111, selected_features)
X117 = fix_names(X117, selected_features)
X210 = fix_names(X210, selected_features)
X245 = fix_names(X245, selected_features)
X269 = fix_names(X269, selected_features)
X295 = fix_names(X295, selected_features)
X308 = fix_names(X308, selected_features)

op_constraints = {
    'X018': X018,
    'X027': X027,
    'X069': X069,
    'X111': X111,
    'X117': X117,
    'X210': X210,
    'X245': X245,
    'X269': X269,
    'X295': X295,
    'X308': X308,
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

selected_features = stump_selection(0.002, df_train_2, False)
df_train_2 = df_train_2[selected_features]
df_test_2 = df_test_2[selected_features]

df_train_2, df_test_2, X021 = binarize_limits('X021', df_train_2, df_test_2, [-0.45, -0.14, 0.1])
df_train_2, df_test_2, X027 = binarize_limits('X027', df_train_2, df_test_2, [-0.3, 0.2])
df_train_2, df_test_2, X069 = binarize_limits('X069', df_train_2, df_test_2, [-0.16, 0.25, -0.08])
df_train_2, df_test_2, X075 = binarize_limits('X075', df_train_2, df_test_2, [-0.08, 0.1])
df_train_2, df_test_2, X119 = binarize_limits('X119', df_train_2, df_test_2, [-0.615, -0.3])
df_train_2, df_test_2, X183 = binarize_limits('X183', df_train_2, df_test_2, [-0.13, 0.15])
df_train_2, df_test_2, X267 = binarize_limits('X267', df_train_2, df_test_2, [-0.16, 0.43])
df_train_2, df_test_2, X284 = binarize_limits('X284', df_train_2, df_test_2, [0.2, -0.18])
df_train_2, df_test_2, X295 = binarize_limits('X295', df_train_2, df_test_2, [0.45, -0.25])
df_train_2, df_test_2, X316 = binarize_limits('X316', df_train_2, df_test_2, [-0.44, -0.23, 0.2])

selected_features = stump_selection(0.3, df_train_2, False)
df_train_2 = df_train_2[selected_features]
df_test_2 = df_test_2[selected_features]

X021 = fix_names(X021, selected_features)
X027 = fix_names(X027, selected_features)
X069 = fix_names(X069, selected_features)
X075 = fix_names(X075, selected_features)
X119 = fix_names(X119, selected_features)
X183 = fix_names(X183, selected_features)
X267 = fix_names(X267, selected_features)
X284 = fix_names(X284, selected_features)
X295 = fix_names(X295, selected_features)
X316 = fix_names(X316, selected_features)

op_constraints = {
    'X021': X021,
    'X027': X027,
    'X069': X069,
    'X075': X075,
    'X119': X119,
    'X183': X183,
    'X267': X267,
    'X284': X284,
    'X295': X295,
    'X316': X316,
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

selected_features = stump_selection(0.0015, df_train_3, False)
df_train_3 = df_train_3[selected_features]
df_test_3 = df_test_3[selected_features]

df_train_3, df_test_3, X021 = binarize_limits('X021', df_train_3, df_test_3, [-0.2, 0, 0.24])
df_train_3, df_test_3, X027 = binarize_limits('X027', df_train_3, df_test_3, [-0.3, -0.1, 0.23])
df_train_3, df_test_3, X069 = binarize_limits('X069', df_train_3, df_test_3, [-0.2, -0.05, 0.24])
df_train_3, df_test_3, X111 = binarize_limits('X111', df_train_3, df_test_3, [0.44, -0.165, -0.2])
df_train_3, df_test_3, X119 = binarize_limits('X119', df_train_3, df_test_3, [-0.6, -0.34, 0.235])
df_train_3, df_test_3, X195 = binarize_limits('X195', df_train_3, df_test_3, [0.465, -0.056])
df_train_3, df_test_3, X217 = binarize_limits('X217', df_train_3, df_test_3, [-0.12, -0.23, 0.2, 0.3])
df_train_3, df_test_3, X308 = binarize_limits('X308', df_train_3, df_test_3, [-0.23, -0.1, 0.16, 0.4])
df_train_3, df_test_3, X316 = binarize_limits('X316', df_train_3, df_test_3, [-0.5, -0.3, 0.2])

selected_features = stump_selection(0.002, df_train_3, False)
df_train_3 = df_train_3[selected_features]
df_test_3 = df_test_3[selected_features]

X021 = fix_names(X021, selected_features)
X027 = fix_names(X027, selected_features)
X069 = fix_names(X069, selected_features)
X111 = fix_names(X111, selected_features)
X119 = fix_names(X119, selected_features)
X195 = fix_names(X195, selected_features)
X217 = fix_names(X217, selected_features)
X308 = fix_names(X308, selected_features)
X316 = fix_names(X316, selected_features)


op_constraints = {
    'X021': X021,
    'X027': X027,
    'X069': X069,
    'X111': X111,
    'X119': X119,
    'X195': X195,
    'X217': X217,
    'X308': X308,
    'X316': X316,
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

selected_features = stump_selection(0.007, df_train_4, False)
df_train_4 = df_train_4[selected_features]
df_test_4 = df_test_4[selected_features]

df_train_4, df_test_4, X027 = binarize_limits('X027', df_train_4, df_test_4, [-0.23, 0, 0.1, 0.235])
df_train_4, df_test_4, X075 = binarize_limits('X075', df_train_4, df_test_4, [0.11, -0.14])
df_train_4, df_test_4, X101 = binarize_limits('X101', df_train_4, df_test_4, [-0.19, 0.08, 0.15])
df_train_4, df_test_4, X117 = binarize_limits('X117', df_train_4, df_test_4, [0.4, 0.14, 0.05, -0.17])
df_train_4, df_test_4, X119 = binarize_limits('X119', df_train_4, df_test_4, [-0.6, -0.27, 0.19])
df_train_4, df_test_4, X179 = binarize_limits('X179', df_train_4, df_test_4, [-0.3, -0.12, 0.15])
df_train_4, df_test_4, X180 = binarize_limits('X180', df_train_4, df_test_4, [0.22, 0.1, -0.08])
df_train_4, df_test_4, X267 = binarize_limits('X267', df_train_4, df_test_4, [0.3, 0.442, -0.09])
df_train_4, df_test_4, X280 = binarize_limits('X280', df_train_4, df_test_4, [0.38, 0.15, -0.06])

selected_features = stump_selection(0.005, df_train_4, False)
df_train_4 = df_train_4[selected_features]
df_test_4 = df_test_4[selected_features]

X027 = fix_names(X027, selected_features)
X075 = fix_names(X075, selected_features)
X101 = fix_names(X101, selected_features)
X117 = fix_names(X117, selected_features)
X119 = fix_names(X119, selected_features)
X179 = fix_names(X179, selected_features)
X180 = fix_names(X180, selected_features)
X267 = fix_names(X267, selected_features)
X280 = fix_names(X280, selected_features)

op_constraints = {
    'X027': X027,
    'X075': X075,
    'X101': X101,
    'X117': X117,
    'X119': X119,
    'X179': X179,
    'X180': X180,
    'X267': X267,
    'X280': X280,
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

selected_features = stump_selection(0.004, df_train_5, False)
df_train_5 = df_train_5[selected_features]
df_test_5 = df_test_5[selected_features]

df_train_5, df_test_5, X021 = binarize_limits('X021', df_train_5, df_test_5, [0, -0.36, -0.5, 0.32])
df_train_5, df_test_5, X111 = binarize_limits('X111', df_train_5, df_test_5, [0.4, 0.06])
df_train_5, df_test_5, X119 = binarize_limits('X119', df_train_5, df_test_5, [-0.615, -0.28, 0.23, 0.3])
df_train_5, df_test_5, X150 = binarize_limits('X150', df_train_5, df_test_5, [0.3, 0])
df_train_5, df_test_5, X151 = binarize_limits('X151', df_train_5, df_test_5, [0.045, -0.02, -0.01, -0.03])
df_train_5, df_test_5, X195 = binarize_limits('X195', df_train_5, df_test_5, [0.11, -0.056])
df_train_5, df_test_5, X217 = binarize_limits('X217', df_train_5, df_test_5, [0.07, -0.17, 0])
df_train_5, df_test_5, X280 = binarize_limits('X280', df_train_5, df_test_5, [0.17, -0.063])
df_train_5, df_test_5, X285 = binarize_limits('X285', df_train_5, df_test_5, [-0.2, 0.18, 0])
df_train_5, df_test_5, X308 = binarize_limits('X308', df_train_5, df_test_5, [0.4, -0.23, 0.1])

selected_features = stump_selection(0.005, df_train_5, False)
df_train_5 = df_train_5[selected_features]
df_test_5 = df_test_5[selected_features]

X021 = fix_names(X021, selected_features)
X111 = fix_names(X111, selected_features)
X119 = fix_names(X119, selected_features)
X150 = fix_names(X150, selected_features)
X151 = fix_names(X151, selected_features)
X195 = fix_names(X195, selected_features)
X217 = fix_names(X217, selected_features)
X280 = fix_names(X280, selected_features)
X285 = fix_names(X285, selected_features)
X308 = fix_names(X308, selected_features)

op_constraints = {
    'X021': X021,
    'X111': X111,
    'X119': X119,
    'X150': X150,
    'X151': X151,
    'X195': X195,
    'X217': X217,
    'X280': X280,
    'X285': X285,
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

selected_features = stump_selection(0.0038, df_train_6, False)
df_train_6 = df_train_6[selected_features]
df_test_6 = df_test_6[selected_features]

df_train_6, df_test_6, X008 = binarize_limits('X008', df_train_6, df_test_6, [-0.05, 0.17])
df_train_6, df_test_6, X111 = binarize_limits('X111', df_train_6, df_test_6, [-0.13, 0.05, 0.18])
df_train_6, df_test_6, X119 = binarize_limits('X119', df_train_6, df_test_6, [-0.618, 0.22])
df_train_6, df_test_6, X195 = binarize_limits('X195', df_train_6, df_test_6, [0.02, -0.055])
df_train_6, df_test_6, X203 = binarize_limits('X203', df_train_6, df_test_6, [-0.18, 0.12])
df_train_6, df_test_6, X245 = binarize_limits('X245', df_train_6, df_test_6, [-0.13, -0.03, 0.12])
df_train_6, df_test_6, X267 = binarize_limits('X267', df_train_6, df_test_6, [-0.16, 0.37])
df_train_6, df_test_6, X295 = binarize_limits('X295', df_train_6, df_test_6, [0.3, 0.43])
df_train_6, df_test_6, X302 = binarize_limits('X302', df_train_6, df_test_6, [-0.225, -0.15, 0.02])
df_train_6, df_test_6, X316 = binarize_limits('X316', df_train_6, df_test_6, [-0.46, 0.3, -0.56])

selected_features = stump_selection(0.002, df_train_6, False)
df_train_6 = df_train_6[selected_features]
df_test_6 = df_test_6[selected_features]

X008 = fix_names(X008, selected_features)
X111 = fix_names(X111, selected_features)
X119 = fix_names(X119, selected_features)
X195 = fix_names(X195, selected_features)
X203 = fix_names(X203, selected_features)
X245 = fix_names(X245, selected_features)
X267 = fix_names(X267, selected_features)
X295 = fix_names(X295, selected_features)
X302 = fix_names(X302, selected_features)
X316 = fix_names(X316, selected_features)

op_constraints = {
    'X008': X008,
    'X111': X111,
    'X119': X119,
    'X195': X195,
    'X203': X203,
    'X245': X245,
    'X267': X267,
    'X295': X295,
    'X302': X302,
    'X316': X316,
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

selected_features = stump_selection(0.01, df_train_7, False)
df_train_7 = df_train_7[selected_features]
df_test_7 = df_test_7[selected_features]

df_train_7, df_test_7, X018 = binarize_limits('X018', df_train_7, df_test_7, [0, -0.06])
df_train_7, df_test_7, X021 = binarize_limits('X021', df_train_7, df_test_7, [0.2, -0.04, -0.3, -0.13])
df_train_7, df_test_7, X111 = binarize_limits('X111', df_train_7, df_test_7, [-0.17, 0])
df_train_7, df_test_7, X119 = binarize_limits('X119', df_train_7, df_test_7, [-0.59, -0.43, -0.14, 0.07, 0.2])
df_train_7, df_test_7, X172 = binarize_limits('X172', df_train_7, df_test_7, [0.1])
df_train_7, df_test_7, X189 = binarize_limits('X189', df_train_7, df_test_7, [0.14, 0.04, -0.14])
df_train_7, df_test_7, X217 = binarize_limits('X217', df_train_7, df_test_7, [0.08, 0.13, 0.25])
df_train_7, df_test_7, X269 = binarize_limits('X269', df_train_7, df_test_7, [-0.025, -0.12, 0.17])
df_train_7, df_test_7, X295 = binarize_limits('X295', df_train_7, df_test_7, [-0.3, 0.1])

selected_features = stump_selection(0.02, df_train_7, False)
df_train_7 = df_train_7[selected_features]
df_test_7 = df_test_7[selected_features]

X018 = fix_names(X018, selected_features)
X021 = fix_names(X021, selected_features)
X111 = fix_names(X111, selected_features)
X119 = fix_names(X119, selected_features)
X172 = fix_names(X172, selected_features)
X189 = fix_names(X189, selected_features)
X217 = fix_names(X217, selected_features)
X269 = fix_names(X269, selected_features)
X295 = fix_names(X295, selected_features)

op_constraints = {
    'X018': X018,
    'X021': X021,
    'X111': X111,
    'X119': X119,
    'X172': X172,
    'X189': X189,
    'X217': X217,
    'X269': X269,
    'X295': X295,
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

selected_features = stump_selection(0.006, df_train_8, False)
df_train_8 = df_train_8[selected_features]
df_test_8 = df_test_8[selected_features]

df_train_8, df_test_8, X021 = binarize_limits('X021', df_train_8, df_test_8, [0, 0.25, -0.3])
df_train_8, df_test_8, X069 = binarize_limits('X069', df_train_8, df_test_8, [-0.2, -0.07, 0.11, 0.37])
df_train_8, df_test_8, X111 = binarize_limits('X111', df_train_8, df_test_8, [0.35, 0.04, -0.22])
df_train_8, df_test_8, X117 = binarize_limits('X117', df_train_8, df_test_8, [0.4, -0.18, -0.195])
df_train_8, df_test_8, X119 = binarize_limits('X119', df_train_8, df_test_8, [-0.1, -0.618, 0.24])
df_train_8, df_test_8, X217 = binarize_limits('X217', df_train_8, df_test_8, [-0.22, 0.11, -0.14, 0.28])
df_train_8, df_test_8, X245 = binarize_limits('X245', df_train_8, df_test_8, [-0.04, 0.37])
df_train_8, df_test_8, X280 = binarize_limits('X280', df_train_8, df_test_8, [0.23, -0.06])
df_train_8, df_test_8, X302 = binarize_limits('X302', df_train_8, df_test_8, [-0.23, -0.15, 0.12])
df_train_8, df_test_8, X316 = binarize_limits('X316', df_train_8, df_test_8, [-0.5, 0.25, 0.12])

selected_features = stump_selection(0.002, df_train_8, False)
df_train_8 = df_train_8[selected_features]
df_test_8 = df_test_8[selected_features]

X021 = fix_names(X021, selected_features)
X069 = fix_names(X069, selected_features)
X111 = fix_names(X111, selected_features)
X117 = fix_names(X117, selected_features)
X119 = fix_names(X119, selected_features)
X217 = fix_names(X217, selected_features)
X245 = fix_names(X245, selected_features)
X280 = fix_names(X280, selected_features)
X302 = fix_names(X302, selected_features)
X316 = fix_names(X316, selected_features)

op_constraints = {
    'X021': X021,
    'X069': X069,
    'X111': X111,
    'X117': X117,
    'X119': X119,
    'X217': X217,
    'X245': X245,
    'X280': X280,
    'X302': X302,
    'X316': X316,
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
print('Accuracy 0 = %.2f' % (accuracy_score(y_test_0, np.around(np.array(y_pred_0)))))

# model 1
X_train_1 = df_train_1.iloc[:,1:].values
y_train_1 = df_train_1.iloc[:,0].values
X_test_1 = df_test_1.iloc[:,1:].values
y_test_1 = df_test_1.iloc[:,0].values
rm1.fit(X_train_1, y_train_1)
y_pred_1 = rm1.predict_proba(X_test_1)
print('Accuracy 1 = %.2f' % (accuracy_score(y_test_1, np.around(np.array(y_pred_1)))))

# model 2
X_train_2 = df_train_2.iloc[:,1:].values
y_train_2 = df_train_2.iloc[:,0].values
X_test_2 = df_test_2.iloc[:,1:].values
y_test_2 = df_test_2.iloc[:,0].values
rm2.fit(X_train_2, y_train_2)
y_pred_2 = rm2.predict_proba(X_test_2)
print('Accuracy 2 = %.2f' % (accuracy_score(y_test_2, np.around(np.array(y_pred_2)))))

# model 3
X_train_3 = df_train_3.iloc[:,1:].values
y_train_3 = df_train_3.iloc[:,0].values
X_test_3 = df_test_3.iloc[:,1:].values
y_test_3 = df_test_3.iloc[:,0].values
rm3.fit(X_train_3, y_train_3)
y_pred_3 = rm3.predict_proba(X_test_3)
print('Accuracy 3 = %.2f' % (accuracy_score(y_test_3, np.around(np.array(y_pred_3)))))

# model 4
X_train_4 = df_train_4.iloc[:,1:].values
y_train_4 = df_train_4.iloc[:,0].values
X_test_4 = df_test_4.iloc[:,1:].values
y_test_4 = df_test_4.iloc[:,0].values
rm4.fit(X_train_4, y_train_4)
y_pred_4 = rm4.predict_proba(X_test_4)
print('Accuracy 4 = %.2f' % (accuracy_score(y_test_4, np.around(np.array(y_pred_4)))))

# model 5
X_train_5 = df_train_5.iloc[:,1:].values
y_train_5 = df_train_5.iloc[:,0].values
X_test_5 = df_test_5.iloc[:,1:].values
y_test_5 = df_test_5.iloc[:,0].values
rm5.fit(X_train_5, y_train_5)
y_pred_5 = rm5.predict_proba(X_test_5)
print('Accuracy 5 = %.2f' % (accuracy_score(y_test_5, np.around(np.array(y_pred_5)))))

# model 6
X_train_6 = df_train_6.iloc[:,1:].values
y_train_6 = df_train_6.iloc[:,0].values
X_test_6 = df_test_6.iloc[:,1:].values
y_test_6 = df_test_6.iloc[:,0].values
rm6.fit(X_train_6, y_train_6)
y_pred_6 = rm6.predict_proba(X_test_6)
print('Accuracy 6 = %.2f' % (accuracy_score(y_test_6, np.around(np.array(y_pred_6)))))

# model 7
X_train_7 = df_train_7.iloc[:,1:].values
y_train_7 = df_train_7.iloc[:,0].values
X_test_7 = df_test_7.iloc[:,1:].values
y_test_7 = df_test_7.iloc[:,0].values
rm7.fit(X_train_7, y_train_7)
y_pred_7 = rm7.predict_proba(X_test_7)
print('Accuracy 7 = %.2f' % (accuracy_score(y_test_7, np.around(np.array(y_pred_7)))))

# model 8
X_train_8 = df_train_8.iloc[:,1:].values
y_train_8 = df_train_8.iloc[:,0].values
X_test_8 = df_test_8.iloc[:,1:].values
y_test_8 = df_test_8.iloc[:,0].values
rm8.fit(X_train_8, y_train_8)
y_pred_8 = rm8.predict_proba(X_test_8)
print('Accuracy 8 = %.2f' % (accuracy_score(y_test_8, np.around(np.array(y_pred_8)))))

print('Accuracy 0 = %.2f' % (accuracy_score(y_test_0, np.around(np.array(y_pred_0)))))
print('Accuracy 1 = %.2f' % (accuracy_score(y_test_1, np.around(np.array(y_pred_1)))))
print('Accuracy 2 = %.2f' % (accuracy_score(y_test_2, np.around(np.array(y_pred_2)))))
print('Accuracy 3 = %.2f' % (accuracy_score(y_test_3, np.around(np.array(y_pred_3)))))
print('Accuracy 4 = %.2f' % (accuracy_score(y_test_4, np.around(np.array(y_pred_4)))))
print('Accuracy 5 = %.2f' % (accuracy_score(y_test_5, np.around(np.array(y_pred_5)))))
print('Accuracy 6 = %.2f' % (accuracy_score(y_test_6, np.around(np.array(y_pred_6)))))
print('Accuracy 7 = %.2f' % (accuracy_score(y_test_7, np.around(np.array(y_pred_7)))))
print('Accuracy 8 = %.2f' % (accuracy_score(y_test_8, np.around(np.array(y_pred_8)))))

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
