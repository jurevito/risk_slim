import numpy as np
import math

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from riskslim.helper_functions import load_data_from_csv, check_data, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa


class RiskModel(BaseEstimator):

    def __init__(self, sample_weights_csv_file = None, data_headers = None, fold_csv_file = None, params = None, settings = None, show_omitted_variables = False, threshold = 0.5):

        self.sample_weights_csv_file = sample_weights_csv_file
        self.data_headers = data_headers
        self.fold_csv_file = fold_csv_file
        self.settings = settings
        self.show_omitted_variables = show_omitted_variables
        self.threshold = threshold

        self.max_coefficient = params['max_coefficient']
        self.max_L0_value = params['max_L0_value']
        self.max_offset = params['max_offset']
        self.c0_value = params['c0_value']
        self.w_pos = params['w_pos']

    def fit_transform(self, X, y):

        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        # transforming data
        raw_data = np.insert(X, 0, y, axis=1)
        N = raw_data.shape[0]

        # setup Y vector and Y_name
        Y_col_idx = [0]
        Y = raw_data[:, Y_col_idx]
        Y_name = self.data_headers[Y_col_idx[0]]
        Y[Y == 0] = -1

        # setup X and X_names
        X_col_idx = [j for j in range(raw_data.shape[1]) if j not in Y_col_idx]
        X = raw_data[:, X_col_idx]
        variable_names = [self.data_headers[j] for j in X_col_idx]

        # insert a column of ones to X for the intercept
        X = np.insert(arr=X, obj=0, values=np.ones(N), axis=1)
        variable_names.insert(0, '(Intercept)')

        if self.sample_weights_csv_file is None:
            sample_weights = np.ones(N)
        else:
            if os.path.isfile(self.sample_weights_csv_file):
                sample_weights = pd.read_csv(self.sample_weights_csv_file, sep=',', header=None)
                sample_weights = sample_weights.as_matrix()
            else:
                raise IOError('could not find sample_weights_csv_file: %s' % self.sample_weights_csv_file)

        self.data = {
            'X': X,
            'Y': Y,
            'variable_names': variable_names,
            'outcome_name': Y_name,
            'sample_weights': sample_weights,
        }

        #load folds
        if self.fold_csv_file is not None:
            if not os.path.isfile(self.fold_csv_file):
                raise IOError('could not find fold_csv_file: %s' % self.fold_csv_file)
            else:
                fold_idx = pd.read_csv(self.fold_csv_file, sep=',', header=None)
                fold_idx = fold_idx.values.flatten()
                K = max(fold_idx)
                all_fold_nums = np.sort(np.unique(fold_idx))
                assert len(fold_idx) == N, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), N)
                assert np.all(all_fold_nums == np.arange(1, K+1)), "folds should contain indices between 1 to %r" % K
                assert fold_num in np.arange(0, K+1), "fold_num should either be 0 or an integer between 1 to %r" % K
                if fold_num >= 1:
                    test_idx = fold_num == fold_idx
                    train_idx = fold_num != fold_idx
                    data['X'] = data['X'][train_idx,]
                    data['Y'] = data['Y'][train_idx]
                    data['sample_weights'] = data['sample_weights'][train_idx]

        assert check_data(self.data)
        
        # create coefficient set and set the value of the offset parameter
        coef_set = CoefficientSet(variable_names = self.data['variable_names'], lb = -self.max_coefficient, ub = self.max_coefficient, sign = 0)
        conservative_offset = get_conservative_offset(self.data, coef_set, self.max_L0_value)
        self.max_offset = min(self.max_offset, conservative_offset)
        coef_set['(Intercept)'].ub = self.max_offset
        coef_set['(Intercept)'].lb = -self.max_offset

        # edit contraints here
        constraints = {
            'L0_min': 0,
            'L0_max': self.max_L0_value,
            'coef_set':coef_set,
        }

        # fit using ltca
        model_info, mip_info, lcpa_info = run_lattice_cpa(self.data, constraints, self.settings)
        rho = model_info['solution']
        print_model(model_info['solution'], self.data)

        # fitting model
        variable_names = self.data['variable_names']
        rho_values = np.copy(rho)
        rho_names = list(variable_names)

        # removes intercept value or sets it to 0
        if '(Intercept)' in rho_names:
            intercept_ind = variable_names.index('(Intercept)')
            self.intercept_val = int(rho[intercept_ind])
            rho_values = np.delete(rho_values, intercept_ind)
            rho_names.remove('(Intercept)')
        else:
            self.intercept_val = 0

        self.filter_mask = np.array(rho_values) != 0

        # removes zero values
        if not self.show_omitted_variables:
            selected_ind = np.flatnonzero(rho_values)
            self.rho_values = rho_values[selected_ind]
            self.rho_names = [rho_names[i] for i in selected_ind]

        return self

    def predict(self, X):

        X = check_array(X, accept_sparse=True)
        X = X[:,self.filter_mask]

        scores = (np.squeeze(np.asarray(np.dot(X, self.rho_values.T)))).astype('float64')
        y = np.array(scores)

        for i,score in enumerate(scores):
            y[i] = round(float(1.0/(1.0 + math.exp(-(self.intercept_val + score)))) - self.threshold + 0.5)

        return y