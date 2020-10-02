import logging
import os.path
import sys
import time
import warnings
import numpy as np
import pandas as pd
import prettytable as pt
import csv

from riskmodel import RiskModel

def build_model(rho, data,  show_omitted_variables = False):

    variable_names = data['variable_names']
    rho_values = np.copy(rho)
    rho_names = list(variable_names)

    print("rho values %d" % len(rho_values))

    # removes intercept value or sets it to 0
    if '(Intercept)' in rho_names:
        intercept_ind = variable_names.index('(Intercept)')
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.remove('(Intercept)')
    else:
        intercept_val = 0

    # makes prediction string with intercept value
    if 'outcome_name' in data:
        predict_string = "Pr(Y = +1) = 1.0/(1.0 + exp(-(%d + score))" % intercept_val
    else:
        predict_string = "Pr(%s = +1) = 1.0/(1.0 + exp(-(%d + score))" % (data['outcome_name'].upper(), intercept_val)

    # removes zero values
    if not show_omitted_variables:
        filter_mask = np.array(rho_values)
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]

    # create risk slim model
    rm = RiskModel(rho_values, rho_names, intercept_val, selected_ind, filter_mask != 0)

    return rm

def split_data(csv_path, split_value):
    print(csv_path)
    file = open(csv_path, "rt", encoding="utf8")
    train_file = open("train_file.csv","w+")
    test_file = open("test_file.csv","w+")

    train_file.write(file.readline() + '\n')
    for i,line in enumerate(csv.reader(file, delimiter='\t')):

        if i%split_value == 0:
            test_file.write(line[0] + '\n')
        else:
            train_file.write(line[0] + '\n')

    train_file.close()
    test_file.close()
        
