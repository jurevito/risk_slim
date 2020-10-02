import numpy as np
import math

class RiskModel:
    def __init__(self, rho_values, rho_names, intercept_val, selected_ind, filter_mask):

    	self.rho_values = rho_values
    	self.rho_names = rho_names
    	self.intercept_val = intercept_val
    	self.selected_ind = selected_ind
    	self.filter_mask = filter_mask
    	self.threshold = 0.5

    def predict_single(self, input):

    	input = np.array(input)[self.filter_mask]
    	score = float(np.dot(input, self.rho_values.T))
    	return 1.0/(1.0 + math.exp(-(self.intercept_val + score)))

    def test_model(self, test_data):

    	test_data['X'] = test_data['X'][:,1:]
    	test_data['X'] = test_data['X'][:,self.filter_mask]

    	scores = (np.squeeze(np.asarray(np.dot(test_data['X'], self.rho_values.T)))).astype('float64')
    	for i,score in enumerate(scores):
    		scores[i] = float(1.0/(1.0 + math.exp(-(self.intercept_val + score))))
    	# print(scores)

    	n = 0
    	N = len(scores)

    	# testing on test data
    	for i,x in enumerate(test_data['Y']):
    		y_value = x[0]

    		if y_value == -1:
    			y_value = 0

    		if y_value == round(scores[i] - self.threshold + 0.5):
    			n += 1

    	print(n)
    	print(N)
    	print(n/N)
    	print(scores[0:20])
    	print("_____________")


    		
    		
