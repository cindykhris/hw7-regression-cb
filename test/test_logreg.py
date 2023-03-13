"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from regression import logreg, utils
import sklearn.preprocessing as StandardScaler

# test prediction function for logistic regression model 
def test_make_prediction():
    # create a logistic regression model
    model = logreg.LogisticRegressor(num_feats=3)
    # create a matrix of feature values
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # call the prediction function
    y_pred = model.make_prediction(X)
    # check the output
    assert y_pred.shape == (3, 1)
    assert y_pred.dtype == np.int64
    
# test loss function for logistic regression model
def test_loss_function():
    # create a logistic regression model
	model = logreg.LogisticRegressor(num_feats=3)
	# create a matrix of feature values
	X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	# create a vector of target values
	y = np.array([0, 1, 1])
	# call the prediction function
	y_pred = model.make_prediction(X)
	# call the loss function
	loss = model.loss_function(y, y_pred)
	# check the output
	assert loss.shape == ()
	assert loss.dtype == np.float64
        
# test gradient function for logistic regression model
def test_calculate_gradient():
	# create a logistic regression model
	model = logreg.LogisticRegressor(num_feats=3)
	# create a matrix of feature values
	X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	# create a vector of target values
	y = np.array([0, 1, 1])
	# call the prediction function
	y_pred = model.make_prediction(X)
	# call the gradient function
	grad = model.calculate_gradient(y, X)
	# check the output
	assert grad.shape == (3, 1)
	assert grad.dtype == np.float64

# test training function for logistic regression model
def test_train():
	# create a logistic regression model
	model = logreg.LogisticRegressor(num_feats=3)
	# create a matrix of feature values
	X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	# create a vector of target values
	y = np.array([0, 1, 1])
	# call the training function
	model.train(X, y)
	# check the output
	assert model.W.shape == (3, 1)
	assert model.W.dtype == np.float64