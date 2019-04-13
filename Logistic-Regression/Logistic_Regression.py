import numpy as np
import pandas as pd
import matplotlib as plt
import argparse
import os
import sys

class Logistic_Regression:
	"""
	This class implements a Logistic Regression using the gradient descent algorithm.

	"""
	pass

data=pd.read_csv('data_logistic.txt', header=None).values
X=data[:,:4]
t=data[:,4]
learning_parameter=0.01

np.random.seed(0)
weights=np.random.rand(4)
#print(weights)

def sigmoid(x):
	# Returns the sigmoid value transformation for any input X
	return 1/(1+np.exp(-x))

def prediction(attributes, weights):
	predicted=np.sum(weights.T*attributes, axis=1)
	return sigmoid(predicted)

def gradient(X, weights):
	difference=t-prediction(X, weights)
	a=np.multiply(X.T, difference)
	del_E=-np.sum(np.multiply(X.T, difference), axis=1)
	return del_E

def update(weights):
	updated_weights=weights-learning_parameter*gradient(X, weights)
	return updated_weights

def visualise_data(weights):
	figure, ax = plt.subplots()

print(update(weights))
for i in range(6000):
	print(update(weights))
	weights=update(weights)