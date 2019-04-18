import numpy as np
import pandas as pd
import matplotlib as plt
import argparse
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Logistic_Regression:
	"""
	This class implements a Logistic Regression using the gradient descent algorithm.

	"""
	pass

data=pd.read_csv('data_logistic.txt', header=None).values
data=np.random.shuffle(data)
X=data[:,:4]
y=data[:,4]
#X_train, X_test, y_train, y_test = X[:split,:], X[split: ,:], y[:split,:], y[split: ,:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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

def gradient(X_train, weights):
	#Calculates the gradient based on the newly predicted values
	difference=y_train-prediction(X_train, weights)
	del_E=-np.sum(np.multiply(X_train.T, difference), axis=1)
	return del_E

def update(weights):
	#updates the weights using the gradient descent algorithm
	updated_weights=weights-learning_parameter*gradient(X_train, weights)
	return updated_weights

def determine_class(weights, test):
	np.multiply(X_test.T, weights)

for i in range(10):
	#print(weights)
	weights=update(weights)
	test_prediction=sigmoid(np.sum(np.multiply(X_test, weights), axis=1))
	y_pred=np.heaviside((test_prediction-0.5),0)
	print(accuracy_score(y_test, y_pred))

