#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("dataset_1.csv", header=None)#Reading the dataset
dataset.insert(1,'bias',1)#addind the bias component to the dataframe
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, -1].values
t=np.where(y==0, -1, y)#Replaces class 0 to class -1

plt.figure(1)   
"""
wTx=(w_Transpose)*x
E=Error
"""
E=0 #setting inital error to 0
w=np.random.rand(3)#Forming the initial value of w which is a set of 3 points between 0 and 1
w_T=w.T#Finding the transpose of w
learning_parameter=0.01

for i in range(X.shape[0]):#iterating over all the points
    wTx=np.dot(w_T,X[i])#Calculating W_transpose*x
    if wTx*t[i]<0:#Finding those points incorrectly misclassified
        E+=-1*wTx*t[i]#Calculating total error
        del_E=X[i]*t[i]#Calculating delE
        w=w-learning_parameter*del_E # Finding the new w from del+E
    if t[i]==1:
        plt.plot(X[i][1],X[i][2],'bo')# Plotting points correspoinding to class 1
    elif t[i]==-1:
        plt.plot(X[i][1],X[i][2],'ro')# Plotting points correspoinding to class 0
        w_T=w.T

#plt.subplots(212)
"""
Final line is of the form w0+w1x1+w2x2=0
The graph is plotted between x1 and x2
"""
x= np.linspace(-2,2,100)        
plt.plot(x,-(w_T[0]/w_T[2])-(w_T[1]/w_T[2])*x, 'k')#plotting the line that linearly separates class 1 and 0
plt.figure(1)
    
