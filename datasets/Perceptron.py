#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    """docstring for Perceptron."""

    def __init__(self, X,y,t,learning_parameter=0.01):
        self.X = X
        self.y = y
        self.t = t
        self.E=0 #setting inital error to 0
        self.w=np.random.rand(3)#Forming the initial value of w which is a set of 3 points between 0 and 1
        self.w_T=self.w.T#Finding the transpose of w
        self.learning_parameter = learning_parameter
        self.del_E = 0
        plt.figure(1)

    def plot(self,count):
        plt.figure(1)
        for i in range(self.X.shape[0]):
            if self.t[i]==1:
                plt.plot(self.X[i][1],self.X[i][2],'bo')# Plotting points correspoinding to class 1
            elif t[i]==-1:
                plt.plot(self.X[i][1],self.X[i][2],'ro')# Plotting points correspoinding to class 0
                self.w_T=self.w.T

        x= np.linspace(-2,2,100)
        plt.plot(x,-(self.w_T[0]/self.w_T[2])-(self.w_T[1]/self.w_T[2])*x, 'k')#plotting the line that linearly separates class 1 and 0
        plt.figure(1)
        plt.savefig('foo'+count+'.png')

    def epoch(self):
        """
        wTx=(w_Transpose)*x
        E=Error
        """
        for i in range(self.X.shape[0]):
            wTx=np.dot(self.w_T,self.X[i])#Calculating W_transpose*x
            if wTx*t[i]<0:#Finding those points incorrectly misclassified
                self.E+=-1*wTx*self.t[i]#Calculating total error
                self.del_E=X[i]*self.t[i]#Calculating delE
                self.w=self.w-self.learning_parameter*self.del_E # Finding the new w from del+E


#plt.subplots(212)
"""
Final line is of the form w0+w1x1+w2x2=0
The graph is plotted between x1 and x2
"""


if __name__ == '__main__':
    dataset=pd.read_csv("dataset_1.csv", header=None)#Reading the dataset
    dataset.insert(1,'bias',1)#addind the bias component to the dataframe
    X = dataset.iloc[:, 1:4].values
    y = dataset.iloc[:, -1].values
    t=np.where(y==0, -1, y)#Replaces class 0 to class -1
    percep = Perceptron(X,y,t)
    for count in total_cycles:
        percep.epoch()
        percep.plot(count)
