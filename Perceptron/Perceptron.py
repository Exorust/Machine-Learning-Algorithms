#Perceptron Code
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    """docstring for Perceptron."""

    def __init__(self, X,y,t,learning_parameter=0.01):
        self.X = X
        self.y = y
        self.t = t
        # TODO: Make E and del_E arrays
        self.E=0 #setting inital error to 0
        self.w=np.random.rand(3)#Forming the initial value of w which is a set of 3 points between 0 and 1
        self.w_T=self.w.T#Finding the transpose of w
        self.learning_parameter = learning_parameter
        self.del_E = 0
        #Needed?
        plt.figure(1)

    def scatterplotting(self,count):
        figure, axes = plt.subplots()
        # axes.plot(self.X, self.y)
        axes.set_title("Scatterplot  "+ str(count))
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

        for i in range(self.X.shape[0]):
            if self.t[i]==1:
                axes.plot(self.X[i][1],self.X[i][2],'bo')# Plotting points correspoinding to class 1
            elif t[i]==-1:
                axes.plot(self.X[i][1],self.X[i][2],'ro')# Plotting points correspoinding to class 0
                self.w_T=self.w.T

        x= np.linspace(-3,3,100)
        axes.plot(x,-(self.w_T[0]/self.w_T[2])-(self.w_T[1]/self.w_T[2])*x, 'k')#plotting the line that linearly separates class 1 and 0
        plt.savefig('Scatterplots/foo'+str(count)+'.png')
        print("Finished plotting:"+str(count))

    def epoch(self,count,learning_parameter):
        for i in range(self.X.shape[0]):
            wTx=np.dot(self.w_T,self.X[i])#Calculating W_transpose*x
            if wTx*t[i]<0:#Finding those points incorrectly misclassified
                self.E+=-1*wTx*self.t[i]#Calculating total error
                self.del_E=X[i]*self.t[i]#Calculating delE
                self.w=self.w-learning_parameter*self.del_E # Finding the new w from del+E
        print("Finished epoch:"+str(count))

    def estimate_learning(self):
        lp_possibilities = np.geomspace(0.001,10,10)
        total_cycles = 10

        for learning_parameter in lp_possibilities:
            figure, axes = plt.subplots()
            # axes.plot(self.X, self.y)
            axes.set_title("Learning Rate"+str(learning_parameter))
            axes.set_xlabel("Number of Iterations")
            axes.set_ylabel("Error")

            for count in range(1,total_cycles+1):
                percep.epoch(count,learning_parameter)
                axes.plot(count,self.E,'bo')

            plt.savefig('Learning'+str(learning_parameter)+'.png')

        print("Finished plotting:"+str(count))

#plt.subplots(212)
"""
Final line is of the form w0+w1x1+w2x2=0
The graph is plotted between x1 and x2
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a perceptron')
    parser.add_argument('-c', metavar='total_cycles', type=int, default=20,help='Number of cycles')
    args = parser.parse_args()

    # TODO: Save in different folders
    dataset=pd.read_csv("../datasets/dataset_1.csv", header=None)#Reading the dataset
    dataset.insert(1,'bias',1)#addind the bias component to the dataframe
    X = dataset.iloc[:, 1:4].values
    y = dataset.iloc[:, -1].values
    t=np.where(y==0, -1, y)#Replaces class 0 to class -1
    learning_parameter = 0.01
    percep = Perceptron(X,y,t,learning_parameter)
    for count in range(1,args.c+1):
        percep.epoch(count,learning_parameter)
        percep.scatterplotting(count)

    # percep.estimate_learning()
