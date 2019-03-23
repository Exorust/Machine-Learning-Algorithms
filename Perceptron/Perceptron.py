#Perceptron Code
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    """docstring for Perceptron."""
    def __init__(self, X,t,total_cycles,learning_parameter=0.01):
        self.X = X
        # self.y = y
        self.t = t
        self.total_cycles = total_cycles
        self.E=np.zeros(self.total_cycles) #setting inital error to 0
        self.del_E = np.zeros((self.total_cycles,3))
        self.w=np.random.rand(3)#Forming the initial value of w which is a set of 3 points between 0 and 1
        self.w_T=self.w.T#Finding the transpose of w
        self.learning_parameter = learning_parameter

        #Needed?
        # plt.figure(1)

    def epoch(self,count):
        for i in range(self.X.shape[0]):
            wTx=np.dot(self.w.T,self.X[i])#Calculating W_transpose*x
            if wTx*t[i]<0:#Finding those points incorrectly misclassified
                self.E[count] += -1*wTx*self.t[i]#Calculating total error
                self.del_E[count] += X[i] * self.t[i]#Calculating delE
        self.w = self.w + (self.learning_parameter * self.del_E[count]) # Finding the new w from del+E
        self.w_T=self.w.T
        print("Finished epoch:"+str(count))

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
        self.w_T=self.w.T
        axes.plot(x,-(self.w_T[0]/self.w_T[2])-(self.w_T[1]/self.w_T[2])*x,'k')#plotting the line that linearly separates class 1 and 0
        plt.savefig('Scatterplots/foo'+str(count)+'.png')
        plt.close()
        print("Finished plotting:"+str(count))

    def compute(self):
        for count in range(0,self.total_cycles):
            self.epoch(count)
            self.scatterplotting(count)

    def compute_noplot(self):
        for count in range(0,self.total_cycles):
            self.epoch(count)
            # self.scatterplotting(count)

class Testing:
    def estimate_learning(self,total_cycles):
        lp_possibilities = np.geomspace(0.001,10,10)

        for learning_parameter in lp_possibilities:
            percep = Perceptron(X,t,total_cycles,learning_parameter)
            percep.compute_noplot()
            figure, axes = plt.subplots()

            x= np.linspace(0,100,100)
            axes.plot(x,percep.E, 'k')


            axes.set_title("Learning Rate"+str(learning_parameter))
            axes.set_xlabel("Number of Iterations")
            axes.set_ylabel("Error")

            plt.savefig('Learning'+str(learning_parameter)+'.png')
            print("Finished estimating:"+str(learning_parameter)+"\n\n")

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
    # print(t.shape[0])
    learning_parameter = 0.01
    percep = Perceptron(X,t,args.c,learning_parameter)
    percep.compute()

    # test = Testing()
    # test.estimate_learning(100)
