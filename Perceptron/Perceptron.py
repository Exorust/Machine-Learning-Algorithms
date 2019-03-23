# Perceptron Code
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    """Generates a perceptron

    """

    def __init__(self, X, t, total_cycles,filetype, learning_parameter=0.01):
        self.X = X
        self.t = t
        self.total_cycles = total_cycles
        self.E = np.zeros(self.total_cycles)  # setting inital error to 0
        self.del_E = np.zeros((self.total_cycles, 3))
        # Forming the initial value of w which is a set of 3 points between 0 and 1
        random.seed(9001)
        self.w = np.random.rand(3)
        self.w_T = self.w.T  # Finding the transpose of w
        self.learning_parameter = learning_parameter
        self.filetype = filetype

    def epoch(self, count):
        for i in range(self.X.shape[0]):
            wTx = np.dot(self.w.T, self.X[i])  # Calculating W_transpose*x
            if wTx * t[i] < 0:  # Finding those points incorrectly misclassified
                # Calculating total error
                self.E[count] += -1 * wTx * self.t[i]
                self.del_E[count] += X[i] * self.t[i]  # Calculating delE
        # Finding the new w from del+E
        self.w = self.w + (self.learning_parameter * self.del_E[count])
        self.w_T = self.w.T
        print("Finished epoch:" + str(count))

    def scatterplotting(self, count):
        figure, axes = plt.subplots()
        # axes.plot(self.X, self.y)
        axes.set_title("Scatterplot  " + str(count))
        axes.set_xlabel("x1")
        axes.set_ylabel("x2")

        for i in range(self.X.shape[0]):
            if self.t[i] == 1:
                # Plotting points correspoinding to class 1
                axes.plot(self.X[i][1], self.X[i][2], 'bo')
            elif t[i] == -1:
                # Plotting points correspoinding to class 0
                axes.plot(self.X[i][1], self.X[i][2], 'ro')
                self.w_T = self.w.T

        x = np.linspace(-3, 3, 100)
        self.w_T = self.w.T
        # plotting the line that linearly separates class 1 and 0
        axes.plot(x, -(self.w_T[0] / self.w_T[2]) -
                  (self.w_T[1] / self.w_T[2]) * x, 'k')
        plt.savefig('Scatterplots/'+filetype+'/Sc' + str(count) + '.png')
        plt.close()
        print("Finished plotting:" + str(count))

    def compute(self):
        for count in range(0, self.total_cycles):
            self.epoch(count)
            self.scatterplotting(count)

    def compute_noplot(self):
        for count in range(0, self.total_cycles):
            self.epoch(count)
            # self.scatterplotting(count)


class Testing:
    def estimate_learning(self, total_cycles,filetype):
        lp_possibilities = np.geomspace(0.001, 10, 30)
        numbering = 1

        for learning_parameter in lp_possibilities:
            percep = Perceptron(X, t, total_cycles, learning_parameter)
            percep.compute_noplot()
            figure, axes = plt.subplots()

            x = np.linspace(0, 100, 100)
            axes.plot(x, percep.E, 'k')

            axes.set_title("Learning Rate" + str(learning_parameter))
            axes.set_xlabel("Number of Iterations")
            axes.set_ylabel("Error")

            plt.savefig('Learning/'+filetype+'/Learning' + str(numbering) + " " + str(learning_parameter) + '.png')
            numbering += 1
            print("Finished estimating:" + str(learning_parameter) + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a perceptron')
    parser.add_argument('-c', metavar='total_cycles',
                        type=int, default=20, help='Number of cycles')
    args = parser.parse_args()

    filetype = "dataset_2"
    filestring = "../datasets/"+filetype+".csv"
    dataset = pd.read_csv(filestring,
                          header=None)
    dataset.insert(1, 'bias', 1)  # addind the bias component to the dataframe
    X = dataset.iloc[:, 1:4].values
    y = dataset.iloc[:, -1].values
    t = np.where(y == 0, -1, y)  # Replaces class 0 to class -1
    # print(t.shape[0])
    learning_parameter = 0.01
    percep = Perceptron(X, t, args.c,filetype, learning_parameter)
    percep.compute()

    # test = Testing()
    # test.estimate_learning(100,filetype)
