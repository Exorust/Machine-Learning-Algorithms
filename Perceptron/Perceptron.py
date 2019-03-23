# Perceptron Code
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    """Generates a perceptron

    Generates a perceptron, which can be run multiple times on the training data
    to compute a decision boundary.

    Atttributes:
        X: The original matrix containing input values
        t: The output matrix containing +1 for +ve classification and -1 for -ve classification
        total_cycles: Number of cycles to run the epoch calculation
        E: Error matrix containing all the errors
        del_E: Array containg the respective gradient of the errors
        w: Weight matrix
        learning_parameter: The rate at which gradient descent is computed
        filetype: The location of the file which is being processed
    """

    def __init__(self, X, t, total_cycles,filetype, learning_parameter=0.01):
        """Intitializes the perceptron

        Args:
            X: The original matrix containing input values
            t: The output matrix containing +1 for +ve classification and -1 for -ve classification
            total_cycles: Number of cycles to run the epoch calculation
            filetype: The location of the file which is being processed
            learning_parameter: The rate at which gradient descent is computed
        """
        self.X = X
        self.t = t
        self.total_cycles = total_cycles
        # setting inital error to 0
        self.E = np.zeros(self.total_cycles)
        self.del_E = np.zeros((self.total_cycles, 3))
        # Setting the seed value and forming the initial value of w, which is a set of 3 points between 0 and 1
        random.seed(9001)
        self.w = np.random.rand(3)
        self.w_T = self.w.T  # Finding the transpose of w
        self.learning_parameter = learning_parameter
        self.filetype = filetype

    def epoch(self, count):
        """Runs the perceptron once

        Runs the perceptron once and then recalulates the weight, via gradient
        descent, ie it will keep track of Error and Gradient of the error.

        Args:
            count: Current count of the the epoch running
        """
        for i in range(self.X.shape[0]):
            # Calculating W_transpose*x
            wTx = np.dot(self.w.T, self.X[i])
            # To be executed on those points incorrectly misclassified
            if wTx * t[i] < 0:
                # Calculating total error
                self.E[count] += -1 * wTx * self.t[i]
                self.del_E[count] += X[i] * self.t[i]
        # Finding the new w from del+E
        self.w = self.w + (self.learning_parameter * self.del_E[count])
        self.w_T = self.w.T
        print("Finished epoch:" + str(count))

    def scatterplotting(self, count):
        """Plots the current status of the perceptron

        Plots the descision boundary of the perceptron on a 2D plot of x1 and x2
        axes.

        Args:
            count: Current count of the the epoch running
        """
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
        """Runs the perceptron multiple times

        Runs the perceptron  and then plots it.
        """
        for count in range(0, self.total_cycles):
            self.epoch(count)
            self.scatterplotting(count)

    def compute_noplot(self):
        """Runs the perceptron multiple times

        Runs the perceptron but does not plot it.
        """
        for count in range(0, self.total_cycles):
            self.epoch(count)


class Testing:
    """Tests values of alpha for a perceptron

    Runs a multiple perceptrons with different values of learning parameter and
    plots these so as to understand which value of alpha is the best
    """
    def estimate_learning(self, total_cycles,filetype):
        """Estimates the value of alpha via plotting

        Runs a multiple perceptrons with different values of learning parameter and
        plots these so as to understand which value of alpha is the best

        Args:
            total_cycles: Number of cycles to run the perceptron.
            filetype: Location to save the plots.
        """
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
    dataset = pd.read_csv(filestring,header=None)
    # Adding the bias component to the dataframe
    dataset.insert(1, 'bias', 1)
    X = dataset.iloc[:, 1:4].values
    y = dataset.iloc[:, -1].values
    # Replaces class 0 to class -1
    t = np.where(y == 0, -1, y)

    # Run the perceptron
    learning_parameter = 0.01
    percep = Perceptron(X, t, args.c,filetype, learning_parameter)
    percep.compute()

    # # Test the values of alpha
    # test = Testing()
    # test.estimate_learning(100,filetype)
