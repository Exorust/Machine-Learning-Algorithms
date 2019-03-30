# Machine-Learning-Algorithms
A scratch implementation of the Perceptron and the Fisher Linear Discriminant Algorithm

# The Perceptron Algorithm

### Description
This project implements the perceptron algorithm on three different datasets using Python3.

### Prerequisites
Python3 along with basic tools such as pandas, numpy and matplolib are required to run this program.

### Structure of project

#### scatterplotting.py : 
##### Parameters-count
1) Creates a scatter plot to visualize the dataset as a set of points color coded to represent their particular class.
2) Saves the scatterplot to a png file each time this function is called

#### epoch.py :
##### Parameters-count, learning_parameter
1) Calculates the cumulative sum of the total error for points that have been misclassified
2) Calculates the new weights by applying a gradient descient over the set of incorrectly classified points

#### estimate_learning,py :
1) Estimates the learning parameter by creating a visual plot of the Error vs the count for different learning parameters

### Built With

The project uses

- Python3
- Numpy
- ArgParse
- Pandas
- Matplotlib

### Authors

Chandrahas Aroori [https://github.com/Exorust] </br>
Prateek Dasgupta [https://github.com/patrik171298] </br>
Soumya
Vaisnavi

### Acknowledgments

We'd like to thank our Machine Learning instructor to give us this opportunity to make such a project.
