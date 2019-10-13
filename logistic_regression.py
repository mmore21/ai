"""
File: logistic_regression.py
Author: Mason Moreland (mmore21)
Description: Logistic regression classifier implemented in Python 3. Uses the
scipy.optimize.minimize Nelder-Mead algorithm to optimize the values of theta.
Example data set consists of test scores and classifier predicts admission rate.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt

def compute_cost(X, y, theta):
    """ Compute logistic regression cost function. """
    m = X.shape[0]
    
    # logistic regression hypothesis
    h = sigmoid(X @ theta)

    errors = (-y * np.log(h) - (1 - y) * np.log(1 - h))

    J_theta = np.sum(errors) / m

    # gradient of cost function with respect to theta
    grad = (X.T @ (h - y)) / m

    return [J_theta, grad]

def sigmoid(z):
    """ Compute sigmoid function of z. """
    return 1 / (1 + np.exp(-z))

def plot_data(X, y):
    """ Plot data on scatterplot. """
    # Get positive and negative indices
    pos_indices = y.nonzero()[0]
    neg_indices = (y == 0).nonzero()[0]

    # Plot positive (1) and negative (0) examples
    plt.plot(X[pos_indices, 0], X[pos_indices, 1], 'k+')
    plt.plot(X[neg_indices, 0], X[neg_indices, 1], 'ko', markerfacecolor='y')

    # Label plot axes
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.show()

def predict(theta, X):
    """ Make binary predictions with trained theta values. """
    # Predict 1 if value is >= 0.5, otherwise return 0
    return sigmoid(X @ theta) >= 0.5

if __name__ == '__main__':
    # Read example data
    data = pd.read_csv("ex2data1.txt", header=None)

    # Convert and split data into numpy arrays
    X = np.asarray(data.iloc[:,0:2])
    y = np.asarray(data.iloc[:,2])

    # Get dimensions of training set
    m, n = X.shape

    # Scatter plot of original dataset
    plot_data(X, y)

    # Initialize default values
    theta = np.zeros(n + 1)
    iterations = 1500
    alpha = 0.01

    # Add a column of ones to X to account for theta 0
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Compute cost with initial parameters
    cost = compute_cost(X, y, theta)

    # Minimize cost function with respect to theta using Nelder-Mead
    wrapped = lambda theta_min: compute_cost(X, y, theta_min)[0]
    result = opt.minimize(wrapped, theta, method='Nelder-Mead', options={"maxiter": 400})
    
    theta = result.x
    cost = result.fun

    # Scatter plot of original dataset with decision boundary
    plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
    plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
    plt.plot(plot_x, plot_y)
    plot_data(X[:, 1:], y)

    # Predict admission probability based on a student's scores
    prob = sigmoid(np.array([1, 45, 85]) @ theta)
    print("For a student with scores 45 and 85, we predict an admission probability of {}".format(prob))

    # Calculate accuracy of training set
    predictions = predict(theta, X)
    print("Train Accuracy: {}".format(np.mean(predictions == y) * 100))
    print("Expected Accuracy (approx): 89.0\n")