"""
File: logistic_regression.py
Author: Mason Moreland (mmore21)
Description: Logistic regression classifier implemented in Python 3. Uses the
scipy.optimize.minimize Nelder-Mead algorithm to optimize the values of theta.
Example data set consists of test scores and classifier predicts admission rate.
"""

import numpy as np
import matplotlib.pyplot as plt

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