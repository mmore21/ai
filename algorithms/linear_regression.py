"""
File: linear_regression.py
Author: Mason Moreland (mmore21)
Description: Linear regression implemented in Python 3. Uses gradient descent
algorithm to minimize theta values. Example dataset consists of population and
city profit and the univariate regressor predicts respective profit per population.
"""

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, theta, alpha, iterations):
    """ Performs gradient descent to optimize theta. """
    m = X.shape[0]

    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        h = X @ theta

        errors = h - y

        delta = X.T @ errors

        # Updates parameters each step with learning rate alpha
        theta = theta - (alpha / m) * delta

        J_history[i] = compute_cost(X, y, theta)
    
    return [theta, J_history]

def compute_cost(X, y, theta):
    """ Compute linear regression cost function. """
    m = X.shape[0]
    
    # Linear regression hypothesis
    h = X @ theta

    squared_errors = (h-y) ** 2

    J_theta = (1 / (2 * m)) * np.sum(squared_errors)

    return J_theta

def plot_data(X, y):
    """ Plot data on scatterplot. """
    plt.scatter(X, y)
    
    # Label plot title and axes
    plt.title("City Profit based on Population")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")

    plt.show()