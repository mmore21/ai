"""
File: linear_regression.py
Author: Mason Moreland (mmore21)
Description: Linear regression implemented in Python 3. Uses gradient descent
algorithm to minimize theta values. Example dataset consists of population and
city profit and the univariate regressor predicts respective profit per population.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


if __name__ == '__main__':
    # Read example data
    data = pd.read_csv("ex1data1.txt", header=None)

    # Convert and split data into numpy arrays
    X = np.asarray(data.iloc[:,0:1])
    y = np.asarray(data.iloc[:,1])

    # Get column dimension of training set
    m, n = X.shape
    X = X.reshape((m, 1))
    y = y.reshape((m, 1))

    # Scatter plot of original dataset
    plot_data(X, y)

    # Initialize default values
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    
    # Add a column of ones to X to account for theta 0
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Compute cost with initial parameters
    cost = compute_cost(X, y, theta)

    # Minimize cost function with respect to theta using gradient descent
    theta = gradient_descent(X, y, theta, alpha, iterations)[0]

    # Make predictions based on arbitrary city population and size
    predict1 = [1, 3.5] @ theta
    print('For population = 35,000, we predict a profit of ${}\n'.format(int(predict1 * 10000)))
    predict2 = [1, 7] @ theta
    print('For population = 70,000, we predict a profit of ${}\n'.format(int(predict2 * 10000)))

    # Plot linear regression line ontop of dataset
    plt.plot(X[:,1], X @ theta, 'r-')
    plot_data(X[:,1], y)