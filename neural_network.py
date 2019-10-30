"""
File: neural_network.py
Author: Mason Moreland (mmore21)
Description: Two layer neural network classifier implemented in Python 3. Implementation
has yet to be finished.

Remaining Implementation Steps:
- Incorporate relevant dataset
- Finish function implementations
- Write main driver function
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def sigmoid(z):
    """ Compute sigmoid function of z. """
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    """ Compute sigmoid function of z. """
    return sigmoid(z) * (1 - sigmoid(z))

# TODO
def cost_function():
    """ Cost function of a two layer neural network classifier. """
    return 1

# TODO
def rand_initialize_weights():
    """ Randomly initialize weights of a layer. """
    return 1

# TODO
def compute_numerical_gradient(J, theta):
    """
    Computes numerical gradient using finite
    differences to give estimate of gradient.
    """

    """
    numgrad = np.zeros((len(theta)))
    perturb = np.zeros((len(theta)))
    e = np.exp(1) - 4

    for p in range(1:len(theta)):
        # Set pertubation vector
        perturb(p) = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        # Compute numerical gradient
        numgrad(p) = (loss2 - loss1) / (2*e)
        perturb(p) = 0
    """
    return 1

if __name__ == '__main__':
    print("Neural network implementation is not yet completed.")