"""
File: neural_network.py
Author: Mason Moreland (mmore21)
Description: Neural network implemented in Python 3.
"""

import numpy as np


# --- Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return max(0, z)

def leaky_relu(z):
    return max(0.01 * z, z)


# --- Forward propagation
def predict():
    pass

def compute_loss():
    pass


# --- Backpropagation
def backpropagation():
    pass