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

def relu_derivative(z):
    return 1 * (z > 0)

def leaky_relu(z):
    return max(0.01 * z, z)


# --- Forward propagation
def predict(X, W, b):
    z1 = np.dot(W, X) + b[0]
    a1 = relu(z1)

    z2 = np.dot(W, a1) + b[1]
    a2 = sigmoid(z2)

    A = [X, a1, a2]

    return A

def loss(yhat, y):
    return -(y * np.log(yhat) + (1-y) * np.log(1-yhat))

def compute_cost(yhats, ys):
    m = len(ys)
    J = 0
    for i in range(m):
        J += loss(yhats[i], ys[i])
    return J / m
    

# --- Backpropagation
def backpropagation(W, A, ys):
    dz2 = A[2] - ys
    dW2 = np.dot(dz2, A[1].T)
    db2 = dz2
    dz1 = np.dot(W[2].T, dz2) * relu_derivative(Z[1])
    dW1 = np.dot(dz1, A[0].T)
    db1 = dz1