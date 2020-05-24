"""
File: neural_network.py
Author: Mason Moreland (mmore21)
Description: Neural network implemented in Python 3.
"""

import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def relu(self, z):
        return np.maximum(0, z)

    def drelu(self, z):
        return 1 * (z > 0)

    def leaky_relu(self, z):
        return max(0.01 * z, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores = None

        z1 = np.dot(X, W1) + b1
        a1 = self.relu(z1)
        scores = np.dot(a1, W2) + b2

        if y is None:
            return scores

        loss = None

        e_sum = np.sum(np.exp(scores), axis=1, keepdims=True)

        prob = np.exp(scores) / e_sum

        loss = -np.sum(np.log(prob[np.arange(N), y]))

        loss = (loss / N) + (reg * (np.sum(W1 * W1) + np.sum(W2 * W2)))

        grads = {}
        dz2 = prob
        dz2[range(N), y]-=1
        dz2 /= N

        grads['W2'] = np.dot(a1.T, dz2) + (reg * W2)
        grads['b2'] = np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, W2.T)
        drelu = (a1 > 0) * dz1
        grads['W1'] = np.dot(X.T, drelu) + (reg * W1)
        grads['b1'] = np.sum(drelu, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay
            
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
        
    def predict(self, X):
        y_pred = None

        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        a1 = self.relu(z1)
        scores = np.dot(a1, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        return y_pred