"""
File: knn_classification.py
Author: Mason Moreland (mmore21)
Description: k-Nearest Neighbors (kNN) classifier implemented in Python 3.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

def knn(X, y, query, k):

    distances_and_indices = [(euclidian_distance(example, query), index) for index, example in enumerate(X)]

    k_nearest_distances_and_indices = sorted(distances_and_indices)[:k]

    k_nearest_labels = [y[i] for distance, i in k_nearest_distances_and_indices]

    return k_nearest_distances_and_indices, mode(k_nearest_labels)

def euclidian_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += math.pow(point2[i] - point1[i], 2)
    return math.sqrt(distance)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def plot_data(data, target):
    plt.scatter(data[:,0], data[:,1], c=target, marker='x')

    # Label plot title and axes
    plt.title("Admission from Exam Scores")
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.show()