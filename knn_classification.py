"""
File: knn_classification.py
Author: Mason Moreland (mmore21)
Description: k-Nearest Neighbors (kNN) classifier implemented in Python 3.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def knn():
    return None

def euclidian_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(distance)

if __name__ == "__main__":
    distance = euclidian_distance([1, 2], [3, 4])
    print(distance)