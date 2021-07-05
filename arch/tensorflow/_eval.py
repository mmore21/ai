"""
Description: Common helper functions used for Keras models.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from resnet import ResNet2D

def evaluate_mnist(model_path):
    # Load data set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data by rescaling the images from [0, 255] to the [0.0, 1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))
    print(x_train.shape, y_train.shape)

    # Create and compile model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Evaluate model on test set
    score = model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Visualize model predictions
    for i in range(10):
        plt.imshow(x_test[i, ..., 0])
        plt.show()
        logits = model.predict(np.expand_dims(x_test[i], axis=0))
        pred = np.argmax(logits, axis=-1)[0]
        print("Prediction:", pred, "\nActual:", y_test[i])
