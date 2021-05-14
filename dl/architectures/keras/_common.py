"""
Description: Common helper functions used for Keras models.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# TODO: Accept a model and evaluate on the MNIST data set
def evaluate_mnist(model):
    # Load data set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data by rescaling the images from [0, 255] to the [0.0, 1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    print("Number of filtered training examples:", len(x_train))
    print("Number of filtered test examples:", len(x_test))
    print(x_train.shape, y_train.shape)

    # Create and compile model
    senet = SENet3D(num_classes=10)
    senet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                    loss=tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True),
                    metrics=["accuracy"],
                    experimental_run_tf_function=False)

    # Train model
    senet.fit(
        x=x_train,
        y=y_train,
        steps_per_epoch=100,
        epochs=10
    )

    # Evaluate model on test set
    score = senet.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Visualize model predictions
    for i in range(10):
        plt.imshow(x_test[i, ..., 0])
        plt.show()
        logits = senet.predict(np.expand_dims(x_test[i], axis=0))
        pred = np.argmax(logits, axis=-1)[0]
        print("SENet:", pred, "\nActual:", y_test[i])