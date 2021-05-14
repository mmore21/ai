"""
Architecture: Squeeze and Excitation Network
Paper: https://arxiv.org/pdf/1709.01507.pdf
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

import ConvBlock3D

class SEBlock3D(tf.keras.layers.Layer):
    def __init__(self, original, r=4):
        super(SEBlock3D, self).__init__()
        self.original = original
        self.p1 = tf.keras.layers.GlobalAvgPool3D()
        self.f1 = tf.keras.layers.Dense(int(original.input_shape[-1] / r))
        self.n1 = tf.keras.layers.BatchNormalization()
        self.r1 = tf.keras.layers.ReLU()
        self.scale1 = tf.keras.layers.Dense(original.input_shape[-1], activation='sigmoid')
        self.scale2 = tf.keras.layers.Reshape((1, 1, 1, original.shape[-1]))

    def call(self, inputs):
        x = inputs
        x = self.p1(x)
        x = self.f1(x)
        x = self.n1(x)
        x = self.r1(x)
        x = self.scale1(x)
        x = self.scale2(x)
        x = self.original * x
        return x

class SENet3D(tf.keras.Model):
    def __init__(self, num_classes):
        super(SENet3D, self).__init__()
        self.convblock1 = ConvBlock3D(filters=16)
        self.seblock1 = SEBlock3D(self.convblock1)
        self.convblock2 = ConvBlock3D(filters=32)
        self.seblock2 = SEBlock3D(self.convblock2)
        self.convblock3 = ConvBlock3D(filters=64)
        self.final_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.final_classifier = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        x = self.b1(x)
        x = self.s1(x)
        x = self.b2(x)
        x = self.s2(x)
        x = self.final_pool(x)
        x = self.final_classifier(x)
        return x

if __name__ == "__main__":
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