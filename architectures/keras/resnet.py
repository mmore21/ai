"""
Architecture: Residual Neural Network (ResNet)
Paper: https://arxiv.org/pdf/1512.03385.pdf
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

import ConvBlock2D

class ResBlock2D(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResBlock2D, self).__init__()
        self.convblock1 = ConvBlock2D(filters=filters[0])
        self.convblock2 = ConvBlock2D(filters=filters[1])
        self.convblock3 = ConvBlock2D(filters=filters[2])
        self.proj = tf.keras.layers.Conv2D(
            filters = filters[1],
            kernel_size = (1, 1),
            padding = 'same'
        )
    
    def call(self, inputs):
        x = self.convblock1(inputs)
        x = self.convblock2(x)
        x = self.convblock3(x + self.proj(inputs))
        return x

class ResNet2D(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet2D, self).__init__()
        self.conv_initial = ConvBlock2D(filters=16, kernel_size=(7,7), strides=(2, 2))
        self.pool_initial = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        self.resblock1 = ResBlock2D([32, 32, 64])
        self.resblock2 = ResBlock2D([128, 128, 256])
        self.resblock3 = ResBlock2D([128, 256, 512])
        self.pool_final = tf.keras.layers.GlobalAvgPool2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation="sigmoid")
    
    def call(self, inputs):
        x = self.conv_initial(inputs)
        x = self.pool_initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.pool_final(x)
        x = self.classifier(x)
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
    resnet = ResNet2D(num_classes=10)
    resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                    loss=tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True),
                    metrics=["accuracy"],
                    experimental_run_tf_function=False)

    # Train model
    resnet.fit(
        x=x_train,
        y=y_train,
        steps_per_epoch=100,
        epochs=10
    )

    # Evaluate model on test set
    score = resnet.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Visualize model predictions
    for i in range(10):
        plt.imshow(x_test[i, ..., 0])
        plt.show()
        logits = resnet.predict(np.expand_dims(x_test[i], axis=0))
        pred = np.argmax(logits, axis=-1)[0]
        print("ResNet:", pred, "\nActual:", y_test[i])