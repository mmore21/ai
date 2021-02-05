import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class ConvNet(tf.keras.Model):
    def __init__(self, name="convnet", **kwargs):
        super(ConvNet, self).__init__(name=name, **kwargs)
        self.b1 = Conv3DBlock(filters=4, input_shape=(28, 28, 1))
        self.s1 = SEBlock(self.b1)
        self.b2 = Conv3DBlock(filters=8)
        self.s2 = SEBlock(self.b2)
        self.b3 = Conv3DBlock(filters=16)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = inputs
        x = self.b1(x)
        x = self.s1(x)
        x = self.b2(x)
        x = self.s2(x)
        x = self.global_pool(x)

        return self.classifier(x)