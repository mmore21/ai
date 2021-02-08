import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import SEBlock

class Conv3DBlock(tf.keras.layers.Layer):
    def __init__(self, name="conv3d_block", kernel_size=(1, 3, 3), filters=2, stride=1, padding="same", input_shape=None, **kwargs):
        super(Conv3DBlock, self).__init__(name=name, **kwargs)
        self.relu = tf.keras.layers.ReLU()
        self.norm = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv3D(
            kernel_size=kernel_size,
            filters=filters,
            strides=stride,
            padding=padding
        )
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return self.relu(x)

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