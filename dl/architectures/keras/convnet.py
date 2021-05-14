"""
Architecture: Convolutional Neural Network (CNN)
Paper: N/A
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class ConvBlock2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.keras.layers.LeakyReLU()):
        super(ConvBlock2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding
        )
        self.norm = tf.keras.layers.BatchNormalization()
        self.acti = activation
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.acti(x)
        return x

class ConvBlock3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation=tf.keras.layers.LeakyReLU()):
        super(ConvBlock3D, self).__init__()
        self.conv = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding
        )
        self.norm = tf.keras.layers.BatchNormalization()
        self.acti = activation
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.acti(x)
        return x

class ConvNet2D(tf.keras.Model):
    def __init__(self):
        super(ConvNet2D, self).__init__()
        pass
    
    def call(self, inputs):
        pass

class ConvNet3D(tf.keras.Model):
    def __init__(self):
        super(ConvNet3D, self).__init__()
        pass
    
    def call(self, inputs):
        pass