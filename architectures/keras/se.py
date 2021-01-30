import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, original, r=4, name="se_block", **kwargs):
        super(SEBlock, self).__init__(name=name, **kwargs)
        self.original = original
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
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
        return self.original * x