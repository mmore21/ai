"""
Architecture: U-Net
Paper: https://arxiv.org/pdf/1505.04597.pdf
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class UNetBlock(tf.keras.layers.Layer):
	def __init__(self):
		super(UNetBlock, self).__init__()
	
	def call(self, inputs):
		pass


class UNet(tf.keras.Model):
	def __init__(self):
		super(UNet, self).__init__()
	
	def call(self, inputs):
		pass

if __name__ == "__main__":
	pass