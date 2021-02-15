"""
Architecture: Densely Connected Convolutional Network (DenseNet)
Paper: https://arxiv.org/pdf/1608.06993v5.pdf
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class DenseNetBlock(tf.keras.layers.Layer):
	def __init__(self):
		super(DenseNetBlock, self).__init__()
	
	def call(self, inputs):
		pass
		
	
class DenseNet(tf.keras.Model):
	def __init__(self):
		super(DenseNet, self).__init__()
	
	def call(self, inputs):
		pass

if __name__ == "__main__":
	pass