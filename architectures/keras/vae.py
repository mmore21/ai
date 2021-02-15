import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class Sampling(tf.keras.layers.Layer):
	def call(self, inputs):
		pass


class Encoder(tf.keras.layers.Layer):
	def __init__(self):
		super(Encoder, self).__init__()
		
	def call(self, inputs):
		pass


class Decoder(tf.keras.layers.Layer):
	def __init__(self):
		super(Decoder, self).__init__()

	def call(self, inputs):
		pass


class VariationalAutoEncoder(tf.keras.Model):
	def __init__(self):
		super(VariationalAutoEncoder, self).__init__()
	
	def call(self, inputs):
		pass

if __name__ == "__main__":
	pass