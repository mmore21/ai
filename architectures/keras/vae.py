import tensorflow as tf

class VariationalAutoEncoder(tf.keras.Model):
	def __init__(self):
		super(VariationalAutoEncoder, self).__init__()