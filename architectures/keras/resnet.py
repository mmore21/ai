import tensorflow as tf

class ResNetBlock(tf.keras.layers.Layer):
	def __init__(self):
		super(ResNetBlock, self).__init__()
	
	def call(self, inputs):
		pass


class ResNet(tf.keras.Model):
	def __init__(self):
		super(ResNet, self).__init__()
		self.block_1 = ResNetBlock()
		self.block_2 = ResNetBlock()
		self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
		self.classifier = tf.keras.layers.Dense(num_classes)
	
	def call(self, inputs):
		x = self.block_1(inputs)
		x = self.block_2(x)
		x = self.global_pool(x)
		return self.classifier(x)
