import tensorflow as tf

class ConvBlock(tf.Keras.Layer):
	def __init__(self):
		super(ConvBlock, self).__init__()
		self.conv = 

class UNet(tf.Keras.Model):
	def __init__(self):
		super(UNet, self).__init__()
		self.block_1 = UNetBlock()
		self.block_2 = UNetBlock()
		self.global_pool = layers.GlobalAveragePooling2D()
		self.classifier = Dense(num_classes)
	
	def call(self, inputs):
		x = self.block_1(inputs)
		x = self.block_2(x)
		x = self.global_pool(x)
		return self.classifier(x)
	