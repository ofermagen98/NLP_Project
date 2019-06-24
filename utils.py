import tensorflow as tf

class Pad(tf.keras.layers.Layer):
	"""
	"""
	def __init__(self,value = 0,axis = -1):
		super(Pad, self).__init__()
		self.value = value
		self.axis = axis

	def call(self,tensor):
		n_shape = tensor.shape.as_list()
		n_shape[self.axis] += 1
		n_shape = [x if x is not None else -1 for x in n_shape]
		padding = [[0,0] for _ in n_shape]
		padding[self.axis] = [0,1]
		res = tf.pad(tensor,padding,mode='CONSTANT', constant_values=self.value)
		return tf.reshape(res,n_shape)