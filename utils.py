import tensorflow as tf
import json
import os

DROPOUT_BOOL = True
DROPOUT_RATE = 0.5

class HistorySaver(tf.keras.callbacks.Callback):
	def __init__(self,fname,save_every = 32,*a,**kw):
		super(HistorySaver,self).__init__(*a,**kw)
		self.fname = fname
		self.save_every = save_every

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accs = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(float(logs.get('loss')))
		self.accs.append(float(logs.get('acc')))

		if len(self.losses) % self.save_every == 0:
			with open(self.fname,'w') as f:
				json.dump({'acc' : self.accs, 'loss' : self.losses},f)

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