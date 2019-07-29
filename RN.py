"""This class include the basic Relational Neural Network Model"""
from utils import DROPOUT_RATE, DROPOUT_BOOL

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Input, Dropout


class ReduceMean(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(ReduceMean, self).__init__()
        self.axis = axis

    def call(self, X):
        return tf.reduce_mean(X, axis=self.axis)


class RelationalProduct(tf.keras.layers.Layer):
    def __init__(self):
        super(RelationalProduct, self).__init__()

    def call(self, X):
        x1, x2 = X
        assert len(x1.shape) == 3 and len(x2.shape) == 3
        n1 = int(x1.shape[1])
        n2 = int(x2.shape[1])
        O1 = tf.expand_dims(x1, axis=1)
        O1 = tf.tile(O1, multiples=(1, n2, 1, 1))
        O2 = tf.expand_dims(x2, axis=2)
        O2 = tf.tile(O2, multiples=(1, 1, n1, 1))
        relation_matrix = tf.concat([O1, O2], axis=3)
        d = int(relation_matrix.shape[3])
        relation_matrix = tf.reshape(
            relation_matrix, shape=(-1, n1 * n2, d), name="relation_matrix"
        )
        return relation_matrix


class ConvolutionalPerceptron(tf.keras.layers.Layer):
    def __init__(self, input_shape, layer_dims):
        super(ConvolutionalPerceptron, self).__init__()
        self._input_shape = input_shape
        self.model = Sequential()
        for i, dim in enumerate(layer_dims):
            if i == 0:
                self.model.add(
                    Conv1D(filters=dim, kernel_size=1, input_shape=input_shape)
                )
            else:
                self.model.add(Conv1D(filters=dim, kernel_size=1))

    def call(self, x):
        assert len(x.shape) == 3
        assert x.shape[1:] == self._input_shape
        return self.model(x)


class Perceptron(tf.keras.layers.Layer):
    def __init__(self, input_dim, layer_dims, dropout=DROPOUT_BOOL):
        super(Perceptron, self).__init__()
        self._input_dim = input_dim
        self.model = Sequential()
        for i, dim in enumerate(layer_dims):
            if i == 0:
                self.model.add(Dense(units=dim, input_shape=(input_dim,)))
            else:
                self.model.add(Dense(units=dim))
            if dropout:
                self.model.add(Dropout(rate=DROPOUT_RATE))

    def call(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self._input_dim
        return self.model(x)
