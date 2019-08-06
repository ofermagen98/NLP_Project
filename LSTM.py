from utils import tensorflow as tf
from tensorflow.keras.layers import LSTM,Bidirectional,TimeDistributed,Dense,Reshape
from tensorflow.keras.models import Sequential
from RN import Perceptron
from utils import DROPOUT_RATE

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, prec_params, input_vocab_size, size=40, word_dim=300, embeddings_initializer='uniform', **kwargs):
        super(Encoder,self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, word_dim, embeddings_initializer=embeddings_initializer
        )
        self._my_size = size
        self._my_units = units
        self._my_output_shape = prec_params[-1][0]
        self.prec = Perceptron(2*units,prec_params)
        self.lstm = Bidirectional(LSTM(units, return_sequences=True), input_shape=(size,word_dim))
        

    def call(self,sent):
        X = self.embedding(sent)
        X = self.lstm(X)
        X = tf.keras.backend.reshape(X,(-1,2*self._my_units))
        X = self.prec(X)
        X = tf.keras.backend.reshape(X,(-1,self._my_size,self._my_output_shape))
        return X

        