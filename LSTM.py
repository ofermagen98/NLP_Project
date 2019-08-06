from utils import tensorflow as tf
from tensorflow.keras.layers import LSTM,Bidirectional,TimeDistributed
from tensorflow.keras.models import Sequential
from utils import DROPOUT_RATE

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units,  input_vocab_size, output_dim = 256, word_dim=300, embeddings_initializer='uniform', **kwargs):
        super(Encoder,self).__init__()
        embedding = tf.keras.layers.Embedding(
            input_vocab_size, word_dim, embeddings_initializer=embeddings_initializer
        )

        self.model = Sequential()
        self.model.add(embedding)
        self.model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=(word_dim,)))
        self.model.add(TimeDistributed(Dense(256, activation='sigmoid')))
    
    def call(sent):
        return self.model(sent)

        