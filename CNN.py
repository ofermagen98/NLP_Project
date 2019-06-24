import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

class ConvolutionalNetwork(tf.keras.layers.Layer):
    """
    """
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.cnn = Sequential()
        self.cnn.add(Conv2D(64, kernel_size=3, activation='relu'))
        self.cnn.add(Conv2D(32, kernel_size=3, activation='relu'))

    def call(self,img,training = None):
        cnn_out = self.cnn(img,training=training)
        channels_n = cnn_out.get_shape().as_list()[-1]
        
        channels = tf.split(cnn_out,num_or_size_splits=channels_n,axis=-1)
        channels = [Flatten()(channel) for channel in channels]
        channels = tf.stack(channels,axis=1)
        
        return channels
