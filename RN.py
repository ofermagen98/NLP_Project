"""This class include the basic Relational Neural Network Model""" 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class RelationalNetwork(tf.keras.layers.Layer):
    """
    """
    def __init__(self,d_o1,d_o2):
        super(RelationalNetwork, self).__init__()
        self.d_o1 = d_o1
        self.d_o2 = d_o2
        
        self.g = Sequential()
        self.g.add(Dense(256,input_dim=d_o1+d_o2))

        g_d_out = self.g.layers[-1].output_shape[1]
        self.f = Sequential()
        self.f.add(Dense(1,input_dim=g_d_out,activation='sigmoid'))

    def call(self,O1,O2):
        n_o1 = O1.get_shape().as_list()[1]
        n_o2 = O2.get_shape().as_list()[1]

        object_pairs_g = []
        for i in range(n_o1):
            o1i = O1[:,i,:]
            for j in range(n_o2):
                o2j = O2[:,j,:]
                opair = tf.concat([o1i,o2j],axis=1)
                g = self.g(opair)
                object_pairs_g.append(g)
        object_pairs_g = tf.stack(object_pairs_g, axis=0)
        g = tf.reduce_mean(object_pairs_g, axis=0)
        output = self.f(g)
        return output
