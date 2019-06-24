"""This class include the basic Relational Neural Network Model""" 
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input,Concatenate

class RelationalNetwork(tf.keras.layers.Layer):
    """
    """
    def __init__(self,d_o1,d_o2):
        super(RelationalNetwork, self).__init__()
        self.d_o1 = d_o1
        self.d_o2 = d_o2
        
        o1 = Input((d_o1,))
        o2 = Input((d_o2,))
        concat = Concatenate()([o1,o2])
        g_dns1 = Dense(256)(concat)
        self.g = Model(inputs = [o1,o2], outputs=g_dns1)
        self.g_d_out = self.g.layers[-1].output_shape[1]

        self.f = Sequential()
        self.f.add(Dense(256,input_dim=self.g_d_out))
        self.f.add(Dense(1,activation='sigmoid'))


    def call(self,Os,training = True):
        assert len(Os) == 2
        assert len(Os[0].shape) == 3 and len(Os[1].shape) == 3
        assert Os[0].shape[2] == self.d_o1 and Os[1].shape[2] == self.d_o2

        n_o1 = int(Os[0].shape[1])
        n_o2 = int(Os[1].shape[1])

        from itertools import product 
        O1 = map(lambda i : Os[0][:,i,:], range(n_o1))
        O2 = map(lambda i : Os[1][:,i,:], range(n_o2))
        g = None
        for o1,o2 in product(O1,O2):
            if g is None: g = self.g([o1,o2])
            else: g += self.g([o1,o2])
        g /= (n_o1 * n_o2)

        output = self.f(g, training=training)
        #output = tf.squeeze(output)
        return output
