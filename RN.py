"""This class include the basic Relational Neural Network Model""" 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Dense,Input,Concatenate,Lambda,Flatten

def relation_product(x1,x2):
    n1 = int(x1.shape[1])
    n2 = int(x2.shape[1])
    O1 = tf.expand_dims(x1,axis=1)
    O1 = tf.tile(O1,multiples=(1,n2,1,1))
    O2 = tf.expand_dims(x2,axis=2)
    O2 = tf.tile(O2,multiples=(1,1,n1,1))
    relation_matrix = tf.concat([O1,O2],axis=3)
    d = int(relation_matrix.shape[3])
    relation_matrix = tf.reshape(relation_matrix,shape=(-1,n1*n2,d),name="relation_matrix")
    return relation_matrix

class ConvolutionalPerceptron(tf.keras.layers.Layer):
    def __init__(self,input_shape,layer_dims):
        super(ConvolutionalPerceptron,self).__init__()
        self.model = Sequential()
        for i,dim in enumerate(layer_dims):
            if i==0:
                self.model.add(Conv1D(filters=dim,kernel_size=1,input_shape=input_shape))
            else:
                self.model.add(Conv1D(filters=dim,kernel_size=1))
    
    def call(self,x):
        return self.model(x)


class Perceptron(tf.keras.layers.Layer):
    def __init__(self,input_dim,layer_dims):
        super(Perceptron,self).__init__()
        self.model = Sequential()
        for i,dim in enumerate(layer_dims):
            if i==0:
                self.model.add(Dense(units=dim,input_shape=(input_dim,)))
            else:
                self.model.add(Dense(units=dim))
    
    def call(self,x):
        return self.model(x)

'''class RelationalNetwork(tf.keras.layers.Layer):
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


    def call(self,Os,training = True, use_pbar=False):
        assert len(Os) == 2
        assert len(Os[0].shape) == 3 and len(Os[1].shape) == 3
        assert Os[0].shape[2] == self.d_o1 and Os[1].shape[2] == self.d_o2

        n_o1 = int(Os[0].shape[1])
        n_o2 = int(Os[1].shape[1])

        from itertools import product 
        O1 = map(lambda i : Os[0][:,i,:], range(n_o1))
        O2 = map(lambda i : Os[1][:,i,:], range(n_o2))
        g = None

        prod = product(O1,O2)
        if use_pbar: 
            from tqdm import tqdm_notebook
            pbar=tqdm_notebook(range(n_o1*n_o2))
            prod = zip(pbar,prod)
        else:
            prod = zip(range(n_o1*n_o2),prod)

        for _,(o1,o2) in prod:
            if g is None: g = self.g([o1,o2])
            else: g += self.g([o1,o2])
        g /= n_o1*n_o2

        if use_pbar:
            pbar.close()

        output = self.f(g, training=training)
        #output = tf.squeeze(output)
        return output
'''