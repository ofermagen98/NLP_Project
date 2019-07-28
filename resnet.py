from utils import DROPOUT_RATE, DROPOUT_BOOL 

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, Conv2D
from tensorflow.keras.layers import Dropout,BatchNormalization,Flatten
from tensorflow.keras.layers import Activation,AveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow import keras

#https://keras.io/examples/cifar10_resnet/

def resnet_layer(inputs,
                num_filters=16,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_normalization=True,
                conv_first=True,
                droput=DROPOUT_BOOL):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-5)
            )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    if droput:
        x = Dropout(rate= DROPOUT_RATE)(x)

    return x


class ResnetV1_FCNN(tf.keras.layers.Layer):
    """
    """
    
    def __init__(self,input_shape,depth):
        super(ResnetV1_FCNN, self).__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                    num_filters=num_filters,
                                    strides=strides)
                y = resnet_layer(inputs=y,
                                    num_filters=num_filters,
                                    activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                        num_filters=num_filters,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=None,
                                        batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        fcnn_out = AveragePooling2D(pool_size=8)(x)
        shape = fcnn_out.shape.as_list()

        fcnn_out = tf.reshape(fcnn_out,(-1,shape[1]*shape[2],shape[3]))
        self.model = Model(inputs=inputs, outputs=fcnn_out)

    def load_weights(self,file):
        raise NotImplementedError

    def call(self,img,training = None):
        return self.model(img,training=training)
