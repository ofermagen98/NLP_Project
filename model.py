import tensorflow as tf
from RN import RelationalNetwork
from CNN import ConvolutionalNetwork
from transformer import Encoder,create_padding_mask

#python3 -m pip install tensorflow
#python3 -m pip install keras-layer-normalization

img1 = tf.placeholder(dtype=tf.float32,shape=[None, 128, 128, 3])
img2 = tf.placeholder(dtype=tf.float32,shape=[None, 128, 128, 3])
sent = tf.placeholder(dtype=tf.int32,shape=[None, 40])
enc_mask = create_padding_mask(sent)

input_vocab_size = 8500
cnn = ConvolutionalNetwork()

embeded_img1 = cnn(img1)
embeded_img2 = cnn(img2)
embeded_imgs = tf.concat([embeded_img1,embeded_img2],axis = 1)

#encoder = Encoder(num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=input_vocab_size)
#embeded_sent = encoder(sent,True,enc_mask)
