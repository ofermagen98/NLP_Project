import tensorflow as tf

from RN import RelationalNetwork
from CNN import ConvolutionalNetwork
from transformer import Encoder,create_padding_mask
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Concatenate
from tensorflow.keras.utils import plot_model

from utils import Pad
#python -m pip install tensorflow==2.0.0-beta1
#python -m pip install keras

#defining model's inputs 
imgL = Input(shape=(128, 128, 3), name="imgL", dtype = tf.float32)
imgR = Input(shape=(128, 128, 3), name="imgR", dtype = tf.float32)
sent = Input(shape=(40,), name="sent", dtype = tf.int32)

#embedding images
CNN = ConvolutionalNetwork()
em_imgL = CNN(imgL,training=True)
em_imgL = Pad(value = 1)(em_imgL)
em_imgR = CNN(imgR,training=True)
em_imgR = Pad(value = -1)(em_imgR)
em_imgs = Concatenate(1)([em_imgL,em_imgR])

#embedding sentence
input_vocab_size = 8500
enc_mask = create_padding_mask(sent)
encoder = Encoder(num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=input_vocab_size)
em_sent = encoder(sent,training=True,mask=enc_mask)

#getting prediction from the Relational Neural Network 
RN = RelationalNetwork(em_sent.shape[2],em_imgs.shape[2])
pred = RN([em_sent,em_imgs],training=True)

#compile model
model = Model(inputs=[imgL,imgR,sent],outputs=pred)
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
plot_model(model, to_file='model.png')
