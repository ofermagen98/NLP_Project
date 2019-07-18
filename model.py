import json
import os

from generator import DataGenerator
import tensorflow as tf

from RN import RelationalNetwork,objectify
from resnet import ResnetV1_FCNN
from transformer import Encoder,create_padding_mask
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Concatenate,Reshape
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks  import ModelCheckpoint

data_dir = 'formatted'
with open(os.path.join(data_dir,'params.json'),'r') as f:
    input_vocab_size = json.load(f)
    input_vocab_size = input_vocab_size['vocab_size']
    
#defining model's inputs 
img_shape = (224,224,3) 
imgL = Input(shape=img_shape, name="imgL", dtype = tf.float32)
imgR = Input(shape=img_shape, name="imgR", dtype = tf.float32)
sent = Input(shape=(40,), name="sent", dtype = tf.int32)

#embedding images
fcnn = ResnetV1_FCNN(img_shape,20)
em_imgL = fcnn(imgL)
em_imgR = fcnn(imgR)
em_imgs = Concatenate(2)([em_imgL,em_imgR])

#embedding sentence
print('creating transformer encoder')
enc_mask = create_padding_mask(sent)
encoder = Encoder(num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=input_vocab_size)
em_sent = encoder(sent,training=True,mask=enc_mask)

#getting prediction from the Relational Neural Network 
print('creating relational network')
RN = RelationalNetwork(em_sent.shape[2],em_imgs.shape[2])
pred = RN([em_sent,em_imgs],training=True)

#compile model
print('compiling model')
model = Model(inputs=[imgL,imgR,sent],outputs=pred)
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model_path = 'checkpoints/first_model.{epoch:03d}.h5'
checkpoint = ModelCheckpoint(filepath=model_path,save_best_only=True)
datagen = DataGenerator(data_dir)
callbacks = [checkpoint]
model.fit_generator(datagen, epochs=200, verbose=1, workers=4,callbacks=callbacks)