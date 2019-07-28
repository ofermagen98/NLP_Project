import json
import os
from progressbar import progressbar
import sys

#data sources
#data_dir = '/Users/ofermagen/Coding/NLP_Project_Data/formatted_images'
assert len(sys.argv) > 1
if sys.argv[1] == 'J':
    data_dir = '/specific/netapp5/joberant/home/ofermagen/semiformatted_images/train'
    model_path = '/specific/netapp5/joberant/home/ofermagen/checkpoints/model.{epoch:03d}.h5'
elif sys.argv[1] == 'O':
    data_dir = '/home/ofermagen/data/training_data_formatted/train/'
    model_path = '/home/ofermagen/checkpoints/model.{epoch:03d}.h5'
else:
    raise NotImplementedError
assert os.path.isdir(data_dir)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.callbacks  import ModelCheckpoint

from generator import DataGenerator
from RN import relation_product,ConvolutionalPerceptron,Perceptron
from resnet import ResnetV1_FCNN
from transformer import Encoder,create_padding_mask

#load params
with open(os.path.join(data_dir,'params.json'),'r') as f:
    input_vocab_size = json.load(f)
    input_vocab_size = input_vocab_size['vocab_size']
    
#defining model's inputs 
img_shape = (224,224,3) 
imgL = Input(shape=img_shape, name="imgL", dtype = "float32")
imgR = Input(shape=img_shape, name="imgR", dtype = "float32")
sent = Input(shape=(40,), name="sent", dtype = "int32")

#embedding images
fcnn = ResnetV1_FCNN(img_shape,20)
em_imgL = fcnn(imgL)
em_imgR = fcnn(imgR)
em_imgs = tf.keras.layers.Concatenate(axis=2)([em_imgL,em_imgR])

#embedding sentence
print('creating transformer encoder')
enc_mask = create_padding_mask(sent)
encoder = Encoder(num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=input_vocab_size)
em_sent = encoder(sent,training=True,mask=enc_mask)

#getting prediction from the Relational Neural Network 
print('creating relational network')
relation_matrix = relation_product(em_sent,em_imgs)
g = ConvolutionalPerceptron(relation_matrix.shape[1:],[256,256])
em_relations = g(relation_matrix)
relation_out = tf.reduce_mean(em_relations,axis=1)
f = Perceptron(relation_out.shape[1],[256,256])
relation_out = f(relation_out)
pred = Dense(1,activation='sigmoid')(relation_out)

#compile model
print('compiling model')
model = Model(inputs=[imgL,imgR,sent],outputs=pred)
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.load_weights(model_path)
checkpoint = ModelCheckpoint(filepath=model_path,monitor='acc',verbose=1,save_best_only=True,mode='max')
print('creating generators')
datagen = DataGenerator(data_dir)
callbacks = [checkpoint]

print('training model')
model.fit_generator(datagen, epochs=200, verbose=1, workers=4,callbacks=callbacks,shuffle=False)