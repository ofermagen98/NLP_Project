import json
import os
from progressbar import progressbar
from PIL import Image
import numpy as np

#data sources
orig_dir = '/Users/ofermagen/Coding/NLP_Project_Data/'
orig_dir = '/home/ofermagen/'
data_dir = orig_dir + 'pretraining_data'
sample_dir = orig_dir + 'formatted_images/0'
assert os.path.isdir(data_dir)
assert os.path.isdir(sample_dir)

with open(os.path.join(data_dir,"synset2num.json")) as f:
    classes = json.load(f)
    class_num = len(classes)

import tensorflow as tf
from resnet import ResnetV1_FCNN
from RN import Perceptron
from utils import HistorySaver

from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks  import ModelCheckpoint
from tensorflow.keras.models import Model

#defining model's inputs 
img_shape = (224,224,3) 
img = Input(shape=img_shape, name="img", dtype = tf.float32)

#embedding images
fcnn = ResnetV1_FCNN(img_shape,20)
em_img = fcnn(img)
em_img = Flatten()(em_img)

preceptron = Perceptron(em_img.shape[1],[1024,512,512])
X = preceptron(em_img)
pred = Dense(class_num,activation='softmax')(X)

model = Model(inputs=img,outputs=pred)
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_path = 'pretrained_cnn/checkpoints/cnn_model.{epoch:03d}.h5'
history_path = 'pretrained_cnn/checkpoints/history.json'

gen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=True,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=True,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

suffix = ".png"
sample_images = os.listdir(sample_dir)
sample_images = filter(lambda p: len(p) > len(suffix) and p[-len(suffix):] == suffix, sample_images)
sample_images = map(lambda s: os.path.join(sample_dir,s), sample_images)
sample_images = list(sample_images)[:1000]
sample_images = np.stack([Image.open(path) for path in sample_images])
gen.fit(sample_images)

gen = gen.flow_from_directory(data_dir,batch_size=32,class_mode='categorical',target_size=img_shape[:2])

checkpoint = ModelCheckpoint(filepath=model_path,monitor='acc',verbose=1,save_best_only=True,mode='max')
history_saver = HistorySaver(history_path)
callbacks = [checkpoint,history_saver]

model.fit_generator(gen, epochs=200, verbose=1, workers=4,callbacks=callbacks,shuffle=False)