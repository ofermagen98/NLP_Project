import os
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import json
from random import shuffle as _shuffle
from copy import deepcopy
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#TODO shuffle
class DataGenerator(Sequence):
    '''
    '''
    def __init__(self, ddir, batch_size = 16,  shuffle = True, augmentation = True):
        assert os.path.isdir(ddir)
        super(DataGenerator,self).__init__()
        self.ddir = ddir
        self.batch_size = batch_size
        self.last_dir = None

        self.gen = ImageDataGenerator(
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

        self.dir_num = os.listdir(ddir)
        self.dir_num = map(lambda s: os.path.join(ddir,s), self.dir_num)
        self.dir_num = filter(os.path.isdir, self.dir_num)
        self.dir_num = len(list(self.dir_num))

        self.sizes = []
        self.sentences = []
        self.labels = []
        self.paths = []

        for i in range(self.dir_num):
            path = os.path.join(ddir,str(i))
            with open(os.path.join(path,"data.json"),"r") as f:
                data = json.load(f)
            self.sizes.append(len(data))
            for i,d in enumerate(data):
                self.sentences.append(d['sentence'])
                self.labels.append(d['label'])
                self.paths.append(os.path.join(path,str(i)))
            
        self.sizes = np.array(self.sizes)
        self.sentences = np.array(self.sentences)
        self.labels = np.array(self.labels)

        self.sample_num = len(self.labels)
        self.batch_num = len(range(0,self.sample_num,batch_size))

        if shuffle: self.shuffle()

        sampled_images = range(0, min(1000, self.sample_num))
        sampled_images = map(lambda i: self.paths[i] + "-img0.png",sampled_images)
        sampled_images = np.stack([np.array(Image.open(path)) for path in sampled_images])
        self.gen.fit(sampled_images)


    def shuffle(self):
        self.permutation = np.random.permutation(self.sample_num)
        self.sentences =  self.sentences[self.permutation,:]
        self.labels = self.labels[self.permutation]
        self.paths = [self.paths[i] for i in self.permutation]

    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return self.batch_num

    def __getitem__(self, index):
        'Generate one batch of data'
        idx = min((index+1)*self.batch_size, self.sample_num)
        idx = range(index*self.batch_size, idx)
        idx = np.array(list(idx))

        imgL = map(lambda i: self.paths[i] + "-img0.png",idx)
        imgL = np.stack([np.array(Image.open(path)) for path in imgL])
        imgR = map(lambda i: self.paths[i] + "-img1.png",idx)
        imgR = np.stack([np.array(Image.open(path)) for path in imgR])

        return [imgL, imgR, self.sentences[idx,:]], self.labels[idx]

if __name__ == "__main__":
    gen = DataGenerator('/Users/ofermagen/Coding/NLP_Project_Data/formatted_images')
    for i,x in enumerate(gen):
        print(i,x[0][0].shape)
