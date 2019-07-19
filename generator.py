import os
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import json
from random import shuffle as _shuffle
from copy import deepcopy
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_images(dir_path,suffix):
    paths = os.listdir(dir_path)
    paths = filter(lambda s: s[-len(suffix):] == suffix, paths)
    paths = map(lambda s: os.path.join(dir_path,s), paths)
    paths = list(paths)
    key = lambda s: int(os.path.basename(s)[:-len(suffix)])
    paths.sort(key = key)
    images = [np.array(Image.open(im)) for im in paths]
    images = np.stack(images)
    return images

class DataGenerator(Sequence):
    '''
    '''
    def __init__(self, ddir, batch_size = 32, augmentation = True):
        assert os.path.isdir(ddir)
        super(DataGenerator,self).__init__()
        self.ddir = ddir
        self.batch_size = batch_size
        self.last_dir = None

        self.genL = ImageDataGenerator(
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
        self.genR = deepcopy(self.genL)

        self.dirnum = os.listdir(ddir)
        self.dirnum = map(lambda s: os.path.join(ddir,s), self.dirnum)
        self.dirnum = filter(os.path.isdir, self.dirnum)
        self.dirnum = len(list(self.dirnum))

        suffix = "-img0.png"
        def dir_size(i,ddir,suffix):
            lst = os.listdir(os.path.join(ddir,str(i)))
            lst = filter(lambda s: s[-len(suffix):] == suffix,lst)
            return sum(1 for _ in lst)

        self.sizes = map(lambda i: dir_size(i,ddir,suffix), range(self.dirnum))
        self.sizes = list(self.sizes)

        self.len = 0
        self.files = []
        self.indexs = []
        for i,sz in enumerate(self.sizes):
            for j in range(0,sz,self.batch_size):
                self.len += 1
                self.files.append(i)
                self.indexs.append(j // self.batch_size)

        self.files = np.array(self.files,dtype=np.dtype('int32'))
        self.indexs = np.array(self.indexs,dtype=np.dtype('int32'))

        sampled_images = 0
        sampled_images = os.path.join(ddir,str(sampled_images))
        sampled_images = get_images(sampled_images,"-img0.png")

        print('fitting generators')
        self.genL.fit(sampled_images)
        self.genR.fit(sampled_images)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def load_folder(self,j):
        dir_path = os.path.join(self.ddir,str(j))
        with open(os.path.join(dir_path,'data.json'), 'r') as f:
            data = json.load(f)
            sz = self.sizes[j]
            assert len(data) == sz

            self.sentence = [np.array(d['sentence']) for d in data]
            self.sentence = [np.stack(self.sentence[i:i+self.batch_size]) for i in range(0,sz,self.batch_size)]

            self.label = [bool(d['label']) for d in data]
            self.label = [np.stack(self.label[i:i+self.batch_size]) for i in range(0,sz,self.batch_size)]

        imgL = get_images(dir_path,"-img0.png")
        self.imgL = self.genL.flow(imgL,y=None,batch_size=self.batch_size,shuffle=False)
        imgR = get_images(dir_path,"-img1.png")
        self.imgR = self.genR.flow(imgR,y=None,batch_size=self.batch_size,shuffle=False)
        self.last_dir  = j

    def __getitem__(self, index):
        'Generate one batch of data'
        file = self.files[index]
        if self.last_dir != file: self.load_folder(file)
        index = self.indexs[index]
        print(index)
        return [self.imgL[index], self.imgR[index], self.sentence[index]], self.label[index]

