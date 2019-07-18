import os
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import json
from random import shuffle as _shuffle
from copy import deepcopy

class DataGenerator(Sequence):
    '''
    '''
    def __init__(self, ddir,batch_size = 32,batch_norm = True):
        assert os.path.isdir(ddir)
        super(DataGenerator,self).__init__()
        self.ddir = ddir
        self.batch_size = batch_size
        self.batch_norm = batch_norm

        with open(os.path.join(ddir,'params.json'),'r') as f:
            self.params = json.load(f)

        self.sizes = self.params['sizes']
        self.shuffle()
        self.loaded_file_idx = -1

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        super_batch_order = list(range(len(self.sizes)))
        _shuffle(super_batch_order)

        self.file = []
        self.idxs = []

        for super_batch in super_batch_order:
            sz =  self.sizes[super_batch]
            batch_order = list(range(sz))
            _shuffle(batch_order)
            for i in range(0,sz,self.batch_size):
                self.idxs.append(batch_order[i:i+self.batch_size])
                self.file.append(super_batch)
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.idxs)

    def load_file(self,j):
        with open(os.path.join(self.ddir,'imgL-%d.npy' % j),'rb') as f:
            self.imgL = np.load(f)
        with open(os.path.join(self.ddir,'imgR-%d.npy' % j),'rb') as f:
            self.imgR = np.load(f)
        with open(os.path.join(self.ddir,'sentence-%d.npy' % j),'rb') as f:
            self.sentence = np.load(f)
        with open(os.path.join(self.ddir,'label-%d.npy' % j),'rb') as f:
            self.label = np.load(f)
        self.loaded_file_idx = j

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.loaded_file_idx != self.file[index]:
            self.load_file(self.file[index])
        
        idxs = self.idxs[index]
        imgL = self.imgL[idxs,:,:,:]
        imgR = self.imgR[idxs,:,:,:]
        sentence = self.sentence[idxs,:]
        label = self.label[idxs]

        imgL = imgL.astype(np.dtype('float32'))
        imgR = imgR.astype(np.dtype('float32'))

        if self.batch_norm:
            imgL -= np.mean(imgL)
            imgL /= np.linalg.norm(imgL)
            imgR -= np.mean(imgR)
            imgR /= np.linalg.norm(imgR)
            
        return [imgL,imgR,sentence], label
