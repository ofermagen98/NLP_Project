import cv2
from PIL import Image
from time import sleep
import json
import os
import imagehash
import numpy as np
from tqdm import tqdm as progressbar
from random import shuffle as _shuffle
import pickle
import sys

from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

with open("./word_embeddings/word2num.json", "r") as f:
    word2num = json.load(f)
    word2num = {w: i for i, w in enumerate(word2num)}


def sent2list(sent, size=40):
    global word2num
    replace = lambda c: c if c.isdigit() or c.isalpha() else " "
    sent = "".join(map(replace, sent))
    sent = sent.split(" ")
    sent = filter(lambda w: len(w) > 0, sent)
    sent = list(sent)
    tonum = lambda w: word2num["unk"] if w not in word2num else word2num[w]
    sent = list(map(tonum, sent[:size]))
    for _ in range(len(sent), size):
        sent.append(0)
    return list(sent)


def read_features(imgL, imgR, size=30):
    with open(imgL, "rb") as f:
        imgL = pickle.load(f)
    with open(imgR, "rb") as f:
        imgR = pickle.load(f)

    OBJ = dict()
    OBJ['features'] = np.concatenate([imgL['features'], imgR['features']])
    OBJ["img_side"] = [1] * len(imgL["features"]) + [-1] * len(imgR["features"])
    OBJ["img_side"] = np.asarray(OBJ["img_side"], dtype=np.dtype("float32"))

    if len(OBJ["features"]) > size:
        perm = list(range(len(OBJ["features"])))
        perm.sort(key=lambda i: OBJ["features"][i][-1], reverse=True)
        perm = perm[:size]
        for key in OBJ.keys():
            OBJ[key] = OBJ[key][perm]

    count = len(OBJ["features"])
    count = size - count

    if count > 0:
        for key in OBJ.keys():
            A = np.zeros(OBJ[key].shape[1:], dtype=OBJ[key].dtype)
            A = np.stack([A] * count)
            OBJ[key] = np.concatenate([OBJ[key], A])

    return OBJ


class DataGenerator(Sequence):
    """
    """

    def __init__(self, json_file, ddir, batch_size=16, shuffle=True):
        assert os.path.isdir(ddir)
        super(DataGenerator, self).__init__()
        self.ddir = ddir
        self.batch_size = batch_size
        self.examples = [json.loads(s) for s in open(json_file).readlines()]
        
        if sys.argv[1] == "checkpoints_single":
            self.examples = list(filter(lambda ex: ex['synset'] == 'gorilla',self.examples))
        self.batch_num = (len(self.examples) + batch_size - 1) // batch_size

        with open(os.path.join(ddir, "ID2Path.json"), "r") as f:
            self.ID2path = json.load(f)

        if shuffle:
            _shuffle(self.examples)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.batch_num

    def read_example(self, idx):
        ex = self.examples[idx]
        ID = ex["identifier"]
        ID = "-".join(ID.split("-")[:3])
        sent = ex["sentence"]
        sent = sent2list(sent)

        imgL = self.ID2path[ID + "-img0.png"]
        imgL = self.ddir + os.path.basename(imgL)
        imgR = self.ID2path[ID + "-img1.png"]
        imgR = self.ddir + os.path.basename(imgR)

        OBJ = read_features(imgL, imgR)
        OBJ["sent"] = np.asarray(sent)
        #OBJ["label"] = [0.0,0.0]
        OBJ["label"] = int(ex["label"][0] == 'T')

        OBJ["ID"] = ID
        OBJ["synset"] = ex["synset"]
        return OBJ

    def __getitem__(self, index):
        "Generate one batch of data"
        idx = min((index + 1) * self.batch_size, len(self.examples))
        idx = range(index * self.batch_size, idx)
        idx = list(idx)
        
        #keys = ["features", "img_side", "sent"] 
        labels = []
        features = []
        sides = []
        sents = []

        for i in idx:
            OBJ = self.read_example(i)
            features.append(OBJ["features"])
            sides.append(OBJ["img_side"])
            labels.append(OBJ["label"])
            sents.append(OBJ["sent"]) 
        
        features = np.asarray(features,dtype=np.dtype('float32'))
        sides = np.asanyarray(sides,dtype=np.dtype('float32'))
        labels = np.asarray(labels,dtype=np.dtype('float32'))
        sents = np.asarray(sents,dtype=np.dtype('int32'))
        return [features , sides, sents], labels


if __name__ == "__main__":
    src_dir = (
        "/Users/ofermagen/Coding/NLP_Project_Data/home/ofermagen/data/objects/train/"
    )
    json_file = "/Users/ofermagen/Coding/NLP_Project_Data/nlvr/nlvr2/data/train.json"
    gen = DataGenerator(json_file, src_dir)
    print(gen[0][0][4])
