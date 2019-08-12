import ssl
import os
import shutil

from Imagenet_class2name import D
ssl._create_default_https_context = ssl._create_unverified_context

SDIR = '/specific/disk1/home/gamir/ofer/data/object_boxes/test1/'
IMG_DIR = '/specific/disk1/home/gamir/ofer/data/unformatted_images/test1/'
HASH_FILE = "/specific/disk1/home/gamir/ofer/data/nlvr/nlvr2/util/hashes/test1_hashes.json"
RDIR = '/specific/netapp5/joberant/home/ofermagen/pretrained_cnn_objects/test1'
weight_path = "/home/joberant/home/ofermagen/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

SIZE = 299

assert os.path.isdir(SDIR)
assert os.path.isdir(IMG_DIR)
assert os.path.isfile(HASH_FILE)
assert os.path.isdir(os.path.dirname(RDIR))
if os.path.isdir(RDIR): shutil.rmtree(RDIR)
os.mkdir(RDIR)

import pickle
import json
import cv2
from tqdm import tqdm as progressbar
import numpy as np
from PIL import Image
import imagehash

from utils import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications.inception_v3 import preprocess_input

def resize(im, desired_size=SIZE):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [128, 128, 128]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def read_img(path):
    try:
        # check hashes
        im = Image.open(path)
        name = os.path.basename(path)
        real = hashes[name]
        pred = str(imagehash.average_hash(im))
        assert real == pred, "bad hash"

        # convert cv2 matrix
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = np.asarray(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = resize(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    except Exception as ex:
        print(ex)
        return None

base_model = tf.keras.applications.InceptionV3(input_shape=(SIZE,SIZE,3),
                                               weights=None,
                                               include_top=False)
base_model.load_weights(weight_path)
print("loaded weights")

id2path = dict()
for root, _, files in os.walk(IMG_DIR):
    for f in files:
        if os.path.splitext(f)[1] == ".png":
            id2path[f] = os.path.join(root, f)
hashes = json.loads(open(HASH_FILE).read())

objs = os.listdir(SDIR)
objs = filter(lambda p: os.path.splitext(p)[1] == '.pickle', objs)
objs = map(lambda p: os.path.join(SDIR,p), objs)
objs = progressbar(list(objs))

print("formatting")
for path in objs:
    with open(path,'rb') as f:
        OBJ = pickle.load(f)

    ID = os.path.basename(path)
    ID = os.path.splitext(ID)[0] + ".png"
    img = id2path[ID]
    img = read_img(img)

    sub_images = []
    boxes = []
    scores = []

    
    for i,(xm,ym,xM,yM) in enumerate(OBJ['boxes']):
        if OBJ['classes'][i] == 0:
            xm = 0
            ym = 0
            xM = SIZE
            yM = SIZE 
        else:
            xm = int(xm*SIZE)
            xM = int(xM*SIZE)
            ym = int(ym*SIZE)
            yM = int(yM*SIZE)

        sub_img = img[xm:xM,ym:yM,:]
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)
        sub_img = resize(sub_img)
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)

        sub_images.append(preprocess_input(sub_img))
        boxes.append(np.asarray([xm/SIZE,ym/SIZE,xM/SIZE,yM/SIZE]))
        scores.append(OBJ['scores'][i])
    
    features = base_model.predict(np.stack(sub_images))
    features = np.mean(features,axis=1)
    features = np.mean(features,axis=1)
    scores = np.expand_dims(np.stack(scores),axis=-1)

    features = np.concatenate([features,np.stack(boxes),scores],axis=-1)

    new_OBJ = dict()
    new_OBJ['ID'] = ID
    new_OBJ['features'] = features
    new_path = os.path.join(RDIR,os.path.basename(path))

    with open(new_path,'wb') as f:
        pickle.dump(new_OBJ,f,pickle.HIGHEST_PROTOCOL)
    
