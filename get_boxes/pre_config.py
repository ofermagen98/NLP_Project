import os
DEVICE_NUM = 8
SOURCE = '/specific/disk1/home/gamir/ofer/'
SDIR = SOURCE + 'data/pretraining_data_formatted/dev/'
DDIR = SOURCE + 'data/pretraining_boxes/dev/'
json_file = SOURCE + 'NLP_Project/get_boxes/config.json'

assert os.path.isdir(SDIR)
assert os.path.isdir(DDIR)

import json
from random import shuffle
import numpy as np
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUS  = get_available_gpus()

if len(GPUS) == 0:
    print('no available GPUs')
    DEVICE_NUM = 1
    DEVICES = ['/device:cpu:0']
else:
    print(GPUS)
    print('found GPUs')
    shuffle(GPUS)
    DEVICE_NUM = min(DEVICE_NUM,len(GPUS))
    DEVICES = GPUS[:DEVICE_NUM]

res = dict()
res['DEVICE_NUM'] = DEVICE_NUM
res['SDIR'] = SDIR
res['DDIR'] = DDIR

paths = []
dirs = os.listdir(SDIR)
dirs = filter(lambda s: os.path.isdir(os.path.join(SDIR,s)) ,dirs)
dirs = list(dirs)

for d in dirs:
    try: os.mkdir(os.path.join(DDIR,d))
    except: pass
    files = os.listdir(os.path.join(SDIR,d))
    images = filter(lambda s: os.path.splitext(s)[1] == '.png',files)
    images = map(lambda s: os.path.join(d,s), images)
    paths += list(images)
shuffle(paths)

batch_size = (len(paths) + DEVICE_NUM - 1) // DEVICE_NUM
batches = [paths[i:i+batch_size] for i in range(0,len(paths),batch_size)]
res['OBJS'] = []

for name,paths in zip(DEVICES,batches):
    res_paths = map(lambda s: os.path.splitext(s)[0] + ".pickle", paths)
    res['OBJS'].append({"device":name,"in_paths":paths,"out_paths":list(res_paths)})

with open(json_file,'w') as f:
    json.dump(res,f)

