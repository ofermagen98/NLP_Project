import os
import pickle
import json
from tqdm import tqdm as progressbar

SDIR = '/home/joberant/home/ofermagen/pretrained_cnn_objects/train/'
objs = os.listdir(SDIR)
objs = filter(lambda p: os.path.splitext(p)[1] == '.pickle', objs)
objs = map(lambda p: os.path.join(SDIR,p), objs)

objs = list(objs)
id2path = dict()

for path in progressbar(objs):
    with open(path,'rb') as f:
        ID = pickle.load(f)['ID']
    id2path[ID] = path

with open(os.path.join(SDIR,'ID2Path.json'),'w') as f:
    json.dump(id2path,f)