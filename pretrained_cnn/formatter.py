import json 
import os
import cv2
import numpy as np
from progressbar import progressbar
from PIL import Image
import imagehash

def resize(im, desired_size = 224):
	old_size = im.shape[:2] # old_size is in (height, width) format
	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	# new_size should be in (width, height) format
	im = cv2.resize(im, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [128, 128, 128]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	return new_im

def read_img(path):
    try:
        #check hashes
        im = Image.open(path)
        name = os.path.basename(path)
        real = hashes[name]
        pred = str(imagehash.average_hash(im))
        assert real == pred, "bad hash"

        #convert cv2 matrix
        if im.mode != "RGB": im = im.convert("RGB")
        im = np.asarray(im)
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        return resize(im)

    except Exception as ex:
        print(ex)
        return None

#####################

orig_dir = '/home/ofermagen/data/'
assert os.path.isdir(orig_dir)
json_file = orig_dir + 'nlvr/nlvr2/data/dev.json'
imgs_dir = orig_dir + 'unformatted_images/dev'
hash_file = orig_dir + 'nlvr/nlvr2/util/hashes/dev_hashes.json'

DDIR = '/home/ofermagen/data/pretraining_data/dev'
if not os.path.isdir(DDIR): os.mkdir(DDIR)
#####################

examples = [json.loads(s) for s in open(json_file).readlines()]
hashes = json.loads(open(hash_file).read())

#####################

synset2num = set(ex['synset'] for ex in examples)
synset2num = {synset:i for i,synset in enumerate(synset2num)}

with open(os.path.join(DDIR,'synset2num.json'),'w') as f:
    json.dump(synset2num,f)

id2synet = dict()
for ex in examples:
    ID = ex['identifier']
    ID = "-".join(ID.split("-")[:3])
    for _ID in [ID + "-img0.png", ID + "-img1.png"]:
        if _ID not in id2synet:
            id2synet[_ID] = set()
        id2synet[_ID].add(synset2num[ex['synset']])
lens = set(len(k) for _,k in id2synet.items())
assert lens == {1}

for ID in id2synet:
    id2synet[ID] = list(id2synet[ID])[0]

#####################

id2path = dict()
for root, _, files in os.walk(imgs_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                id2path[file] = os.path.join(root,file)

examples = [json.loads(line) for line in open(json_file).readlines()]
hashes = json.loads(open(hash_file).read())
paths = os.listdir(imgs_dir)

for ID in progressbar(id2path):
    img = read_img(id2path[ID])
    if img is None: continue

    C = id2synet[ID]
    res_dir = os.path.join(DDIR,str(C))
    if not os.path.isdir(res_dir): os.mkdir(res_dir)

    path = os.listdir(res_dir)
    path = filter(lambda p: os.path.splitext(p)[1] == '.png', path)
    path = sum(1 for _ in path)
    path = os.path.join(res_dir,str(path) + ".png")

    cv2.imwrite(path,img)
    