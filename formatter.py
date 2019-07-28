import cv2
from PIL import Image
from time import sleep
import json
import os
import imagehash
import numpy as np
from progressbar import progressbar
from random import shuffle

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

def sent2list(sent,size = 40):
    replace = lambda c: c if c.isdigit() or c.isalpha() else ' '
    sent = ''.join(map(replace,sent))
    sent = sent.split(' ')
    sent = filter(lambda w: len(w) > 0, sent)
    sent = list(sent)
    sent = sent[:size]
    for _ in range(len(sent),size): sent.append(pad_word)
    return list(sent)

#######################

orig_dir = '/Users/ofermagen/Coding/NLP_Project_Data/data/'
res_dir = '/Users/ofermagen/Coding/NLP_Project_Data/data/'
img_dir = orig_dir + 'unformatted_images/train'
json_file = orig_dir + 'nlvr/nlvr2/data/train.json'
hash_file = orig_dir + 'nlvr/nlvr2/util/hashes/train_hashes.json'
DDIR =  res_dir + 'semiformatted_images/train'
if not os.path.isdir(DDIR): os.mkdir(DDIR)

#######################

id2path = dict()
for root, _, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                id2path[file] = os.path.join(root,file)

examples = [json.loads(line) for line in open(json_file).readlines()]
hashes = json.loads(open(hash_file).read())

#######################

word2num = dict()
pad_word = '_UNIQUE_PADDING_WORD'
word2num[pad_word] = 0
for sent in examples:
  sent = sent['sentence']
  sent = sent2list(sent)
  for w in sent:
    if w not in word2num:
      tmp = len(word2num)
      word2num[w] = tmp
with open(os.path.join(DDIR,'word2num.json'),'w') as f:
  json.dump(word2num,f)

params = dict()
params['vocab_size'] = len(word2num)

with open(os.path.join(DDIR,"params.json"),"w") as f:
    json.dump(params,f)

#######################

shuffle(examples)
batch_size = 1024
dir_count = -1
res_dir = None  
data = []

for ex in progressbar(examples):
    if res_dir is None or len(data) >= batch_size:
        if res_dir is not None:
            with open(os.path.join(res_dir,"data.json"),"w") as f:
                json.dump(data,f)

        dir_count += 1
        res_dir = os.path.join(DDIR,str(dir_count))
        data = []

        if not os.path.isdir(res_dir): os.mkdir(res_dir)

    ID = ex['identifier'].split("-")[:3]
    ID = "-".join(ID)
    if ID + "-img0.png" not in id2path:
        print(ID + "-img0.png") 
        continue
    if ID + "-img1.png" not in id2path:
        print(ID + "-img1.png") 
        continue
    imgL = read_img(id2path[ID + "-img0.png"])
    imgR = read_img(id2path[ID + "-img1.png"])
    if imgL is None or imgR is None: continue

    cv2.imwrite(os.path.join(res_dir,str(len(data)) + "-img0.png"),imgL)
    cv2.imwrite(os.path.join(res_dir,str(len(data)) + "-img1.png"),imgR)
    sent = map(lambda w: word2num[w],sent2list(ex["sentence"]))
    label = ex['label'][0] == 'T'
    data.append({"id": ID, "sentence" : list(sent), "label" : label})


if len(data) > 0:
    with open(os.path.join(res_dir,"data.json"),"w") as f:
        json.dump(data,f)