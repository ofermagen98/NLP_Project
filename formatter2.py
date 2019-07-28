import cv2
from PIL import Image
from time import sleep
import json
import pickle
import os
import numpy as np
from progressbar import progressbar
from random import shuffle

def resize(im, desired_size = 64):
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

def get_imgs(img_id):
    global ID2boxes_path,ID2images_path
    img = ID2images_path[img_id]
    img = np.array(Image.open(img))
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    OBJ = ID2boxes_path[img_id]
    with open(OBJ,'rb') as f: OBJ = pickle.load(f)

    res_imgs = []
    for x0,y0,x1,y1 in OBJ['detection_boxes']:
        x0 = int(x0*img.shape[0])
        x1 = int(x1*img.shape[0])
        y0 = int(y0*img.shape[1])
        y1 = int(y1*img.shape[1])

        sub_img = img[x0:x1,y0:y1]
        sub_img = resize(sub_img,64)
        res_imgs.append(sub_img)

    fix = lambda S: np.expand_dims(OBJ[S],axis=1)
    features = [fix('detection_classes'),fix('detection_scores'),OBJ ['detection_boxes']]
    for A in features: print(A.shape)
    features = np.concatenate(features,axis=1)
    print(features.shape)
    return res_imgs,features

def sent2list(sent,size = 40):
    replace = lambda c: c if c.isdigit() or c.isalpha() else ' '
    sent = ''.join(map(replace,sent))
    sent = sent.split(' ')
    sent = filter(lambda w: len(w) > 0, sent)
    sent = list(sent)
    sent = sent[:size]
    for _ in range(len(sent),size): sent.append(pad_word)
    return list(sent)

def get_example(ex):
    global ID2images_path
    ID = ex['identifier'].split("-")[:3]
    ID = "-".join(ID)

    if ID + "-img0.png" not in ID2images_path or not os.path.isfile(ID2boxes_path[ID+"-img0.png"]):
        #print(ID + "-img0.png") 
        return None
    if ID + "-img1.png" not in ID2images_path or not os.path.isfile(ID2boxes_path[ID+"-img1.png"]):
        #print(ID + "-img1.png") 
        return None

    res = dict()
    res['dataL'] = get_imgs(ID + "-img0.png")
    res['dataR'] = get_imgs(ID + "-img1.png")
    res['sent'] = list(map(lambda w: word2num[w],sent2list(ex["sentence"])))
    res['label'] = ex['label'][0] == 'T'
    res['ID'] = ex['identifier']
    return res


#######################
examples = '/Users/ofermagen/Coding/NLP_Project_Data/data/nlvr/nlvr2/data/train.json'
boxes_orig_dir = '/Users/ofermagen/Coding/NLP_Project_Data/data/boxes/train/'
ID2images_path = '/Users/ofermagen/Coding/NLP_Project_Data/data/pretraining_data_formatted/train/ID2path.json'
ID2boxes_path = '/Users/ofermagen/Coding/NLP_Project/get_boxes/google_train_ID2path.json'
DDIR =  '/Users/ofermagen/Coding/NLP_Project_Data/data/fully_formatted_data/train/'

with open(examples,'r') as f: examples = [json.loads(s) for s in f.readlines()]
with open(ID2images_path,'r') as f: ID2images_path = json.load(f)
with open(ID2boxes_path,'r') as f: ID2boxes_path = json.load(f)

for ID,path in ID2boxes_path.items():
    path = "/".join(path.split("/")[-2:])
    path = boxes_orig_dir + os.path.splitext(path)[0] + ".pickle"
    ID2boxes_path[ID] = path 
os.mkdir(DDIR)

##########################################################

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
for ex in progressbar(examples):
    data = get_example(ex)
    if data is not None:
        res_path = os.path.join(DDIR,data['ID']) + ".pickle"
        with open(res_path,'wb') as f:
            pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)