import cv2
from PIL import Image
import pickle
import numpy as np
import json
from random import sample
import os

def plot(img,boxes):
  for x0,y0,x1,y1 in boxes:
    x0 = int(x0*img.shape[0])
    x1 = int(x1*img.shape[0])
    y0 = int(y0*img.shape[1])
    y1 = int(y1*img.shape[1])
    cv2.rectangle(img,(y0,x0),(y1,x1),color=(255,0,0),thickness=2)

  return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

pickle_file = '/Users/ofermagen/Coding/NLP_Project/get_boxes/0.pickle'
local_dict = '/Users/ofermagen/Coding/NLP_Project_Data/data/pretraining_data_formatted/train/ID2path.json'
with open(local_dict,'r') as f: local_dict = json.load(f)
global_dict = '/Users/ofermagen/Coding/NLP_Project/get_boxes/google_train_ID2path.json'
with open(global_dict,'r') as f: global_dict = json.load(f)

key = None
while key != 27:
    found = False
    while not found:
        ID = sample(local_dict.keys(),1)[0]
        
        getfile_cmd = global_dict[ID]
        getfile_cmd = getfile_cmd.split('/')
        getfile_cmd[-4] = 'pretraining_boxes'
        getfile_cmd = '/'.join(getfile_cmd)
        getfile_cmd = os.path.splitext(getfile_cmd)[0] + ".pickle"
        getfile_cmd = 'gcloud compute --project "craft-216310" scp --zone "us-west1-b" "nlp-project-vm":' + getfile_cmd
        getfile_cmd = getfile_cmd + ' ' + pickle_file
        found = not bool(os.system(getfile_cmd))

    img = np.array(Image.open(local_dict[ID]))

    with open(pickle_file,'rb') as f: OBJ = pickle.load(f)
    print(OBJ['num_detections'])
    img = plot(img,OBJ['detection_boxes'])
    cv2.imshow('img',img)
    key = cv2.waitKey() & 0xff 

