import pickle
import cv2
from PIL import Image
import numpy as np
import os
import json

img_dir = '/home/ofermagen/data/unformatted_images/train/'
json_file = 'data/pretraining_data_formatted/train/ID2path.json'
boxes_dir = '/home/ofermagen/data/pretraining_boxes/train/'
res_path = '/home/ofermagen/data/objects/train/'

print(boxes_dir)
assert os.path.isdir(img_dir)
assert os.path.isdir(boxes_dir)
assert os.path.isfile(json_file)
assert not os.path.isdir(res_path)
os.mkdir(res_path)

def crop_object(img,box):
    x0, y0, x1, y1 = box
    x0 = int(x0 * img.shape[0])
    x1 = int(x1 * img.shape[0])
    y0 = int(y0 * img.shape[1])
    y1 = int(y1 * img.shape[1])
    return img[x0:x1,y0:y1]


def resize(im, desired_size,color = (128, 128, 128)):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def id2_objects(ID):
    global id2_image_path,id2_boxes_path,class_dict
    assert ID in id2_image_path and ID in id2_boxes_path
    size = 128
    orig_img = Image.open(id2_path[ID])
    orig_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = resize(orig_img,1024)

    with open(id2_boxes_path[ID],'rb') as f: OBJ = pickle.load(f)

    images = [resize(orig_img,size)]
    for box in OBJ['detection_boxes']:
        cropped_image = crop_object(img,box)
        cropped_image = resize(cropped_image,size)
        images.append(cropped_image)
    
    images = np.stack(images)
    boxes = np.concat([np.asarray([0,1,0,1],dtype=float),  OBJ['detection_boxes']] ,axis=0)
    classes = [0] + [class_dict[c] for c in OBJ["detection_classes"]]
    classes = np.asarray(classes)
    scores = [1.0] + [s for s in OBJ["detection_scores"]]
    scores = np.asarray(scores)

    desc = lambda A: 'shape='+str(A.shape)+", dtype="+str(A.dtype)
    print('images',desc(images))
    print('scores',desc(scores))
    print('boxes',desc(boxes))
    print('classes',desc(classes))
    
    return images,boxes,classes,scores

def format_OBJ(images,boxes,classes,scores):
    raise NotImplementedError
    return 0

id2_image_path = dict()
for root, _, files in os.walk(img_dir):
    for f in files:
        if os.path.splitext(f)[1] == ".png":
            id2_image_path[f] = os.path.join(root, f)

id2_boxes_path = dict()
with open(json_file,'r') as f:
    tmp = json.load(f)
    for ID,path in tmp.items():
        path = path.splitext('/')[-2:]
        path = '/'.join(path)
        path = os.path.splitext(path)[0] + ".pickle"
        path = os.path.join(boxes_dir,path)
        id2_boxes_path[ID] = path

class_dict = dict()
special_classes = 1
for _,path in id2_boxes_path.items():
    with open(path,'rb') as f: 
        OBJ = pickle.load(f)
        for c in OBJ['detection_classes']:
            if c not in class_dict:
                class_dict[c] = len(class_dict) + special_classes

print('formatting')
res_dict = dict()
for ID in id2_image_path:
    if ID not in id2_boxes_path: continue
    OBJ = id2_objects(ID)
    OBJ = format_OBJ(*OBJ)

    path = str(len(res_dict)) + ".pickle"
    path = os.path.join(res_path,path)
    res_dict[ID] = path
    with open(path,'wb') as f:
        pickle.dump(OBJ,f,pickle.HIGHEST_PROTOCOL)

path = os.path.join(res_path,'ID2path.json')
with open(path,'w') as f:
    json.dump(res_dict,f)