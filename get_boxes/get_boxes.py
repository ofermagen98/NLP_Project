import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
#import cv2

from progressbar import progressbar
import json
import pickle
from time import time

INDEX = 0
if len(sys.argv) > 1: INDEX = sys.argv[0]
CONFIG_PATH = '/home/ofermagen/NLP_Project/get_boxes/config.json'
MODELS_DIR =  '/home/ofermagen/models/'
with open(CONFIG_PATH,'r') as f:
  CONFIG = json.load(f)
  OBJ = CONFIG['OBJS'][INDEX]
  CONFIG.pop('OBJS',None)

assert os.path.isdir(CONFIG['SDIR'])
assert os.path.isdir(CONFIG['DDIR'])

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
MODEL_PATH = os.path.join(MODELS_DIR,MODEL_NAME)
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_PATH,'frozen_inference_graph.pb')
assert os.path.isfile(PATH_TO_FROZEN_GRAPH)

#create graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#main function
def run_inference(images, graph,device):
  with tf.device(device):
    with graph.as_default():
      # Get output tensors
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
      
      #Get input tensor
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      
      with tf.Session() as sess:
        # Run inference
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = output_dict['num_detections'].astype(np.int32)
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int32)
  output_dict['detection_boxes'] = output_dict['detection_boxes']
  output_dict['detection_scores'] = output_dict['detection_scores']

  res = []
  for i,n in enumerate(output_dict['num_detections']):
    obj = dict()
    obj['num_detections'] = n
    obj['detection_classes'] = output_dict['detection_classes'][i,:n]
    obj['detection_scores'] = output_dict['detection_scores'][i,:n]
    obj['detection_boxes'] = output_dict['detection_boxes'][i,:n,:]
    res.append(obj)
  
  return res

def plot(img,boxes):
  for x0,y0,x1,y1 in boxes:
    x0 = int(x0*img.shape[0])
    x1 = int(x1*img.shape[0])
    y0 = int(y0*img.shape[1])
    y1 = int(y1*img.shape[1])
    cv2.rectangle(img,(y0,x0),(y1,x1),color=(255,0,0),thickness=2)

  return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

non_existing = lambda i: not os.path.isfile(os.path.join(CONFIG['DDIR'],OBJ['out_paths'][i]))
non_existing = list(filter(non_existing,range(len(OBJ['in_paths']))))
OBJ['in_paths'] = [OBJ['in_paths'][i] for i in non_existing]
OBJ['out_paths'] = [OBJ['out_paths'][i] for i in non_existing]

img_num = len(non_existing)
print(img_num)
batch_size = 48
in_paths =  [OBJ['in_paths'][i:i+batch_size]  for i in range(0,img_num,batch_size)]
out_paths = [OBJ['out_paths'][i:i+batch_size] for i in range(0,img_num,batch_size)]

for in_batch, out_batch in zip(in_paths,out_paths):
  start = time()
  print(in_batch)
  imgs = [Image.open(os.path.join(CONFIG['SDIR'],path)) for path in in_batch]
  imgs = np.stack(imgs)
  
  print('estimating')
  output_dict = run_inference(imgs, detection_graph,OBJ['device'])
  
  for res,out_path in zip(output_dict,out_batch):
    with open(os.path.join(CONFIG['DDIR'],out_path),'wb') as f:
      pickle.dump(res,f,pickle.HIGHEST_PROTOCOL)

  print('took', time() - start)