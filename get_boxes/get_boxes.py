
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image

from progressbar import progressbar
import pickle
from time import time

import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
MODELS_DIR = '/Users/ofermagen/Coding/NLP_Project_Data/models/'
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
def run_inference(images, graph):
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
    output_dict['detection_classes'] = list(output_dict['detection_classes'].astype(np.int32))
    output_dict['detection_boxes'] = list(output_dict['detection_boxes'])
    output_dict['detection_scores'] = list(output_dict['detection_scores'])

    for i,n in enumerate(output_dict['num_detections']):
      output_dict['detection_classes'][i] = output_dict['detection_classes'][i][:n]
      output_dict['detection_scores'][i] = output_dict['detection_scores'][i][:n]
      output_dict['detection_boxes'][i] = output_dict['detection_boxes'][i][:n,:]
    
  return output_dict

def plot(img,boxes):
  for x0,y0,x1,y1 in boxes:
    x0 = int(x0*img.shape[0])
    x1 = int(x1*img.shape[0])
    y0 = int(y0*img.shape[1])
    y1 = int(y1*img.shape[1])

    cv2.rectangle(img,(y0,x0),(y1,x1),color=(255,0,0),thickness=2)

  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
  cv2.imshow('img',img)
  cv2.waitKey()

imgs = ['/Users/ofermagen/Coding/NLP_Project_Data/data/pretraining_data_formatted/dev/0/' + str(i) + '.png' for i in range(10)]
imgs = np.stack(np.array(Image.open(img)) for img in imgs)

print("running prediction")
begin = time()
output_dict = run_inference(imgs, detection_graph)
end = time()

print(end-begin)
for i in range(imgs.shape[0]):
  plot(imgs[i,:,:,:], output_dict['detection_boxes'][i])
