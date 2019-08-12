import sys
sys.path.append("..")

import numpy as np
import os
from utils import tensorflow as tf
from PIL import Image

import cv2

from tqdm import tqdm as progressbar
import json
import pickle
from time import time

IMG_DIR = '/specific/disk1/home/gamir/ofer/data/unformatted_images/test1/'
RES_DIR = '/specific/disk1/home/gamir/ofer/data/object_boxes/test1/'
GPU_DEVICE = '/device:XLA_GPU:0'
MODELS_DIR = '/specific/disk1/home/gamir/ofer/models'

def path2output(p):
    name = os.path.basename(p)
    name = os.path.splitext(name)[0]
    return os.path.join(RES_DIR,p+'.pickle')

def get_batches(paths, desired_size = 256, batch_size=35):
    color = [128, 128, 128]

    images = []
    names = []
    for path in progressbar(paths):
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = np.asarray(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        old_size = im.shape[:2]  # old_size is in (height, width) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        images.append(im)
        names.append(path2output(path))

        if len(images) > batch_size:
            yield names, images
            images = []
            names = []
    
    if len(images) > 0:
        yield names, images
        images = []
        names = []

# This is needed since the notebook is stored in the object_detection folder.
MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
assert os.path.isfile(PATH_TO_FROZEN_GRAPH)

# create graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

# main function
def run_inference(images, graph, device):
    with tf.device(device):
        with graph.as_default():
            # Get output tensors
            tensor_dict = {}
            for key in [
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
            ]:
                tensor_name = key + ":0"
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name
                )

            # Get input tensor
            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            with tf.Session() as sess:
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict["num_detections"] = output_dict["num_detections"].astype(np.int32)
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int32)
    output_dict["detection_boxes"] = output_dict["detection_boxes"]
    output_dict["detection_scores"] = output_dict["detection_scores"]

    res = []
    for i, n in enumerate(output_dict["num_detections"]):
        obj = dict()
        obj["num_detections"] = n
        obj["detection_classes"] = output_dict["detection_classes"][i, :n]
        obj["detection_scores"] = output_dict["detection_scores"][i, :n]
        obj["detection_boxes"] = output_dict["detection_boxes"][i, :n, :]
        res.append(obj)

    return res

non_existing = lambda p: not os.path.isfile(path2output(p))
paths = os.listdir(IMG_DIR)
paths = map(lambda p: os.path.join(IMG_DIR,p), paths)
paths = filter(non_existing,paths)
paths = list(paths)

img_num = len(non_existing)
count = 0
# 36,40
for out_paths, images in get_batches(paths):
    print("\nremaining", img_num - count)
    start = time()
    print("estimating")
    output_dict = run_inference(images, detection_graph, GPU_DEVICE)
    print("took", time() - start)

    out_paths = map(lambda p: os.path.join(RES_DIR,p))
    for res, out_path in zip(output_dict, out_paths):
        with open(out_path, "wb") as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

    count += len(images)

