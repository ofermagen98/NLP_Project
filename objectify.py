import os
from tqdm import tqdm as progressbar
import pickle
import numpy as np

DIR = "/specific/disk1/home/gamir/ofer/data/object_boxes/test1/"

paths = os.listdir(DIR)
paths = filter(lambda p: ".pickle" in p, paths)
paths = map(lambda p: os.path.join(DIR, p), paths)
paths = list(paths)


def objectify(OBJ):
    res = dict()
    print(OBJ["detection_boxes"].shape)
    A = np.asarray([[0.0, 0.0, 1.0, 1.0]])
    print(A.shape)

    res["scores"] = np.concatenate(
        [np.asarray([1.0]), OBJ["detection_scores"]], axis=0
    )
    res["classes"] = np.concatenate(
        [np.asarray([0]), OBJ["detection_classes"] + 1], axis=0
    )
    res["boxes"] = np.concatenate(
        [A, OBJ["detection_boxes"]], axis=0
    )
    exit()
    return res

for path in progressbar(paths):
    with open(path, "rb") as f:
        OBJ = pickle.load(f)

    OBJ = objectify(OBJ)

    with open(path, "wb") as f:
        OBJ = pickle.load(f)

