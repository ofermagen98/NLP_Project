import psutil
import signal
import json
import os
import random
import sys


DROPOUT_BOOL = True
DROPOUT_RATE = 0.5
LAYER_NORM_EPSION = 1e-6
L2_REG = 5e-7


def smaple_images(dir, num, seed=1234):
    samples = []
    for root, _, files in os.walk(dir):
        for f in files:
            if os.path.splitext(f)[1] == ".png":
                samples.append(os.path.join(root, f))
    # samples.sort(key = os.path.basename)
    samples.sort()
    random.Random(seed).shuffle(samples)
    samples = samples[:num]
    return samples


def import_tensorflow():
    def kill_children(*a, **kw):
        print("timeout")
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        if len(children) > 0:
            os.kill(children[0].pid, signal.SIGTERM)

    # export LD_LIBRARY_PATH=/usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-10.0-v7/lib64/
    os.environ[
        "LD_LIBRARY_PATH"
    ] = "/usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-10.0-v7/lib64/"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

    signal.signal(signal.SIGALRM, kill_children)
    signal.alarm(20)
    import tensorflow
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    return tensorflow


tensorflow = import_tensorflow()


class HistorySaver(tensorflow.keras.callbacks.Callback):
    def __init__(self, fname, save_every=32, *a, **kw):
        super(HistorySaver, self).__init__(*a, **kw)
        self.fname = fname
        self.save_every = save_every

    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(float(logs.get("loss")))
        self.train_acc.append(float(logs.get("acc")))
        self.val_loss.append(float(logs.get("val_loss")))
        self.val_acc.append(float(logs.get("val_acc")))

        with open(self.fname, "w") as f:
            json.dump(
                {
                    "train_loss": self.train_loss,
                    "train_acc": self.train_acc,
                    "val_loss": self.val_loss,
                    "val_acc": self.val_acc,
                },
                f,
            )


class Pad(tensorflow.keras.layers.Layer):
    """
	"""

    def __init__(self, value=0, axis=-1):
        super(Pad, self).__init__()
        self.value = value
        self.axis = axis

    def call(self, tensor):
        n_shape = tensor.shape.as_list()
        n_shape[self.axis] += 1
        n_shape = [x if x is not None else -1 for x in n_shape]
        padding = [[0, 0] for _ in n_shape]
        padding[self.axis] = [0, 1]
        res = tensorflow.pad(
            tensor, padding, mode="CONSTANT", constant_values=self.value
        )
        return tensorflow.reshape(res, n_shape)

