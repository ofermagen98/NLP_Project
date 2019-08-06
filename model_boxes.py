import json
import os
import sys
import numpy as np
from PIL import Image

# data sources
# data_dir = '/Users/ofermagen/Coding/NLP_Project_Data/formatted_images'
assert len(sys.argv) > 2

if len(sys.argv) > 1:
    if sys.argv[1] == "J":
        orig_dir = "/specific/netapp5/joberant/home/ofermagen/"
        train_data = (
            orig_dir + "nlvr/nlvr2/data/train.json",
            orig_dir + "objects/train/",
        )
        dev_data = (orig_dir + "nlvr/nlvr2/data/dev.json", orig_dir + "objects/dev/")
        model_path = "/specific/disk1/home/gamir/ofer/checkpoint_best/model.h5"
    elif sys.argv[1] == "O":
        train_data_dir = "/home/ofermagen/data/semiformatted_images/train/"
        model_path = "/home/ofermagen/checkpoint_best/model.h5"
    else:
        raise NotImplementedError

    for p in [train_data, dev_data]:
        assert os.path.isdir(p[1])
        assert os.path.isfile(p[0])

else:
    # raise NotImplementedError
    pass

from utils import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from utils import HistorySaver

from tensorflow.keras.initializers import Constant

from generator_boxes import DataGenerator
from RN import (
    ReduceMean,
    MaskedReduceMean,
    RelationalProduct,
    ConvolutionalPerceptron,
    Perceptron,
)
from resnet import Simple_CNN, ResnetV1_FCNN
from LSTM import Encoder


NUM_EPOCHS = 200

def lr_schedualer(epoch, *a, **kw):
    global NUM_EPOCHS
    base = 1e-6
    x = float(epoch) / NUM_EPOCHS
    frac = pow(2, -4 * x)
    return base * frac


# defining model's inputs
size = 30
features_dim = 2048 + 4 + 1 #image embedding + box + score

features = Input(shape=(size, features_dim), name="img", dtype="float32")
sides = Input(shape=(size,), name="img_sides", dtype="int32")
sent = Input(shape=(40,), name="sent", dtype="int32")

feature_mask = tf.math.not_equal(sides, 0)
sent_mask = tf.math.not_equal(sent, 0)


class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def call(self, X):
        global size, features_dim
        features, sides = X

        # embedd features
        em_features = tf.concat([features, tf.cast(sides, dtype="float32")], axis=-1)
        n_dim = features_dim + 1
        em_features = tf.reshape(em_features, shape=(-1, n_dim))
        prec_params = [(1024, "sigmoid"), (1024, "sigmoid"), (1024, "sigmoid")]
        prec = Perceptron(n_dim, prec_params)
        em_features = prec(em_features)
        n_dim = prec_params[-1][0]
        em_features = tf.reshape(em_features, shape=(-1, size, n_dim))

        # returning embedded features
        return em_features


em_featurs = FeatureExtractor()([features, sides])

# embedding sentence
print("creating transformer encoder")
GloVe_embeddings = np.load("word_embeddings/embedding.npy")
print(GloVe_embeddings.shape)
encoder = Encoder(
    units=2048,
    out_dim=1024,
    input_vocab_size=GloVe_embeddings.shape[0],
    word_dim=300,  # also the word embedding dim
    embeddings_initializer=Constant(GloVe_embeddings),
)
em_sent = encoder(sent)

# getting prediction from the Relational Neural Network
print("creating relational network")
relation_matrix = RelationalProduct()([em_sent, em_featurs])
print(relation_matrix.shape)

g = ConvolutionalPerceptron(relation_matrix.shape[1:], [1024, 1024, 1024, 1024])
em_relations = g(relation_matrix)
relation_out = MaskedReduceMean()(em_relations, O1_mask=sent_mask, O2_mask=img_mask)

prec_params = [
    (1024, "relu"),
    (512, "relu"),
    (256, "relu"),
    (128, "relu"),
    (64, "relu"),
    (32, "relu"),
    (16, "relu"),
    (1, "sigmoid")
]
f = Perceptron(relation_out.shape[1], prec_params)
pred = f(relation_out)

# compile model
print("compiling model")
model = Model(inputs=[features, sides, sent], outputs=pred)
model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

# model.load_weights(model_path)
checkpoint = ModelCheckpoint(
    filepath=model_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
)
lrate = LearningRateScheduler(lr_schedualer)
saver = HistorySaver("/specific/netapp5/joberant/home/ofermagen/train_loss.json")
callbacks = [checkpoint, lrate, saver]
# model.load_weights(model_path)

print("creating generators")
# sampled_images = smaple_images(train_data_dir, 1000)
# sampled_images = np.stack([np.array(Image.open(path)) for path in sampled_images])
train_gen = DataGenerator(*train_data, batch_size=16)
val_gen = DataGenerator(*dev_data, batch_size=16)

print("training model")
model.fit_generator(
    train_gen,
    epochs=200,
    verbose=1,
    workers=4,
    callbacks=callbacks,
    validation_data=val_gen,
    shuffle=False,
)

