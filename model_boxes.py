import json
import os
import sys
import numpy as np
from utils import smaple_images
from PIL import Image
from tensorflow.keras.utils import plot_model

# data sources
# data_dir = '/Users/ofermagen/Coding/NLP_Project_Data/formatted_images'

if len(sys.argv) > 1:
    if sys.argv[1] == "J":
        orig_dir = "/specific/netapp5/joberant/home/ofermagen/"
        train_data = (
            orig_dir + "nlvr/nlvr2/data/train.json",
            orig_dir + "objects/train",
        )
        dev_data = (orig_dir + "nlvr/nlvr2/data/dev.json", orig_dir + "objects/dev")
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

from tensorflow.keras.initializers import Constant

from generator_boxes import DataGenerator
from RN import MaskedReduceMean, RelationalProduct, ConvolutionalPerceptron, Perceptron
from resnet import Simple_CNN
from transformer import Encoder, create_padding_mask


def lr_schedualer(epoch, num_epoches=200):
    base = 1e-6
    x = float(epoch) / num_epoches
    frac = pow(2, -4 * x)
    return base * frac


# defining model's inputs
size = 30
num_class = 540
img_shape = (size, 128, 128, 3)

imgs = Input(shape=img_shape, name="img", dtype="float32")
boxes = Input(shape=(size, 4), name="boxes", dtype="float32")
scores = Input(shape=(size,), name="scores", dtype="float32")
classes = Input(shape=(size,), name="classes", dtype="int32")
sides = Input(shape=(size,), name="img_sides", dtype="int32")
sent = Input(shape=(40,), name="sent", dtype="int32")


img_mask = tf.math.not_equal(sides, 0)
sent_mask = tf.math.not_equal(sent, 0)

# cast to float32
em_sides = tf.cast(sides, dtype="float32")
em_sides = tf.expand_dims(em_sides, axis=2)
em_scores = tf.expand_dims(scores, axis=2)

# embed classes
class_embedding = tf.keras.layers.Embedding(
    num_class, 64, embeddings_initializer="uniform"
)
em_classes = class_embedding(classes)

# embedd images
embedded_imgs = tf.reshape(imgs, shape=(-1,) + img_shape[1:])
cnn_params = [(3, 16), (3, 32), (3, 32), (3, 32), (3, 64), (3, 64)]
fcnn = Simple_CNN(img_shape[1:], cnn_params)
embedded_imgs = fcnn(embedded_imgs)
n_shape = embedded_imgs.get_shape().as_list()
n_shape = [-1, size] + n_shape[1:]
embedded_imgs = tf.reshape(embedded_imgs, shape=n_shape)

#
em_featurs = tf.concat([embedded_imgs, boxes, em_classes, em_sides,em_scores], axis=-1)

# embedding sentence
print("creating transformer encoder")
GloVe_embeddings = np.load("word_embeddings/embedding.npy")
print(GloVe_embeddings.shape)
enc_mask = create_padding_mask(sent)
encoder = Encoder(
    num_layers=4,
    d_model=300,  # also the word embedding dim
    num_heads=12,
    dff=512,
    input_vocab_size=GloVe_embeddings.shape[0],
    embeddings_initializer=Constant(GloVe_embeddings),
)
em_sent = encoder(sent, training=True, mask=enc_mask)

# getting prediction from the Relational Neural Network
print("creating relational network")
relation_matrix = RelationalProduct()([em_sent, em_featurs])
print(relation_matrix.shape)

g = ConvolutionalPerceptron(relation_matrix.shape[1:], [512, 256, 256])
em_relations = g(relation_matrix)
relation_out = MaskedReduceMean()(em_relations, O1_mask=sent_mask, O2_mask=img_mask)

f = Perceptron(relation_out.shape[1], [256, 256, 256])
relation_out = f(relation_out)
pred = Dense(1, activation="sigmoid")(relation_out)

# compile model
print("compiling model")
model = Model(inputs=[imgs, boxes, classes, scores, sides, sent], outputs=pred)
model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

#plot_model(model, "model.png")

# model.load_weights(model_path)
checkpoint = ModelCheckpoint(
    filepath=model_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
)
lrate = LearningRateScheduler(lr_schedualer)
callbacks = [checkpoint, lrate]
model.load_weights(model_path)

print("creating generators")
sampled_images = smaple_images(train_data_dir, 1000)
sampled_images = np.stack([np.array(Image.open(path)) for path in sampled_images])
train_gen = DataGenerator(*train_data)
val_gen = DataGenerator(*dev_data)

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

