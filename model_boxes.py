import os
import sys
import numpy as np
from math import cos,pi
assert len(sys.argv) > 2

orig_dir = "/specific/netapp5/joberant/home/ofermagen/"
train_data = (
    orig_dir + "nlvr/nlvr2/data/train.json",
    orig_dir + "pretrained_cnn_objects/train/",
)
dev_data = (
    orig_dir + "nlvr/nlvr2/data/dev.json",
    orig_dir + "pretrained_cnn_objects/dev/",
)

RES_DIR = "/specific/disk1/home/gamir/ofer/" + sys.argv[1]
if not os.path.isdir(RES_DIR): os.mkdir(RES_DIR)
model_path =  os.path.join(RES_DIR,"model{epoch:08d}.h5")
history_path = os.path.join(RES_DIR,"metrics.json")

if not os.path.isdir(os.path.dirname(model_path)): 
    os.mkdir(os.path.dirname(model_path))

for p in [train_data, dev_data]:
    assert os.path.isdir(p[1])
    assert os.path.isfile(p[0])



from utils import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.initializers import Constant
from RN import (
    ReduceMean,
    MaskedReduceMean,
    RelationalProduct,
    TimedPerceptron,
    Perceptron,
)
from LSTM import Encoder

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from utils import HistorySaver, params_number
from generator_boxes import DataGenerator

NUM_EPOCHS = 200
PERRIOD = 40

def lr_schedualer(epoch, *a, **kw):
    b = 1.5e-4
    if epoch == 0: return b
    epoch -= 1
    x = (epoch % PERRIOD) / PERRIOD
    x = b/2 * (1+cos(pi*x))
    return x

# defining model's inputs
size = 30
features_dim = 2048 + 4 + 1  # image embedding + box + score

features = Input(shape=(size, features_dim), name="img", dtype="float32")
sides = Input(shape=(size,), name="img_sides", dtype="int32")
sent = Input(shape=(40,), name="sent", dtype="int32")

feature_mask = tf.math.not_equal(sides, 0)
sent_mask = tf.math.not_equal(sent, 0)

class CastSides(tf.keras.layers.Layer):
    def __init__(self):
        super(CastSides,self).__init__()

    def call(self,sides):
        em_sides = tf.keras.backend.cast(sides, "float32")
        em_sides = tf.keras.backend.expand_dims(em_sides,-1)
        return em_sides

#generate features
em_sides = CastSides()(sides)
em_features = Concatenate(axis=-1)([features, em_sides])
    
# embedd features
prec_params = [(1024, "relu"),(1024,"relu")]
prec = TimedPerceptron((size,features_dim+1), prec_params)
em_features = prec(em_features)

# embedding sentence
print("creating transformer encoder")
GloVe_embeddings = np.load("word_embeddings/embedding.npy")
print(GloVe_embeddings.shape)
prec_params = [(1024, "relu"),(1024, "relu")]
encoder = Encoder(
    units=512,
    prec_params=prec_params,
    input_vocab_size=GloVe_embeddings.shape[0],
    word_dim=300,  # also the word embedding dim
    embeddings_initializer=Constant(GloVe_embeddings),
)
em_sent = encoder(sent)

# creating the Relational Neural Network
print("creating relational network")
relation_matrix = RelationalProduct()([em_sent, em_features])
print(relation_matrix.shape)
prec_params = [(1024, "relu"),(1024, "relu")]
g = TimedPerceptron(relation_matrix.shape[1:], prec_params)
em_relations = g(relation_matrix)
relation_out = MaskedReduceMean()(em_relations, O1_mask=sent_mask, O2_mask=feature_mask)

# getting prediction from averaged relation
prec_params = [
    (1024, "relu"),
    (512, "relu"),
    (256, "relu"),
    (128, "relu"),
    (16, "relu"),
    (1, "sigmoid"),
]
f = Perceptron(relation_out.shape[1], prec_params,dropout=False)
pred = f(relation_out)

# compile model
print("compiling model")
model = Model(inputs=[features, sides, sent], outputs=pred)
model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

# callbacks
checkpoint = ModelCheckpoint(
    filepath=model_path, verbose=1, mode="max",period=PERRIOD
)
lrate = LearningRateScheduler(lr_schedualer)
saver = HistorySaver(history_path)
callbacks = [checkpoint, lrate, saver]

# generators
print("creating generators")
train_gen = DataGenerator(*train_data, batch_size=32,augment=True)
val_gen = DataGenerator(*dev_data, batch_size=32,augment=False)


# loading weights
# training
print("paramater number",params_number())
print("training model")
model.fit_generator(
    train_gen,
    epochs=NUM_EPOCHS,
    verbose=1,
    workers=4,
    callbacks=callbacks,
    validation_data=val_gen,
    shuffle=False,
)
