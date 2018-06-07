# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate
from keras.utils import to_categorical
from keras.models import Model
from keras.models import load_model

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

import keras
from keras import backend as K
from sklearn.metrics import f1_score
threshold = 0.8

def accuracy_with_threshold(y_true, y_pred):
	y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
	return K.mean(K.equal(y_true, y_pred))

def f_score(y_true, y_pred):
  #y_true = tf.cast(y_true, "int32")
  y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
  y_correct = y_true * y_pred
  sum_true = tf.reduce_sum(y_true, axis=1)
  sum_pred = tf.reduce_sum(y_pred, axis=1)
  sum_correct = tf.reduce_sum(y_correct, axis=1)
  precision = sum_correct / sum_pred
  recall = sum_correct / sum_true
  f_score = 2 * precision * recall / (precision + recall)
  f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
  return tf.reduce_mean(f_score)

path = "/data/lucas/chest_xray_20/"

# Load the dataset
test_dir = "test/"

# data generator sem o augmentation - para a validação
datagen_no_aug = ImageDataGenerator(rescale=1./255)

# Create the model
# Create the model
input_img = Input(shape=(224,224,3))

# pre-trained model
pt_model = VGG16(
		    					include_top=True,
    							weights='imagenet',
    							input_tensor=input_img,
	    						input_shape=(224,224,3))
#	    						pooling='avg'
#                                                        )

for layer in pt_model.layers:
	layer.trainable = False

# new fully connected layer
x = pt_model.layers[-2].output
#x = Flatten()(x)
fc_1 = Dense(128, activation='selu')(x)
#fc_1 = Dropout(0.5)(fc_1)
#fc_2 = Dense(512, activation='relu')(fc_1)
#fc_2 = Dropout(0.5)(fc_2)
output = Dense(1, activation='sigmoid')(fc_1)

# Compile the model
model = Model(inputs=input_img, outputs=output)

model.load_weights(sys.argv[1])

#opt = RMSprop(lr=0.001, decay=1e-9)
#opt = Adagrad(lr=0.001, decay=1e-6)
#opt = Adadelta(lr=0.075, decay=1e-6)
opt = Adam(lr=0.001, decay=5e-6)
model.compile(loss='binary_crossentropy',
							optimizer=opt,
							metrics=['accuracy', accuracy_with_threshold, fscore])

test_generator = datagen_no_aug.flow_from_directory(path+test_dir, target_size=(224,224),
																									batch_size=1,
																									color_mode='rgb',
																									class_mode='binary',
																									shuffle=False)

print model.evaluate_generator(test_generator)

Y_pred = model.predict_generator(test_generator) > threshold

print classification_report(test_generator.classes, Y_pred, digits=5)
