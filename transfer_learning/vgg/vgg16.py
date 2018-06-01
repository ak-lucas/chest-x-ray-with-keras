# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate
from keras.utils import to_categorical
from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

path = "/data/lucas/chest_xray_20/"

# Load the dataset
train_dir = "train/"
val_dir = "val/"

# data generator com augmentation - para o treino
datagen_aug = ImageDataGenerator(
#    width_shift_range=0.2,
#    height_shift_range=0.2,
    rescale=1./255,
    rotation_range=5,
    horizontal_flip=False)

# data generator sem o augmentation - para a validação
datagen_no_aug = ImageDataGenerator(rescale=1./255)

# Create the model
input_img = Input(shape=(224,224,3))

# pre-trained model
pt_model = VGG16(
		    					include_top=False,
    							weights='imagenet',
    							input_tensor=input_img,
	    						input_shape=(224,224,3),
	    						pooling='max'
                                                        )

for layer in pt_model.layers[:-2]:
	layer.trainable = False

for layer in pt_model.layers[-2:]:
    layer.trainable = True
# new fully connected layer
x = pt_model.output
fc_1 = Dense(512, activation='selu')(x)
fc_1 = Dropout(0.5)(fc_1)
fc_2 = Dense(512, activation='selu')(fc_1)
fc_2 = Dropout(0.25)(fc_2)
output = Dense(2, activation='softmax')(fc_2)

# Compile the model
model = Model(inputs=input_img, outputs=output)

print model.summary()

#opt = RMSprop(lr=0.001, decay=1e-9)
#opt = Adagrad(lr=0.001, decay=1e-6)
#opt = Adadelta(lr=0.075, decay=1e-6)
opt = Adam(lr=0.0005, decay=5e-6)
model.compile(loss='categorical_crossentropy',
							optimizer=opt,
							metrics=['accuracy'])

checkpoint = ModelCheckpoint('saved_models/model_{epoch:002d}--{loss:.2f}--{val_loss:.2f}.hdf5',
              save_best_only=True,
              save_weights_only=False)

# treina e valida o modelo - sem data augmentation
#model.fit(X_train[train_idx], to_categorical(Y_train[train_idx]),
#					batch_size=128,
#					shuffle=True,
#					epochs=250,
#					validation_data=(X_train[val_idx], to_categorical(Y_train[val_idx])),
#					callbacks=[EarlyStopping(min_delta=0.001, patience=10), CSVLogger('training_fold_' + str(fold) + '.log', separator=',', append=False), checkpoint])

# treina e valida o modelo - com data augmentation
train_generator = datagen_aug.flow_from_directory(path+train_dir, target_size=(224,224),
																									batch_size=128,
																									color_mode='rgb',
																									class_mode='categorical',
																									seed=7,
																									)
val_generator = datagen_no_aug.flow_from_directory(path+val_dir, target_size=(224,224),
																									batch_size=128,
																									color_mode='rgb',
																									class_mode='categorical',
																									seed=7)

model.fit_generator(
									train_generator,workers=1,
									class_weight={0:4, 1:1}, # balance
									steps_per_epoch=33, # partition size / batch size
									epochs=500,
									shuffle=True,
                  max_queue_size=30,
									validation_data=val_generator,
									callbacks=[EarlyStopping(min_delta=0.001, patience=20), CSVLogger('training.log', separator=',', append=False), checkpoint])

print model.summary()