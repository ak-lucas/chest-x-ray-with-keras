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

from keras.preprocessing.image import ImageDataGenerator

path = "/data/lucas/chest_xray/"

# Load the dataset
train_dir = "train/"
val_dir = "val/"

# data generator com augmentation - para o treino
datagen_aug = ImageDataGenerator(
#    width_shift_range=0.2,
#    height_shift_range=0.2,
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=False)

# data generator sem o augmentation - para a validação
datagen_no_aug = ImageDataGenerator(rescale=1./255)

# Create the model
input_img = Input(shape=(150,150,1))

# inception module with dimension reduction
flow_1 = Conv2D(32, kernel_size=(1, 1), padding='same', strides=(2,2))(input_img)
#flow_1 = BatchNormalization()(flow_1)
flow_1 = Activation('relu')(flow_1)

flow_2 = Conv2D(32, kernel_size=(1, 1), padding='same')(input_img)
#flow_2 = BatchNormalization()(flow_1)
flow_2 = Activation('relu')(flow_2)
flow_2 = Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2,2))(flow_2)
#flow_2 = BatchNormalization()(flow_1)
flow_2 = Activation('relu')(flow_2)
#flow_2 = MaxPooling2D(pool_size=(2,2))

flow_3 = Conv2D(32, kernel_size=(1, 1), padding='same')(input_img)
#flow_3 = BatchNormalization()(flow_1)
flow_3 = Activation('relu')(flow_3)
flow_3 = Conv2D(32, kernel_size=(5, 5), padding='same', strides=(2,2))(flow_3)
#flow_3 = BatchNormalization()(flow_1)
flow_3 = Activation('relu')(flow_3)
#flow_3 = MaxPooling2D(pool_size=(2,2))

flow_4 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(1, 1))(input_img)
flow_4 = Conv2D(32, kernel_size=(5, 5), padding='same', strides=(2,2))(flow_4)
#flow_4 = BatchNormalization()(flow_1)
flow_4 = Activation('relu')(flow_4)

concat = concatenate([flow_1, flow_2, flow_3, flow_4], axis=3)
#concat = concatenate([flow_2, flow_3],axis=3)

flatten = Flatten()(concat)

# fully connected

fc_1 = Dense(128, activation='selu', kernel_initializer='lecun_uniform')(flatten)
fc_1 = Dropout(0.25)(fc_1)

#fc_2 = Dense(64, activation='selu', kernel_initializer='lecun_uniform')(fc_1)
#fc_2 = Dropout(0.5)(fc_2)

output = Dense(2, kernel_initializer='lecun_uniform')(fc_1)
output = BatchNormalization()(output)
output = Activation('softmax')(output)

# Compile the model
model = Model(inputs=input_img, outputs=output)

#opt = RMSprop(lr=0.001, decay=1e-9)
#opt = Adagrad(lr=0.001, decay=1e-6)
#opt = Adadelta(lr=0.075, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-6)
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
train_generator = datagen_aug.flow_from_directory(path+train_dir, target_size=(150,150),
																									batch_size=128,
																									color_mode='grayscale',
																									class_mode='categorical',
																									seed=7,
																									)
val_generator = datagen_no_aug.flow_from_directory(path+val_dir, target_size=(150,150),
																									batch_size=128,
																									color_mode='grayscale',
																									class_mode='categorical',
																									seed=7)
model.fit_generator(
									train_generator,
									steps_per_epoch=37, # partition size / batch size
									epochs=500,
									shuffle=True,
									validation_data=val_generator,
									callbacks=[EarlyStopping(min_delta=0.001, patience=10), CSVLogger('training.log', separator=',', append=False), checkpoint])
