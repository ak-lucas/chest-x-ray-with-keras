# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

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
from keras.models import load_model

from keras.applications.xception import Xception

from keras.preprocessing.image import ImageDataGenerator

path = "/data/lucas/chest_xray_20/"

# Load the dataset
test_dir = "test/"

# data generator sem o augmentation - para a validação
datagen_no_aug = ImageDataGenerator(rescale=1./255)

# Create the model
input_img = Input(shape=(299,299,3))

# pre-trained model
pt_model = Xception(
		    					include_top=False,
    							weights='imagenet',
    							input_tensor=input_img,
	    						input_shape=(299,299,3),
	    						pooling='avg'
                                                        )
print pt_model.summary()

for layer in pt_model.layers[:-15]:
	layer.trainable = False

for layer in pt_model.layers[-15:]:
    layer.trainable = True

# new fully connected layer
x = pt_model.layers[-1].output

"""
conv1 = Conv2D(32, kernel_size=(3,3), padding='valid')(x)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Conv2D(32, kernel_size=(3,3), padding='valid')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv1 = Dropout(0.25)(conv1)

conv2 = Conv2D(64, kernel_size=(3,3), padding='valid')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Conv2D(64, kernel_size=(3,3), padding='valid')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv2 = Dropout(0.25)(conv2)

conv3 = Conv2D(128, kernel_size=(2,2), padding='valid')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
conv3 = Dropout(0.25)(conv3)
"""
#flatten = Flatten()(x)

x = Dropout(0.25)(x)
fc_1 = Dense(1024, activation='selu')(x)
fc_1 = Dropout(0.5)(fc_1)
fc_2 = Dense(128, activation='selu')(fc_1)
#fc_2 = Dropout(0.25)(fc_2)
output = Dense(2, activation='softmax')(fc_2)

# Compile the model
model = Model(inputs=input_img, outputs=output)

model.load_weights(sys.argv[1])

#opt = RMSprop(lr=0.001, decay=1e-9)
#opt = Adagrad(lr=0.001, decay=1e-6)
#opt = Adadelta(lr=0.075, decay=1e-6)
opt = Adam(lr=0.001, decay=5e-6)
model.compile(loss='categorical_crossentropy',
							optimizer=opt,
							metrics=['accuracy'])

test_generator = datagen_no_aug.flow_from_directory(path+val_dir, target_size=(299,299),
																									batch_size=32,
																									color_mode='rgb',
																									class_mode='categorical',
																									seed=7)

model.evaluate_generator(
									test_generator)

#print model.summary()
