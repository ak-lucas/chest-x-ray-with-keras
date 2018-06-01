# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

import numpy as np

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
		    					include_top=False,
    							weights='imagenet',
    							input_tensor=input_img,
	    						input_shape=(224,224,3),
	    						pooling='avg'
                                                        )

for layer in pt_model.layers:
	layer.trainable = False

# new fully connected layer
x = pt_model.output
fc_1 = Dense(512, activation='selu')(x)
fc_2 = Dense(512, activation='selu')(fc_1)

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

test_generator = datagen_no_aug.flow_from_directory(path+test_dir, target_size=(224,224),
																									batch_size=1,
																									color_mode='rgb',
																									class_mode='categorical',
																									shuffle=False)

print model.evaluate_generator(test_generator)

Y_pred = np.argmax(model.predict_generator(test_generator), axis=1)
print classification_report(test_generator.classes, Y_pred, digits=5)