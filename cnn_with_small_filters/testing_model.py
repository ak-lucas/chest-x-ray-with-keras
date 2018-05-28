# -*- coding: utf8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import itertools

import sys

sys.path.insert(0, "../")

from load_dataset import XRay

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from keras.models import load_model

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import numpy as np
import sys


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', input_shape=(150, 150, 1))) # 30x30
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid')) # 28x28
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 14x14
model.add(Dropout(0.25))


model.add(Conv2D(64, kernel_size=(3, 3), padding='valid')) # 12x12
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid')) # 10x10
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 5x5
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2, 2), padding='valid')) # 4x4
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Conv2D(128, kernel_size=(2, 2), padding='valid')) # 3x3
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 2x2
model.add(Dropout(0.25))  

model.add(Flatten())

model.add(Dense(128, activation='selu', kernel_initializer='lecun_uniform'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='selu', kernel_initializer='lecun_uniform'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='selu', kernel_initializer='lecun_uniform'))

model.add(Dense(2, kernel_initializer='lecun_uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

try:
	model.load_weights(sys.argv[1])
	opt = Adam(lr=0.000001, decay=1e-4)
	model.compile(loss='categorical_crossentropy',
								optimizer=opt,
								metrics=['accuracy'])

except IOError:
	print "IOError: não é possível abrir o arquivo", sys.argv[1]
	exit(1)

except:
	print "Unknown Error: algo deu errado"
	exit(1)

# Load the dataset
path = "/data/lucas/"

# Load the dataset
test_dir_p =  "chest_xray/test/PNEUMONIA"
test_dir_n =  "chest_xray/test/NORMAL"

xray = XRay()
X_p, Y_p, X_n, Y_n = xray.load_images(path + test_dir_p, path + test_dir_n, target_size=(150,150))

X_test = np.concatenate((X_p, X_n), axis=0)
Y_test = np.concatenate((Y_p, Y_n), axis=0)

# Evaluate the model
scores = model.evaluate(X_test, to_categorical(Y_test))
print('Accuracy: %.3f' % scores[1])

Y_pred = np.argmax(model.predict(X_test), axis=1)
print Y_pred.shape, Y_test.shape
print classification_report(Y_test, Y_pred)

# adaptado de: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
																			normalize=False,
																			title='Confusion matrix',
																			cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(Y_test, Y_pred)

class_names = [
								'NORMAL',
								'PNEUMONIA'
							]
# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cm, classes=class_names,
#                      title='Confusion matrix, without normalization')
#plt.show()
#plt.savefig('confusion_matrix.png')
