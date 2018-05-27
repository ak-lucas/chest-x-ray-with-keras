import cv2
import os
import numpy as np

from sklearn.utils import shuffle as np_shuffle

import matplotlib.image as mpimg

class XRay:
  def __init__(self):
    pass

  def load_images(self, folder_p, folder_n, target_size, shuffle=True):
    images = []
    labels = []
    # load class pneumonia
    for filename in os.listdir(folder_p):
      img = mpimg.imread(os.path.join(folder_p, filename))
      if img is not None:
        if img.shape[-1] == 3:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)
        images.append(img)
        labels.append(1)
      else:
        print "algo deu errado"
        exit(1)
        
    # load class normal
    for filename in os.listdir(folder_n):
      img = mpimg.imread(os.path.join(folder_n, filename))
      if img is not None:
        if img.shape[-1] == 3:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)
        images.append(img)
        labels.append(0)

      else:
        print "algo deu errado"
        exit(1)
          
    if shuffle:
      images, labels = np_shuffle(images, labels, random_state=0)
      
    return images, labels