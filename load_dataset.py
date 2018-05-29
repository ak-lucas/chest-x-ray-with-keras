import cv2
import os
import numpy as np

from sklearn.utils import shuffle as np_shuffle

import matplotlib.image as mpimg

class XRay:
  def __init__(self):
    pass

  def load_images(self, folder_p, folder_n, target_size):
    images_p = []
    labels_p = []

    images_n = []
    labels_n = []
    # load class pneumonia
    for filename in os.listdir(folder_p):
      img = mpimg.imread(os.path.join(folder_p, filename))
      if img is not None:
        if img.shape[-1] == 3:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, target_size)
        img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images_p.append(img)
        labels_p.append(1)
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
        img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images_n.append(img)
        labels_n.append(0)

      else:
        print "algo deu errado"
        exit(1)
      
    return np.expand_dims(np.array(images_p), axis=-1), np.array(labels_p), np.expand_dims(np.array(images_n), axis=-1), np.array(labels_n)
