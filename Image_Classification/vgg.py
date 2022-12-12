import keras
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import random as rand
#!pip install opencv-python
import cv2
import math
from skimage.io import imshow
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras import backend as K





def processing_image(dataset_path):
    train_image = []

    img = load_img(dataset_path, target_size=(256,256,3))
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
    X = np.array(train_image)
    return X

MODEL_PATH = "vgg19.checkpoint"
IMG_PATH = "ai_ready/images/silos_256-0-0--6-14--19-28655.png"
THRESHOLD = 0.5

model_load = keras.models.load_model(MODEL_PATH)
img = processing_image(IMG_PATH) 
img = img.reshape(1,256,256,3)
predictions = np.round(model_load.predict(img))
print(predictions)

