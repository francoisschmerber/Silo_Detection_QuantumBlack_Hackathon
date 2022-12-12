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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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



    
def oversampler(X, y):    
    X = list(X)
    counter = int(y.mean() * len(y))
    angles = [90, 180, 270]
    i = 0
    angle = 90
    while counter / len(y) < 0.5:
        for i in range(len(y)):
            if y[i] == 1:
                # get dims, find center
                image = X[i]
                (h, w) = image.shape[:2]
                (cX, cY) = (w // 2, h // 2)

                # grab the rotation matrix (applying the negative of the
                # angle to rotate clockwise), then grab the sine and cosine
                # (i.e., the rotation components of the matrix)
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))

                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY

                # perform the actual rotation and return the image
                image = cv2.warpAffine(image, M, (nW, nH), False)

                X.append(image)
                y = np.append(y, y[i])
                counter += 1
            if counter / len(y) >= 0.5:
                break

        i += 1
        angle = angles[i%3]
    X = np.array(X)
    return X, y

def processing_image(df):
    train_image = []
    for i in tqdm(range(df.shape[0])):
        img = load_img(dataset_path + "train/" + df["filename"][i], target_size=(224,224,3))
        img = img_to_array(img)
        img = img/255
        train_image.append(img)
    X = np.array(train_image)
    return X

class model_cnn():
    def __init__(self):
        
        base_model = Sequential([
          layers.Rescaling(1./255, input_shape=(224, 224, 3)),
          layers.Conv2D(16, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          #layers.Flatten(),
          #layers.Dense(512, activation='relu'),
        ])

         # Flatten the output layer to 1 dimension
        x = layers.Flatten()(base_model.output)
        # Add a fully connected layer with 512 hidden units and ReLU activation
        x = layers.Dense(512, activation='relu')(x)
        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5)(x)
        # Add a final sigmoid layer with 1 node for classification output
        x = layers.Dense(1, activation='sigmoid')(x)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model = tf.keras.models.Model(base_model.input, x)
        # compile the model
        self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',
                metrics=[AUC()])

    def train(self, X_train, y_train,validation_data, epochs, batch_size):
      
        self.model.fit(X_train, y_train, validation_data = validation_data, epochs = epochs, batch_size = batch_size)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
    
    def predict(self, df):
        test_pred = self.model.predict(df)
        test_pred_1 = np.round(test_pred)
      
        return test_pred_1
    

IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT =224,224
MODEL_PATH = "test_.checkpoint"
IMG_PATH = "../ai_ready/images/silos_256-0-0--6-14--19-28655.png"
THRESHOLD = 0.5

model_load = keras.models.load_model(MODEL_PATH)
img = load_img(IMG_PATH, grayscale=False,target_size=(IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT)) 
img = img_to_array(img)
img = img.astype('float32') / 255.0
imshow(img)
plt.show()
img = np.expand_dims(img, axis=0)
predictions = model_load.predict(img)
img = np.expand_dims(img, axis=0)
predictions = np.round(model_load.predict(img))
