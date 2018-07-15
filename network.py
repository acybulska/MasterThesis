from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(32, (2, 2)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Conv2D(64, (2, 2)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (2, 2)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.1))

        model.add(Conv2D(128, (2, 2)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(1000, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(500, activation="relu"))

        # softmax classifier
        model.add(Dense(classes, activation="softmax"))

        # return the constructed network architecture
        return model

class LeNet2:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (8, 8), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.05))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import re
from PIL import Image

EPOCHS = 100
INIT_LR = 1e-4
BS = 32 # batchsize
HEIGHT = 100
WIDTH = 100


def top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


top_3.__name__='top_3'

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images('/Users/lisa/DB_Full/')))
random.seed(20)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    imgData=img_to_array(image)
    data.append(imgData)
    label = int(re.findall("0(\d+)*", imagePath)[0][0])
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42, shuffle=True)

trainY = to_categorical(trainY-1, num_classes=10)
testY = to_categorical(testY-1, num_classes=10)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=WIDTH, height=HEIGHT, depth=3, classes=10)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy", top_3])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("f14.model")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["top_3"], label="top3")
plt.plot(np.arange(0, N), H.history["val_top_3"], label="val_top3")


plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("f14.jpg")