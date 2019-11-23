import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from skimage import transform

DATADIR = "./images"
CATEGORIES = ['hamburger', 'pizza', 'sushi']
IMG_SIZE = 100

training_data = []

if sys.argv[1] == "new":
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num =CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    random.shuffle(training_data)

    X = []
    y = []

    for features,label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y, dtype=np.uint8)

    X = X / 255.0

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X, y, batch_size=8, validation_split=0.1, epochs=1)

    model.save("food_model.h5")
elif sys.argv[1] == "run":
    model = load_model("food_model.h5")
    test_img = Image.open(sys.argv[2])
    test_img = np.array(test_img).astype('float32')/255
    test_img = transform.resize(test_img, (IMG_SIZE, IMG_SIZE, 1))
    test_img = np.expand_dims(test_img, axis=0)

    index = np.argmax(model.predict(test_img))
    result = CATEGORIES[index]
    print(result)

# pickle_out = open("X.pickle","wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle","wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()
