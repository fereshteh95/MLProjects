import cv2 as cv
import os
import numpy as np
import pandas as pd


class ListImages:
    def __init__(self):
        pass

    @staticmethod
    def list_images(path, class_label):
        images = []
        for image_path in os.listdir(path):
            new_path = os.path.join(path, image_path)
            img = cv.imread(new_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            images.append([img, class_label])
        return images

listim = ListImages()
train_class1 = "butterflies/train/maniola_jurtina"
train_class2 = "butterflies/train/pyronia_tithonus"

val_class1 = "butterflies/valid/maniola_jurtina"
val_class2 = "butterflies/valid/pyronia_tithonus"

train_c1 = np.array(listim.list_images(train_class1, 0))
train_c2 = np.array(listim.list_images(train_class2, 1))
val_c1 = np.array(listim.list_images(val_class1, 0))
val_c2 = np.array(listim.list_images(val_class2, 1))

train_data = np.concatenate([train_c1, train_c2], axis=0)
val_data = np.concatenate([val_c1, val_c2], axis=0)

np.random.shuffle(train_data)
np.random.shuffle(val_data)

X_train = train_data[:, 0]
y_train = train_data[:, 1]
X_val = val_data[:, 0]
y_val = val_data[:, 1]
y_val = pd.get_dummies(y_val.astype(str))
X = []
for i in range(len(X_train)):
    X.append(X_train[i])

X = np.array(X)
print(X.shape)
X_train = X.reshape(1800, 75, 75, 1)
print(X_train.shape)