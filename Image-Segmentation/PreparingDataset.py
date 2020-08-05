import cv2 as cv
import os
import numpy as np


class ListImages:
    def __init__(self):
        pass

    @staticmethod
    def list_images(path):
        images = []
        for image_path in os.listdir(path):
            new_path = os.path.join(path, image_path)
            img = cv.imread(new_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            images.append(img)
        return images

listim = ListImages()
train_path = "dataset/images_prepped_train"
train_label_path = "dataset/annotations_prepped_train"
test_path = "dataset/images_prepped_test"
test_label_path = "dataset/annotations_prepped_test"

train_images = np.array(listim.list_images(train_path))
train_labels = np.array(listim.list_images(train_label_path))
test_images = np.array(listim.list_images(test_path))
test_labels = np.array(listim.list_images(test_label_path))

# Input Shapes
num_train_images = train_images.shape[0]
num_test_images = test_images.shape[0]
num_channels = train_images.shape[3]
input_shape = (train_images.shape[1], train_images.shape[2], num_channels)

print(input_shape)