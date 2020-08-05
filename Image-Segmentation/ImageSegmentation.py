from keras.layers import Input, Conv2D, Dropout, MaxPool2D, UpSampling2D, concatenate
from keras.models import Sequential
from keras_segmentation.models.model_utils import get_segmentation_model
import numpy as np
import os
import cv2 as cv


def list_images(path):
    images = []
    for image_path in os.listdir(path):
        new_path = os.path.join(path, image_path)
        img = cv.imread(new_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        images.append(img)
    return images


# Preparing Data
# listimages = ListImages()
train_path = "dataset/images_prepped_train"
train_label_path = "dataset/annotations_prepped_train"
test_path = "dataset/images_prepped_test"
test_label_path = "dataset/annotations_prepped_test"

train_images = np.array(list_images(train_path))
train_labels = np.array(list_images(train_label_path))
test_images = np.array(list_images(test_path))
test_labels = np.array(list_images(test_label_path))

# Input Shapes
num_train_images = train_images.shape[0]
num_test_images = test_images.shape[0]
num_channels = train_images.shape[3]
input_shape = (train_images.shape[1], train_images.shape[2], num_channels)
n_classes = 50

# Defining Model
model = Sequential()
# Encoder Layer

input_image = Input(shape=input_shape)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last', use_bias=True)(input_image)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last', use_bias=True)(conv1)
pool1 = MaxPool2D((2, 2), padding='SAME')(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last', use_bias=True)(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last', use_bias=True)(conv2)
pool2 = MaxPool2D((2, 2), padding='SAME')(conv2)

# Decoder Layer
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last', use_bias=True)(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last', use_bias=True)(conv3)
up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2])
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)
up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1])
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)
out = Conv2D(n_classes, (1, 1), padding='same', data_format='channels_last')(conv5)

# Get Segmentation Model
model = get_segmentation_model(input_image, out)

# Training the Model
model.train(
    train_images=train_path,
    train_annotations=train_label_path,
    checkpoints_path="dataset/ManualModel",
    epochs=5
)

output_path = "dataset/output"
i = 0
for image in os.listdir(test_path):
    i += 1
    path = os.path.join(output_path, image)
    out = model.predict_segmentation(
        inp=path,
        out_fname=output_path+'/output_{}.png'.format(1)
    )
model.save("dataset/manualmodel")