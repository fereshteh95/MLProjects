from keras_segmentation.models.unet import vgg_unet
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

model = vgg_unet(n_classes, 416, 608)

model.train(
    train_images=train_path,
    train_annotations=train_label_path,
    checkpoints_path="dataset/vgg_unet_1",
    epochs=5
)

output_path = "dataset/output2"
i = 0
for image in os.listdir(test_path):
    i += 1
    path = os.path.join(output_path, image)
    out = model.predict_segmentation(
        inp=path,
        out_fname=output_path+'/output_{}.png'.format(1)
    )
model.save("dataset/vgg_unet1")
