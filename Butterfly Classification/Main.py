import cv2 as cv
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, Input
from keras import optimizers, losses
from sklearn.metrics import accuracy_score

def list_images(path, class_label):
    images = []
    for image_path in os.listdir(path):
        new_path = os.path.join(path, image_path)
        img = cv.imread(new_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        images.append([img, class_label])
    return images

train_class1 = "butterflies/train/maniola_jurtina"
train_class2 = "butterflies/train/pyronia_tithonus"

val_class1 = "butterflies/valid/maniola_jurtina"
val_class2 = "butterflies/valid/pyronia_tithonus"

train_c1 = np.array(list_images(train_class1, 0))
train_c2 = np.array(list_images(train_class2, 1))
val_c1 = np.array(list_images(val_class1, 0))
val_c2 = np.array(list_images(val_class2, 1))

train_data = np.concatenate([train_c1, train_c2], axis=0)
val_data = np.concatenate([val_c1, val_c2], axis=0)

np.random.shuffle(train_data)
np.random.shuffle(val_data)

X_train = train_data[:, 0]
y_train = train_data[:, 1]
X_val = val_data[:, 0]
y_val = val_data[:, 1]
x_train = []
for i in range(len(X_train)):
    x_train.append(X_train[i])

x_val = []
for i in range(len(X_val)):
    x_val.append(X_val[i])

X_val = np.array(x_val)
X_train = np.array(x_train)
X_train = X_train/255
X_train = X_train.reshape(1800, 75, 75, 1)
X_val = X_val/255
X_val = X_val.reshape(600, 75, 75, 1)
y_val = pd.get_dummies(y_val.astype(str))
y_train = pd.get_dummies(y_train.astype(str))
X_test = X_val[0:300][:]
y_test = y_val[0:300][:]
X_val = X_val[300:][:]
y_val = y_val[300:][:]

model = Sequential()

# Add Model Layers
model.add(Conv2D(32, kernel_size=5, padding='SAME',
                 data_format='channels_last',
                 use_bias='True',
                 activation='relu',
                 input_shape=(75, 75, 1)))

model.add(MaxPool2D(pool_size=(2, 2), padding='SAME'))

model.add(Conv2D(64, kernel_size=5, padding='SAME',
                 data_format='channels_last',
                 use_bias='True',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='SAME'))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(2, activation='softmax'))

adam = optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.fit(
    X_train,
    y_train,
    epochs=5,
    shuffle=True,
    validation_data=(X_val, y_val),
    verbose=2
)

predicted_y = model.predict(X_test)
y_pred = np.zeros((300, 1))
y_act = np.zeros((300, 1))
for i in range(len(y_pred)):
    y_pred[i] = list(predicted_y[i][:]).index(predicted_y[i][:].max())
    y_act[i] = list(y_test[i][:]).index(y_test[i][:].max())

acc = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_act[i]:
        acc += 1

test_acc = acc/len(y_pred)
print(test_acc)
