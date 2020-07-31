import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Read the Data
df = pd.read_csv('parkinsons.csv')
print(df.head())

# Get the features and Labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Get the Count of each label
print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])

# Scale Features
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=7
)

# Train the Model
model = XGBClassifier()
model.fit(x_train, y_train)

# Predict the test data and calculate accuracy
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
