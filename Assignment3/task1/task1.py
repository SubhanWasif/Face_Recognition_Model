from sklearn.model_selection import train_test_split
import os 
from PIL import Image
import pandas as pd
import cv2
from numpy import asarray
from PIL import Image 
import tensorflow as tf

import numpy as np

# ===============================================================================================================

data = pd.read_csv("/home/abdullah/Desktop/zip/MLProject/Assignment4/FinalDataSet.csv")

X_train, X_test ,  y_train , y_test = train_test_split(data.drop('Name',axis=1),data['Name'],stratify=data['Name'],random_state=42,test_size=0.2)

images = []
labels=[]

for i,j in zip(data['Path'],data['Name']):
    image = cv2.imread(i)
    new_imagee = cv2.resize(image, (224, 224))
    new_imagee=np.array(new_imagee)
    images.append(new_imagee)
    if j == 'Unknown':
      labels.append(0)
    else:
       labels.append(1)
images = np.array(images)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')




# ==============================================================================================================


X_train, X_test ,  y_train , y_test = train_test_split(data.drop('Expression',axis=1),data['Expression'],stratify=data['Expression'],random_state=42,test_size=0.2)

images = []
labels=[]

for i,j in zip(data['Path'],data['Expression']):
    image = cv2.imread(i)
    new_imagee=np.array(new_imagee)
    images.append(new_imagee)
    if j == 'Smiling':
      labels.append(0)
    else:
       labels.append(1)
images = np.array(images)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')



# ====================================================================================================================


X_train, X_test ,  y_train , y_test = train_test_split(data.drop('Age',axis=1),data['Age'],stratify=data['Age'],random_state=42,test_size=0.2)

images = []
labels=[]

for i,j in zip(data['Path'],data['Age']):
    image = cv2.imread(i)
    new_imagee=np.array(new_imagee)
    images.append(new_imagee)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')