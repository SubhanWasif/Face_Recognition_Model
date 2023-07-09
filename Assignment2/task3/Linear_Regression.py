

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error
import os
from PIL import Image 
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score,f1_score


file = '/home/schneider/Desktop/ML/Assignment3/FinalDataSet.csv'
data = pd.read_csv(file)
X_train, X_test ,  y_train , y_test = train_test_split(data.drop(['Age'], axis=1),data.Age,random_state=42,test_size=0.5)
train_images = []
for i in X_train['Path']:
    pic = os.path.abspath(i)
    img = Image.open(pic)
    img_1d = np.array(img).ravel()
    train_images.append(img_1d)
train_images = np.array(train_images)
  

test_images = []
for i in X_test['Path']:
    pic = os.path.abspath(i)
    img = Image.open(pic)
    img_1d = np.array(img).ravel()
    test_images.append(img_1d)
test_images = np.array(test_images) 
print(train_images.shape)


def processimage(image):
    new_image=[]
    new_size = (32,32,3)
    resized_img = np.resize(image,new_size)
    img_ = np.divide(resized_img, 255)
    img = img_.reshape(3072)
    new_image.append(img)
    return new_image



inputs = Input(train_images)
outputs = Dense(1)(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='sgd', loss='mean_squared_error')
checkpoint = ModelCheckpoint('linear_regression_model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(train_images, test_images, epochs=50, callbacks=[checkpoint])
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
