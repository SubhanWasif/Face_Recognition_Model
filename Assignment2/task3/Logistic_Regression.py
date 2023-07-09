
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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



model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=(3072,), activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])


checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(train_images, test_images, epochs=50, callbacks=[model_checkpoint_callback])


model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(X_test, y_test)


print('Test loss:', loss)
print('Test accuracy:', accuracy)