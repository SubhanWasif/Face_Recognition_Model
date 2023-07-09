
import os 
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from numpy import asarray
from PIL import Image 
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle





data = pd.read_csv("/home/abdullah/Desktop/zip/MLProject/Assignment4/FinalDataSet.csv")



print("======================================binary-label============================================")

X_train, X_test ,  y_train , y_test = train_test_split(data.drop('Name',axis=1),data['Name'],stratify=data['Name'],random_state=42,test_size=0.2)

train_images = []
for i in X_train['Path']:
    img = Image.open(i)
    img_1d = np.array(img).ravel()
    train_images.append(img_1d)
train_images = np.array(train_images)
  

test_images = []
for i in X_test['Path']:
    img = Image.open(i)
    img_1d = np.array(img).ravel()
    test_images.append(img_1d)
test_images = np.array(test_images) 




param_grid = {'C': [0.1, 1, 10, 100],'kernel': ['linear', 'poly', 'rbf']}
classifier = svm.SVC()
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(train_images, y_train)
best_C = grid_search.best_params_['C']

best_kernel = grid_search.best_params_['kernel']
best_score = grid_search.best_score_

print("Best C:", best_C)
print("Best kernel:", best_kernel)
print("Best score:", best_score)

classifier = svm.SVC(kernel=best_kernel,C=best_C)
classifier.fit(train_images, y_train)
predictions = classifier.predict(test_images)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
with open('SVMbinaryClass.pkl', 'wb') as f:
    pickle.dump(classifier, f)

#=============================================================================================================================================
print("======================================Multi-label============================================")

X_train, X_test ,  y_train , y_test = train_test_split(data.drop('Age',axis=1),data['Age'],random_state=42,test_size=0.2)

train_images = []
for i in X_train['Path']:
    img = Image.open(i)
    img_1d = np.array(img).ravel()
    train_images.append(img_1d)
train_images = np.array(train_images)
  

test_images = []
for i in X_test['Path']:
    img = Image.open(i)
    img_1d = np.array(img).ravel()
    test_images.append(img_1d)
test_images = np.array(test_images) 


classifier = svm.SVC()
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(train_images, y_train)

best_C = grid_search.best_params_['C']
best_kernal = grid_search.best_params_['kernel']
best_score = grid_search.best_score_

print("Best C:", best_C)
print("Best kernel:", best_kernel)
print("Best score:", best_score)


classifier = svm.SVC(kernel=best_kernel, C=best_C)
classifier.fit(train_images, y_train)
predictions = classifier.predict(test_images)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
with open('SVMmultiClass.pkl', 'wb') as f:
    pickle.dump(classifier, f)