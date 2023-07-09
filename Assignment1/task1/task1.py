


import pandas as pd
import os
from PIL import Image
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score

file = '/home/schneider/Desktop/MLProject/Assignment2/FinalDataSet.csv'
data = pd.read_csv(file)
X_train, X_test ,  y_train , y_test = train_test_split(data.drop('Age', axis=1),data['Age'],random_state=42,test_size=0.2)



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


# print(len(train_images[0]))
# print((y_train))
# print(test_images)


reg = LinearRegression()
reg = reg.fit(train_images,y_train)
prediction = reg.predict(test_images)
print(reg.coef_)
print(reg.intercept_)

# acc = accuracy_score(prediction,y_test)
r2 = r2_score(prediction,y_test)
MSE = mean_squared_error(prediction,y_test)


# print("ACC: ",acc)
print("R2: ",r2)
print("MSE: ",MSE)


with open('LinearRegression.pkl', 'wb') as f:
    pickle.dump(reg, f)