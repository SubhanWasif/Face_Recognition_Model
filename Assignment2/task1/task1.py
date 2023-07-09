

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import os
from PIL import Image 
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import r2_score, mean_squared_error

param_grid={"penalty" :["l2", "l1", "elasticnet", None], 
           "alpha" :[0.0001, 0.001,0.00002,0.0110]}
file = '/home/schneider/Desktop/ML/Assignment3/FinalDataSet.csv'




data = pd.read_csv(file)
X_train, X_test ,  y_train , y_test = train_test_split(data.drop(['Age'], axis=1),data.Age,random_state=42,test_size=0.2)
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

grid = GridSearchCV(SGDClassifier(),param_grid=param_grid,cv=5)
grid.fit(train_images,y_train)
SGD = grid.best_estimator_
y_pred = SGD.predict(test_images)
print(y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
grid_r2 = r2_score(y_test,y_pred)

print(r2)
print(mse)

alpha =[0.0001, 0.001,0.00002,0.0110]

ridge = RidgeCV(alphas=alpha,cv=5)
ridge.fit(train_images,y_train)
best_ridge = ridge.alpha_

Elastic = ElasticNetCV(alphas=alpha,cv =5)
Elastic.fit(train_images,y_train)
best_elastic = Elastic.alpha_

laso = LassoCV(alphas=alpha,cv = 5)
laso.fit(train_images,y_train)
best_laso = laso.alpha_

print(best_ridge)
print(best_elastic)
print(best_laso)



ridge_model = Ridge(alpha=best_ridge)
ridge_model.fit(train_images, y_train)
y_pred = ridge.predict(test_images)
ridge_mse = mean_squared_error(y_test, y_pred)
ridge_r2 = r2_score(y_test,y_pred)


elastic_model = ElasticNet(alpha=best_elastic)
elastic_model.fit(train_images, y_train)
y_pred = elastic_model.predict(test_images)
elastic_mse = mean_squared_error(y_test, y_pred)
elastic_r2 = r2_score(y_test,y_pred)


laso_model = Lasso(alpha=best_laso)
laso_model.fit(train_images, y_train)
y_pred = laso_model.predict(test_images)
laso_mse = mean_squared_error(y_test, y_pred)
laso_r2 = r2_score(y_test,y_pred)


print("MSE of Ridge Model:", ridge_mse  )
print("MSE of ElasticNet Model:", elastic_mse  )
print("MSE of Lasso Model:", laso_mse  )

print(grid_r2)
print(ridge_r2)
print(elastic_r2)
print(laso_r2)



with open('/home/schneider/Desktop/ML/Assignment3/task1/Ridge_Model.pkl', 'wb') as f:
    pickle.dump(ridge, f)