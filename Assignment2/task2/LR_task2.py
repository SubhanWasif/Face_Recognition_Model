


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os
from PIL import Image 
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score,f1_score


file = '/home/schneider/Desktop/ML/Assignment3/FinalDataSet.csv'
data = pd.read_csv(file)
X_train, X_test ,  y_train , y_test = train_test_split(data.drop(['Name'], axis=1),data.Name,random_state=42,test_size=0.3)
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



logreg = LogisticRegression()
logreg.fit(train_images, y_train)
logreg_pred = logreg.predict(test_images)
log_f1 = f1_score(logreg_pred,y_test,pos_label='Unknown')
log_acc = accuracy_score(logreg_pred,y_test)
print(log_acc)

clf = DecisionTreeClassifier(random_state = 8)
clf.fit(train_images,y_train)
tree_pred = clf.predict(test_images)
tree_f1 = f1_score(tree_pred,y_test,pos_label='Unknown')


print("Logistic F1 score is: ",log_f1)
print("Decision Tree F1 score is: ",tree_f1)


with open('/home/schneider/Desktop/ML/Assignment3/task2/LR.pkl', 'wb') as f:
    pickle.dump(logreg, f)