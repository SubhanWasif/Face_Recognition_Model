import pandas as pd
import numpy as np
import re ,os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

testfile = pd.read_csv('/home/schneider/Desktop/Assignment/Task3/test.csv')
trainfile = pd.read_csv('/home/schneider/Desktop/Assignment/Task3/train.csv')


#below code removes <br /> from dataset
#count keeps the track of indices
count = 0
for i in trainfile['review']:
  trainfile['review'][count] = re.sub("<br />","",i) 
  count = count+1
count = 0
for i in testfile['review']:
  testfile['review'][count] = re.sub("<br />","",i)
  count = count+1
testfile['review']
# print(trainfile['review'])
# print(testfile['review'])


le = LabelEncoder()
trainfile['sentiment'] = le.fit_transform(trainfile['sentiment'])
testfile['sentiment'] = le.fit_transform(testfile['sentiment'])
# print(trainfile['sentiment'])
# print(testfile['sentiment'])


trian_label = trainfile['sentiment'] #returns labels of the train dataset
trian_features = trainfile['review'] #returns features of the train dataset
test_label = testfile['sentiment'] #returns labels of test
test_features = testfile['review'] #returns features of test

vector = CountVectorizer() #CountVectorizer returns the array of frequencies
vector.fit_transform(trian_features) #  makes a vocabulary from the dataset 

feature_names = vector.get_feature_names_out() #returns the name of every feature in the vocabulary
vector_trian = vector.transform(trian_features) #counts the frequencies of every feature in trian dataset
# as well as the test dataset
vector_test= vector.transform(test_features) 



InfoGain = mutual_info_classif(vector_trian,trian_label,random_state=1) 
#mutual_info_classif finds the Information gain of every feature 
sorted_arr = np.argsort(InfoGain)[::-1][:]# returns the sorted indices
sorted_features = [feature_names[i] for i in sorted_arr] #using sorted indices we find sorted featues
info_gain = [InfoGain[i] for i in sorted_arr] # returns sorted information gain




Dtree = DecisionTreeClassifier(criterion = "entropy",random_state=42)
Dtree = Dtree.fit(vector_trian,trainfile['sentiment']) # fit function takes frequencies and their labels to train a model
prediction = Dtree.predict(vector_test) # returns the predict by test 
acc = accuracy_score(prediction,test_label)
f1 = f1_score(prediction,test_label)
recall = recall_score(prediction,test_label)
pre = precision_score(prediction,test_label)

print()
print("Without Feature Selection")
print("Accuracy: ",acc)
print("Precision: ",pre)
print("Recall: ",recall)
print("F1: ",f1)



print()
print("-----------------------------------------------------------------------------------------")
print()


x = pd.DataFrame.sparse.from_spmatrix(vector_trian, columns=vector.get_feature_names_out())
y = pd.DataFrame.sparse.from_spmatrix(vector_test, columns=vector.get_feature_names_out()) 
# x and y are the datafram of spmatrix
# spares returns a dataframe from numpy array 


best50 = sorted_features[:len(sorted_features)//2]
selected_dataframe = x.loc[:,best50]
selected_datafram_test = y.loc[:,best50]
#loc returns all rows but 50 % of columns from the datafram
Dtree = DecisionTreeClassifier(criterion = "entropy",random_state=42)
Dtree= Dtree.fit(selected_dataframe,trainfile['sentiment'])
prediction = Dtree.predict(selected_datafram_test) 

acc = accuracy_score(prediction,test_label)
f1 = f1_score(prediction,test_label)
recall = recall_score(prediction,test_label)
pre = precision_score(prediction,test_label)





print("With Feature Selection")
print("Accuracy: ",acc)
print("Precision: ",pre)
print("Recall: ",recall)
print("F1: ",f1)

print()
print("-----------------------------------------------------------------------------------------")
print()



directory = '/home/schneider/Desktop/Assignment/Task3/FinalDataSet.csv'
myfile = pd.read_csv(directory)

X_train, X_test,Y_trian,Y_test = train_test_split(myfile.drop('Name',axis=1),myfile['Name'],test_size=0.2,random_state=42)

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



Dtree = DecisionTreeClassifier(criterion='entropy',random_state=42)
Dtree = Dtree.fit(train_images,Y_trian )
prediction = Dtree.predict(test_images)
acc = accuracy_score(prediction,Y_test)
f1 = f1_score(prediction,Y_test,average="binary", pos_label='Me')
recall = recall_score(prediction,Y_test,average="binary", pos_label='Me')
pre = precision_score(prediction,Y_test,average="binary", pos_label='Me')

print("Image Prediction")
print("Accuracy: ",acc)
print("Precision: ",pre)
print("Recall: ",recall)
print("F1: ",f1)
