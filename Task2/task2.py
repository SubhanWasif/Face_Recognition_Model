import pandas as pd
import numpy as np
import re 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

testfile = pd.read_csv('/home/schneider/Desktop/Assignment/Task2/test.csv')
trainfile = pd.read_csv('/home/schneider/Desktop/Assignment/Task2/train.csv')



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
vectortrian = vector.transform(trian_features) #counts the frequencies of every feature in trian dataset
# as well as the test dataset
vector_test= vector.transform(test_features) 



InfoGain = mutual_info_classif(vectortrian,trian_label,random_state=32) 
#mutual_info_classif finds the Information gain of every feature 
sorted_arr = np.argsort(InfoGain)# returns the sorted indices
sorted_features = [feature_names[i] for i in sorted_arr] #using sorted indices we find sorted featues
info_gain = [InfoGain[i] for i in sorted_arr] # returns sorted information gain


first10 = sorted_features[-10:] # top ten sorted features
last10 = sorted_features[:10] #last ten sorted featues
print("Top Ten Sorted Features: ")
print(first10)
print("Last Ten Sorted Features: ")
print(last10)