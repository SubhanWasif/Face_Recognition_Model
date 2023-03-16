
import io,os,csv
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
import pandas as pd
#these are the google credentails to use google vision api
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/schneider/Desktop/MLProject/Assignment/mystical-melody-376814-b2a7836d24be.json"
from google.cloud import vision
from google.cloud.vision_v1 import types
client = vision.ImageAnnotatorClient()
from PIL import Image


# directory1 is folder of celeb data and directory2 is the folder of my data
directory1 = '/home/schneider/Desktop/MLProject/dataset'
directory2 = '/home/schneider/Desktop/MLProject/mydatset'
csv_files = [directory1,directory2]


#SmileApi function is using google vision api 
# it determines if the person in the picture is smiling or not
# return backs expression of the picture 

def SmileApi(picture):
    expression = 'Not Smiling'
    image_url=(os.path.abspath(picture))
    with io.open(image_url, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    for label in labels:
        if label.description == 'Smile':
            expression = 'Smiling'
    return expression

# FileRead reads both directory 
# save it in csv file called "FinalDataSet.csv"

def FileRead():
    fields = ["Path","Name", "Expression", "Age"]
    with open("FinalDataSet.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in csv_files:
            if i == directory2:
                for picture in os.scandir(i):
                    expression = SmileApi(picture)
                    writer.writerow({"Path": os.path.abspath(picture), "Name": 'Me', "Expression": expression,'Age':'20'})
            else:
                for picture in os.scandir(i):
                    expression = SmileApi(picture)
                    writer.writerow({"Path": os.path.abspath(picture), "Name":'Unknown' , "Expression": expression,'Age':""})

# shuffle is to shuffle the csv file
# all the store we stored in the extact same file after shuffling
def shuffle(final_file):
    df = pd.read_csv(final_file)  
    final = df.sample(frac=1)
    final.to_csv(final_file, index=False)

#to split data into test and trian files
def SplitData():
    data = pd.read_csv('FinalDataSet.csv')
    X_train, X_test = train_test_split(data,test_size=0.2)
    X_train = X_train.sample(frac=1)
    X_train.to_csv("X_train.csv",index=False)
    X_test = X_test.sample(frac=1)
    X_test.to_csv("X_test.csv",index=False)


def resizing_image():
    for i in csv_files:
        for picture in os.scandir(i):
            pic = os.path.abspath(picture)
            img = Image.open(pic)
            new_imagee = img.resize(32,32)
            new_imagee.save(pic)
def reshape():
    for i in csv_files:
        for picture in os.scandir(i):
            pic = os.path.abspath(picture)
            img = Image.open(pic)
            img_array = np.array(img)
            reshape_size = (32,32,3)
            new_array = np.reshape(img_array, (32,32,3))
            new_image = Image.fromarray(new_array)
            new_image.save(pic)



# resizing_image()
# reshape()
# FileRead()
# shuffle("FinalDataSet.csv")
# SplitData()

