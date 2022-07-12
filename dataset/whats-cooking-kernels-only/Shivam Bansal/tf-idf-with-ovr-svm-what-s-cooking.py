"""
Problem Statement: SVM script for multiclass classification 

What's Cooking : Tf Idf with One Vs Rest Support Vector Machine (SVM) Model
Goal: Use recipe ingredients to categorize the cuisine

Input : Text Data (Ingredients for a Cusine)
Output : Single Class (Cusine Class)

author = sban (https://www.kaggle.com/shivamb)
created date = 26 June, 2018
"""

# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json

# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

df = pd.DataFrame()
df['cuisine_description'] = train_text
df['cuisine'] = target
df.to_csv("cuisine_data.csv", index = False)


