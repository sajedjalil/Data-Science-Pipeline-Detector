# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json

# Dataset Preparation
print ("Reading  Dataset  for JSON FILES... ")
print ("Hello Scientists . Stay tuned")
# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')



trainingdata = pd.read_json('../input/train.json')
testingdata = pd.read_json('../input/test.json')

trainingdata['seperated_ingredients'] = trainingdata['ingredients'].apply(','.join)
testingdata['seperated_ingredients'] = testingdata['ingredients'].apply(','.join)

string1=trainingdata['ingredients'].apply(','.join)
string2=trainingdata['ingredients'].str

print("dumping ingeredients")
print(string1(1))

print("Continue")

print('Max Num of Ingredients in a Dish: ',trainingdata['ingredients'])
print('Min Num of Ingredients in a Dish: ',trainingdata['ingredients'])


print('Max Num of Ingredients in a Dish: ',trainingdata['ingredients'].str.len().max())
print('Min Num of Ingredients in a Dish: ',trainingdata['ingredients'].str.len().min())




