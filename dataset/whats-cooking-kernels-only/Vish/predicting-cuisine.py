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
import pandas as pd
import warnings
import re

# Loading Tools
from sklearn import model_selection 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# Loading Classification Models
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Function to clean text
def CleanText(text):
	returnText = []
	text = text.lower()
	charSet = 'abcdefghijklmnopqrstuvwxyz -,'
	for each in text:
		if each in charSet:
			returnText.append(each)
	returnText = ''.join(returnText)
	# Removing extra spaces
	returnText = re.sub('  ', ' ', returnText)
	if returnText[0] == ' ':
		returnText = returnText[1:]
	return returnText

###############################################################################
# Loading data
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

# Concatenating all the data into a singel Dataframe
train['TYPE-LABEL'] = ['TRAIN'] * train.shape[0]
test['TYPE-LABEL'] = ['TEST'] * test.shape[0]
data = pd.concat([train, test],ignore_index = True, sort=False)

# Creating a string from a 'list' of ingredients
data['ingredients_list'] = data['ingredients']
data['ingredients'] = data['ingredients_list'].apply(', '.join)
# Applying a string cleaning function to the data set
data['ingredients'] = data['ingredients'].apply(lambda x: CleanText(x))

# Creating a TF-IDF vector for the text
tfidfVector = TfidfVectorizer(binary=True)
tfidfVector.fit(data['ingredients'].values)

# Creating Training vector
XTrain = tfidfVector.transform(data.loc[(data['TYPE-LABEL'] == 'TRAIN').tolist(),'ingredients'].values)
XTrain = XTrain.astype('float')
# Encoding Cusine/Target Variable
encoder = LabelEncoder()
yTrain = encoder.fit_transform(data.loc[(data['TYPE-LABEL'] == 'TRAIN').tolist(),'cuisine'].values)

# Creating Test vector
XTest = tfidfVector.transform(data.loc[(data['TYPE-LABEL'] == 'TEST').tolist(),'ingredients'].values)
XTest = XTest.astype('float')

###############################################################################
# Picking SVC Classifier
classifier = SVC(C=20,gamma=1.0)
classifier.fit(XTrain , yTrain)

# Predicting classes
yPredicted = classifier.predict(XTest)
yPredictedCuisine = encoder.inverse_transform(yPredicted)

# Creating submission file
testID = data.loc[(data['TYPE-LABEL'] == 'TEST').tolist(),'id'].values
cuisinePredictions = pd.DataFrame({'id':testID, 'cuisine':yPredictedCuisine})
cuisinePredictions = cuisinePredictions[['id', 'cuisine']]
cuisinePredictions.to_csv('sample_submission.csv', index = False)
