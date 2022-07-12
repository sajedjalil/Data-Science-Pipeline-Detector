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
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
from sklearn.externals import joblib

#import the data
data = pd.read_csv('../input/train.csv')

#correlate the data
corr = data.corr()

#remove the obvious candidates
corr = corr.drop(['Id'], axis=0)

#select the strength of the correlation
reduced_corr = corr[(corr.Cover_Type < -0.01)|(corr.Cover_Type > 0.01)&(corr.Cover_Type != 1)]

#extract row headings
reduced_inputs= list(reduced_corr.index)

#display
reduced_corr

#use row headings to reduce main list
data_inputs = data[reduced_inputs]
data_inputs.head()

#Set the target
expected_output = data[["Cover_Type"]]
expected_output.head()

#Slice and dice the data
inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

#check inputs
print(inputs_train.head())
print(expected_output_train.head())

#Create the machine
rf = RandomForestClassifier (n_estimators=100)

#Run the machine
rf.fit(inputs_train, expected_output_train.values.ravel())

#Results!
accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))

#Save the machine
joblib.dump(rf, "forest_classifier2", compress=9)

#Load the machine
rf = joblib.load('forest_classifier2')

#Load test data and predict
test_data = pd.read_csv('../input/test.csv')
predict = rf.predict(test_data[reduced_inputs])
test_data['Cover_Type'] = predict

output = test_data[['Id','Cover_Type']]

output.to_csv('output_85.csv', sep=',', index=None)            




