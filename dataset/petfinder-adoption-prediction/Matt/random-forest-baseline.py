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

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.model_selection import GridSearchCV

import xgboost as xgb


train = pd.read_csv("../input/pet-data/pets_train.csv")
test = pd.read_csv("../input/pet-data/pets_test.csv")

train.Name.fillna('None', inplace=True)

train.Description.fillna('None', inplace=True)

test.Name.fillna('None', inplace=True)

test.Description.fillna('None', inplace=True)

labels = train.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1)

test_labels = test.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)

labels = pd.get_dummies(labels, columns = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',

                                 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',

                                 'State', 'Type', 'Group'

                                ])

test_labels = pd.get_dummies(test_labels, columns = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',

                                 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',

                                 'State', 'Type', 'Group'

                                ])

labels.columns

test_labels.columns

diff_columns = set(labels.columns).difference(set(test_labels.columns))

for i in diff_columns:

    test_labels[i] = test_labels.apply(lambda _: 0, axis=1)

diff_columns2 = set(test_labels.columns).difference(set(labels.columns))

for i in diff_columns2:

    labels[i] = labels.apply(lambda _: 0, axis=1)

target = train['AdoptionSpeed']

labels['NameLength'] = train['Name'].map(lambda x: 0 if x == 'None' else len(x)).astype('int')
labels['DescLength'] = train['Description'].map(lambda x: len(x)).astype('int')
labels['Cute'] = train['Description'].map(lambda x: 1 if 'CUTE' in x.upper() else 0).astype('int')
test_labels['Cute'] = test['Description'].map(lambda x: 1 if 'CUTE' in x.upper() else 0).astype('int')
test_labels['NameLength'] = test['Name'].map(lambda x: 0 if x == 'None' else len(x)).astype('int')
test_labels['DescLength'] = test['Description'].map(lambda x: len(x)).astype('int')





X_train, X_test, y_train, y_test = train_test_split(labels, target, test_size=0.2)

clf = xgb.XGBClassifier()

labels = labels.drop('Unnamed: 0.1', axis=1)
clf.fit(labels, target)

test_labels = test_labels[labels.columns]

pred = pd.DataFrame()
pred['PetID'] = test['PetID']
pred['AdoptionSpeed'] = clf.predict(test_labels)
pred.set_index('PetID').to_csv("submission.csv", index=True)

