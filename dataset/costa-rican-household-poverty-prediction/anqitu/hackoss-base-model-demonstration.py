# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# 1. Import libraries ==========================================================
pd.set_option("display.max_columns",150)

import warnings
warnings.filterwarnings('ignore')


# 2. Configure path ============================================================
train_data_path = '../input/train.csv'
test_data_path = '../input/test.csv'


# 3. Import data ===============================================================
train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)
target = train['Target']
id = test['Id']
train = train.drop(columns = ['Target'])

# 4. Clean data ===============================================================
def clean_data(train):
    train.drop(columns = ['tamhog', 'hogar_total', 'Id', 'idhogar'], inplace = True)

    # set rental as 0 as most na owned the house
    train['v2a1'] = (np.log(train['v2a1'] + 1)).fillna(0)

    # fill 0 for those with na tablets
    train['v18q1'] = train['v18q1'].fillna(0)
    train.drop(columns = ['v18q'], inplace = True)

    # fille 0 for those with na rez_esc and meaneduc
    train['rez_esc'] = train['rez_esc'].fillna(0).astype(int)
    train['meaneduc'] = train['meaneduc'].fillna(0).astype(float)
    train['SQBmeaned'] = train['SQBmeaned'].fillna(0).astype(float)

    # calculate our own dependency ratio to avoid infinite number
    train['dependency'] = (train['hogar_nin'] + train['hogar_mayor']) / (train['hhsize']).astype(float)

    # replace yes and no with 1 and 0
    train['edjefe'] = train['edjefa'].replace('yes', '1').replace('no', '0').astype(int)
    train['edjefa'] = train['edjefa'].replace('yes', '1').replace('no', '0').astype(int)

    return train

train = clean_data(train)
test = clean_data(test)



# 5. Train data ===============================================================
# split dataset
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.3, random_state = 2018, stratify = target)

from sklearn.metrics import f1_score
def evaluate_prediction(model):
    train_y_predict = model.predict(train_X)
    print('Train score: ' + str(f1_score(train_y, train_y_predict, average='macro')))
    val_y_predict = model.predict(val_X)
    print('Val score: ' + str(f1_score(val_y, val_y_predict, average='macro')))

# # A. Logistic Model
# from sklearn.linear_model import LogisticRegression # target is categorical eg. income level
# model = LogisticRegression()
# model.fit(train_X, train_y)
# evaluate_prediction(model)

# # B. DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier # target is categorical eg. income level
# model = DecisionTreeClassifier(random_state = 2018)
# model.fit(train_X, train_y)
# evaluate_prediction(model)


# # C. RandomForest
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state = 2018)
# model.fit(train_X, train_y)
# evaluate_prediction(model)
# 
# # D. ExtraTreesClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier(random_state = 2018)
# model.fit(train_X, train_y)
# evaluate_prediction(model)
# 
# # E. GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# model = GradientBoostingClassifier(random_state = 2018)
# model.fit(train_X, train_y)
# evaluate_prediction(model)

# # F. KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier()
# model.fit(train_X, train_y)
# evaluate_prediction(model)

# final model -> ExtraTreesClassifier
# extract feature
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(random_state = 2018)
model.fit(train_X, train_y)
evaluate_prediction(model)

from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(model, prefit=True)
train_X_ = selector.transform(train_X)
val_X_ = selector.transform(val_X)

model.fit(train_X_, train_y)
print('Train score: ' + str(f1_score(train_y, model.predict(train_X_), average='macro')))
print('Train score: ' + str(f1_score(val_y, model.predict(val_X_), average='macro')))

# train on entire set
model.fit(train, target)
print('Train score: ' + str(f1_score(target, model.predict(train), average='macro')))
selector = SelectFromModel(model, prefit=True)
train = selector.transform(train)
test = selector.transform(test)

model.fit(train, target)
print('Train score: ' + str(f1_score(target, model.predict(train), average='macro')))
test_y = model.predict(test)
submission = pd.DataFrame(data = {'Id': id, 'Target': test_y})
submission.to_csv('submission_v1.csv', index = False)
