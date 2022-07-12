# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
base_path = '/kaggle/input/siim-isic-melanoma-classification/'
test_image_path = base_path + 'jpeg/test/'
train_image_path = base_path + 'jpeg/train/'

submission_csv = base_path + 'sample_submission.csv'
test_csv = base_path + 'test.csv'
train_csv = base_path + 'train.csv'

# %% [code]
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
submission = pd.read_csv(submission_csv)

# %% [code]
# Mark male female as 1/0
# There are only two values, there are some missing values, which should be filled with mode
train['sex'] = train['sex'].replace({ 'female': 0, 'male': 1 })
test['sex'] = test['sex'].replace({ 'female': 0, 'male': 1 })
train['sex'].fillna(train['sex'].mode()[0], inplace=True)

# Remove benign malignant, it's the same as target
train.drop(['benign_malignant'], inplace=True, axis=1)

# Add dummies for anatom_site_general_challenge
# Fill the nan's with a new dummy
def add_dummies(dataset, column, short_name):
    dummy = pd.get_dummies(
        dataset[column], 
        drop_first=True, 
        prefix=short_name, 
        prefix_sep='_',
        dummy_na=True
    )
    merged = pd.concat([dataset, dummy], axis=1)
    return merged.drop([column], axis=1)

train = add_dummies(train, 'anatom_site_general_challenge', 'anatom')
test = add_dummies(test, 'anatom_site_general_challenge', 'anatom')

# Diagnosis is only in train, removing it
train.drop(['diagnosis'], inplace=True, axis=1)

# Age has some missing values, fill with median
train['age_approx'].fillna(train['age_approx'].median(), inplace=True)

# %% [code]
# Check how many times are their images taken
train['image_count'] = train['patient_id'].map(train.groupby(['patient_id'])['image_name'].count())
test['image_count'] = test['patient_id'].map(test.groupby(['patient_id'])['image_name'].count())

# %% [code]
# no patients overlap between train and test
list(set(test['patient_id'].value_counts().index) & set(train['patient_id'].value_counts().index))

# %% [code]
train.head()

# %% [code]
train.head()

# %% [code]
import xgboost
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

# %% [code]
X = train.drop(['image_name', 'patient_id', 'target'], axis=1)
y = train['target']
X_final = test.drop(['image_name', 'patient_id'], axis=1)

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

cv = StratifiedKFold(5, shuffle=True, random_state=42)

# %% [code]
xgb = xgboost.XGBClassifier(
    n_jobs=-1,
    random_state=42
)

score = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='roc_auc', verbose=1)
print(score.mean(), score.std())

# %% [code]
xgb.fit(X_train, y_train)
pred_score = xgb.predict_proba(X_test)
pred_score

# %% [code]
roc_auc_score(y_test, pred_score[:, 1])

# %% [code]
submission['target'] = xgb.predict_proba(X_final)[:, 1]
submission.to_csv('baseline_submission.csv', index=False, header=True)

# %% [code]
# https://www.kaggle.com/datafan07/analysis-of-melanoma-metadata-and-effnet-ensemble