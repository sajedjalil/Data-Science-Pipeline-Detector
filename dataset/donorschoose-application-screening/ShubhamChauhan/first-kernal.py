# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))

res = pd.read_csv("../input/resources.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Overview
train.head()
res.head()

""" Missing values """
train.isnull().sum()    #4
test.isnull().sum()     #2
res.isnull().sum()      #292


""" Combining test and train the data """
project = pd.concat([train, test], axis = 0)
#project = pd.merge(project, res, how = "left", on = "id")


""" OverView of the combined data set """
project.head()

project.isnull().sum()


""" Filling Missing Values """
project["teacher_prefix"].fillna(method = "ffill", inplace = True)

project["project_essay_3"].fillna(value = "", inplace = True)

project["project_essay_4"].fillna(value = "", inplace = True)


""" Combining the project essay 1, 2, 3, 4 """
project["essays"] = project["project_essay_1"] + project["project_essay_2"] + project["project_essay_3"] + project["project_essay_4"]
project["essays"][0]


""" Exploaration and Feature Engineering """
project['teacher_id'].nunique() #132133

project['teacher_prefix'].unique()

# Does Teacher and Dr has any kind of privelages?
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x = "project_is_approved", hue = "teacher_prefix",
               data = project)

""" School State wise projects """
sns.countplot(x = 'school_state', data = project)

"""Project Submitted Date Time """
project["project_submitted_datetime"] = pd.to_datetime(
        project["project_submitted_datetime"])

project["month"] = project["project_submitted_datetime"].dt.month
project["year"] = project["project_submitted_datetime"].dt.year

sns.countplot(x = "month", hue = "project_is_approved",
              data = project)
              
""" Converting cateorical into encoder"""
project["project_grade_category"].nunique() #4

project["project_subject_categories"].nunique() #51

project["project_subject_subcategories"].nunique() #416

project["project_title"].nunique() #416


features = ["project_grade_category", "project_subject_categories",
            "project_subject_subcategories", "school_state",
            "teacher_number_of_previously_posted_projects"]

target = "project_is_approved"

X = project[features][0:182080]
y = project[target][0:182080]

X_Test = project[project["project_is_approved"].isnull()][features]

dummies = pd.get_dummies(
        X["project_grade_category"])

le = LabelEncoder()

X["project_subject_categories"] = le.fit_transform( 
        X["project_subject_categories"])

X["project_subject_subcategories"] = le.fit_transform(
        X["project_subject_subcategories"])

X["school_state"] = le.fit_transform(
        X["school_state"])

del X["project_grade_category"]

X = pd.concat([X, dummies], axis = 1)

# XGboost model
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


