# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# LOAD TRAIN AND TEST FILES
train_file = pd.read_csv("../input/train.csv")
test_file = pd.read_csv("../input/test.csv")

# DATA OVERVIEW
train_file.head()
# FEATURE ENGINEERING: select and transform input attributes

#Name and OutcomeSubtype are deleted
train_file.drop(["OutcomeSubtype", "Color"], axis=1, inplace=True)
#test_file.drop(["Name"], axis=1, inplace=True)

#DateTime will be converted to categorical Year, Month and Day of the Week
from datetime import datetime
def convert_date(dt):
    d = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    return d.year, d.month, d.isoweekday()

train_file["Year"], train_file["Month"], train_file["WeekDay"] = zip(*train_file["DateTime"].map(convert_date))
test_file["Year"], test_file["Month"], test_file["WeekDay"] = zip(*test_file["DateTime"].map(convert_date))
train_file["HasName"] = len(train_file["Name"])>0
test_file["HasName"] = len(test_file["Name"])>0
train_file.drop(["Name"], axis=1, inplace=True)
test_file.drop(["Name"], axis=1, inplace=True)
train_file.drop(["DateTime"], axis=1, inplace=True)
test_file.drop(["DateTime"], axis=1, inplace=True)

#Separating IDs
train_id = train_file[["AnimalID"]]
test_id = test_file[["ID"]]
train_file.drop(["AnimalID"], axis=1, inplace=True)
test_file.drop(["ID"], axis=1, inplace=True)

#Separating target variable
train_outcome = train_file["OutcomeType"]
train_file.drop(["OutcomeType"], axis=1, inplace=True)


#pd.options.mode.chained_assignment = None  # default='warn
#Encode the categorical data, with the complete set (train and test)
train_file["train"] = 1
test_file["train"] = 0
conjunto = pd.concat([train_file, test_file])
conjunto_encoded = pd.get_dummies(conjunto, columns=conjunto.columns)
train = conjunto_encoded[conjunto_encoded["train_1"] == 1]
test = conjunto_encoded[conjunto_encoded["train_0"] == 1]
train.drop(["train_0","train_1"], axis=1, inplace=True)
test.drop(["train_0","train_1"], axis=1, inplace=True)
# SPLIT TRAIN DATASET 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, train_outcome, test_size=0.2)

#TRAIN A CLASSIFIER 
# logistic regression or randomforest from scikitlearn

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#from sklearn.ensemble import RandomForestClassifier
N_ESTIMATORS=100
#model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=2)

model.fit(X_train, y_train)
# GENERATE PREDICTIONS
y_pred_val = model.predict(X_val)
y_pred_val_prob = model.predict_proba(X_val)
mydataframe = pd.DataFrame(y_pred_val_prob)
mydataframe.columns =  ["Adoption","Died","Euthanasia","Return_to_owner","Transfer"]
mydataframe.head()
#EVALUATE THE MODEL/CLASSIFIER
from sklearn.metrics import classification_report, accuracy_score, log_loss

print(classification_report(y_val, y_pred_val))
print("Accuracy:",accuracy_score(y_val, y_pred_val))
print("log_loss:", log_loss(y_val, y_pred_val_prob))
#GENERATE SUBMISSION FILE

#Training a Logistic Regression or RandomForest with the complete training set

model = LogisticRegression(max_iter=200)
#model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=2)
model.fit(train, train_outcome)
y_pred = model.predict_proba(test)

results = pd.read_csv("../input/sample_submission.csv")

results['Adoption'], results['Died'], results['Euthanasia'], results['Return_to_owner'], results['Transfer'] = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3], y_pred[:,4]
results.to_csv("submission.csv", index=False)