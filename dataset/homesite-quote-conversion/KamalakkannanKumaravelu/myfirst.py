import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gc

#Read Input Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Extrate Date Details
train['Original_Quote_Date_Typed'] = pd.to_datetime(train.Original_Quote_Date)
train['month'] = train.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%m'))
train['day_of_week'] = train.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%w'))

test['Original_Quote_Date_Typed'] = pd.to_datetime(test.Original_Quote_Date)
test['month'] = test.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%m'))
test['day_of_week'] = test.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%w'))

#Drop Quote Date
train.drop(['Original_Quote_Date'],axis=1,inplace=True)
test.drop(['Original_Quote_Date'],axis=1,inplace=True)

#Fill Nan Values
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

#Label Nominal Values (Non Numeric - which is dtype=Object)
for f in train.columns:
    if train[f].dtype=='object':     
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train[f].values) + list(test[f].values)))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

#Prepare Training & Test Data for Classifier         
Y_train = train["QuoteConversion_Flag"]
X_train = train.drop(["QuoteConversion_Flag","QuoteNumber","Original_Quote_Date_Typed"],axis=1)
X_test  = test.drop(["QuoteNumber","Original_Quote_Date_Typed"],axis=1).copy()

#Run RandomForrestClassifier
rf = RandomForestClassifier(n_estimators=500,n_jobs = -1)
rf.fit(X_train, Y_train)
#rf.fit(X.values, Y.values)

#Predict the Output
Y_test=rf.predict(X_test)

#Create Submission
submission = pd.DataFrame()
submission["QuoteNumber"]          = test["QuoteNumber"]
submission["QuoteConversion_Flag"] = Y_test
submission.to_csv('homesite_submission.csv', index=False)


