import numpy as np
import pandas as pd
import random
from sklearn.svm import OneClassSVM
import xgboost as xgb
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier

# Get Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train = train.drop(['QuoteNumber'], axis=1)

# Parse date
train['Year']  = train['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
train['Week']  = train['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test['Year']  = test['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test['Week']  = test['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

train.drop(['Original_Quote_Date'], axis=1,inplace=True)
test.drop(['Original_Quote_Date'], axis=1,inplace=True)

# fill NA
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

# label strings
for f in train.columns:
    if train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train[f].values) + list(test[f].values)))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# sample data
rows = random.sample(list(train.index), int(0.2*train.index.size))
train_sample = train.ix[rows]
train = train.drop(rows)

# Prepare for training
X_train = train.drop("QuoteConversion_Flag",axis=1)
Y_train = train["QuoteConversion_Flag"]
X_test  = test.drop("QuoteNumber",axis=1).copy()

# Remove ouutliers


print('ok')

# Prepare submission file
submission = pd.DataFrame()
submission["QuoteNumber"] = test["QuoteNumber"]

# random forest - works pretty good, ~ 0.82 without bagger.
# Acc score of 0.916 with bags
#rees = BaggingClassifier(ExtraTreesClassifier())
trees = ExtraTreesClassifier()
trees.fit(X_train,Y_train)
Y_pred = trees.predict(X_test)

# GB - acc is 0.92
#tryer = GradientBoostingClassifier()
#tryer.fit(X_train,Y_train)
#Y_pred = tryer.predict(X_test)

submission["QuoteConversion_Flag"] = Y_pred
submission.to_csv('forest_homesite.csv', index=False)

# Check Accuraacy 
X_check = train_sample.drop("QuoteConversion_Flag",axis=1)
Y_check = train_sample["QuoteConversion_Flag"]
check = trees.predict(X_check)
print("yo the accuracy is")
print(accuracy_score(Y_check,check))

#params = {"objective": "binary:logistic"}
#T_train_xgb = xgb.DMatrix(X_train, Y_train)
#X_test_xgb  = xgb.DMatrix(X_test)

#gbm = xgb.train(params, T_train_xgb, 20)
#Y_pred = gbm.predict(X_test_xgb)
#submission["QuoteConversion_Flag"] = Y_pred

#submission.to_csv('xgb_homesite.csv', index=False)