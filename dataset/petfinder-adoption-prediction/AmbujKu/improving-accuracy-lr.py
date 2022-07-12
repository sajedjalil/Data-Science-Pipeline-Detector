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

# Importing the Training Dataset

df = pd.read_csv('../input/train/train.csv')
df.info()

Y = df["AdoptionSpeed"]

from sklearn.feature_extraction.text import TfidfVectorizer

train_desc = df.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3, max_features=10000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')

tfv.fit(list(train_desc))

desc = tfv.transform(train_desc)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=120)
svd.fit(desc)
desc = svd.transform(desc)

desc = pd.DataFrame(desc, columns=['svd_{}'.format(i) for i in range(120)])

df = pd.concat((df, desc), axis=1)

df = df.drop(["Name", "RescuerID", "VideoAmt", "Description", "PetID", "PhotoAmt", "AdoptionSpeed"], axis = 1)

X = df


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

Y_test = Y_test.tolist()

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(X_train, Y_train)


Y_pred = rf.predict(X_test)

#convert to list Y_pred and tointeger

Y_pred = Y_pred.tolist()

for i in range(len(Y_pred)):
	Y_pred[i] = int(Y_pred[i])


from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(Y_test, Y_pred)

sum = 0
for i in range(len(cm_rf)):
	sum = sum + cm_rf[i][i]

acc_rf = sum/len(Y_test)

print("random_forest_acc", acc_rf)


# XGBoost

from xgboost import XGBClassifier
#est = [100,200,300,400, 500]
#acc_lst =[]
#for j in est:
xgb = XGBClassifier(learning_rate=0.2,n_estimators=400, booster='dart')
xgb.fit(X_train, Y_train)

Y_pred_xgb = xgb.predict(X_test)
Y_pred_xgb = Y_pred_xgb.tolist()
cm_xgb = confusion_matrix(Y_test, Y_pred_xgb)

sum = 0
for i in range(len(cm_xgb)):
    sum = sum + cm_xgb[i][i]

acc_xgb = sum/len(Y_test)
#    acc_lst.append(acc_xgb)

print ("XGB accuracy", acc_xgb)


# LightGBM
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.06,
    num_leaves=30,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
    )

clf.fit(X_train, Y_train)

Y_pred_lgbm = clf.predict(X_test)

Y_pred_lgbm = Y_pred_lgbm.tolist()

cm_lgbm = confusion_matrix(Y_test, Y_pred_lgbm)
sum = 0
for i in range(len(cm_lgbm)):
    sum = sum + cm_lgbm[i][i]
acc_lgbm = sum/len(Y_test)

print ("LGBM accuracy", acc_lgbm)

df_test = pd.read_csv('../input/test/test.csv')

# adding Description column:

test_desc = df_test.Description.fillna("none").values

tfv.fit(list(test_desc))

desc = tfv.transform(test_desc)

svd.fit(desc)
desc = svd.transform(desc)

desc = pd.DataFrame(desc, columns=['svd_{}'.format(i) for i in range(120)])

df_test = pd.concat((df_test, desc), axis=1)


df_test = df_test.drop(["Name", "RescuerID", "VideoAmt", "Description", "PetID", "PhotoAmt"], axis = 1)

test_data = df_test


test_pred = clf.predict(test_data)
test_pred = test_pred.tolist()

submit = pd.DataFrame({"AdoptionSpeed":test_pred})

# takeout PetID from original test set

df_test = pd.read_csv('../input/test/test.csv')

submit.insert(loc=0, column='PetID', value=df_test["PetID"])

submit.to_csv('submission.csv', index=False)


