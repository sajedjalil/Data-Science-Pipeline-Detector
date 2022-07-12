import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing 

def feature_processing(X):
    le = preprocessing.LabelEncoder()
    X.fillna(-1, inplace = True)
    x_cat = X.select_dtypes(include = ['object'])
    for cols in x_cat:
        le.fit(list(X[cols].values))
        X[cols] = le.transform(list(X[cols].values))
    x_values = X.select_dtypes(exclude = ['object'])
    x_values = x_values.fillna(-1)
    x_mean = x_values.mean(axis = 0)
    x_var = x_values.var(axis = 0)
    x_norm = (x_values - x_mean)/(x_values.std(axis=0))
    return x_norm

train = pd.read_csv("../input/train.csv",thousands = ',')
test = pd.read_csv("../input/test.csv")

y_field = 'QuoteConversion_Flag'
irrelevant_fields =['QuoteNumber', y_field, 'Original_Quote_Date']
y = train[y_field].values

train.drop(irrelevant_fields,axis = 1, inplace = True)
test.drop(['QuoteNumber','Original_Quote_Date'],axis = 1, inplace = True)

x = feature_processing(train)
x_test = feature_processing(test)

clf = xgb.XGBClassifier(
n_estimators=25,
nthread=-1,
max_depth=10,
learning_rate=0.025,
silent=False,
subsample=0.8,
colsample_bytree=0.8
)
xgb_model = clf.fit(x, y, eval_metric="auc")

preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('xgb_benchmark.csv', index=False)
