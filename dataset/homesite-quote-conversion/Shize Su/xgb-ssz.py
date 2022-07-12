import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

seed = 68# 666 #8

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)

# Lets play with some dates
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek


test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

print('start training')
    
    
clf = xgb.XGBClassifier(n_estimators=1400,
                        nthread=-1,
                        max_depth=8,
                        learning_rate=0.022,
                        silent=True, #True,
                        subsample=0.82,
                        colsample_bytree=0.8)
                        
xgb_model = clf.fit(train, y, eval_metric="auc")

preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('xgb_benchmark_shize_3.csv', index=False)