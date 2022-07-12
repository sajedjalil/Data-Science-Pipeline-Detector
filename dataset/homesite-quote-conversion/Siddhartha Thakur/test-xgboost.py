import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)

print(train.head(n=5))
print(test.head(n=5))
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag','PersonalField7','PersonalField8','PersonalField13','PersonalField34','PersonalField35','PersonalField36','PersonalField37','PersonalField38','PersonalField39','PersonalField40','PersonalField41','PersonalField42','PersonalField43','PersonalField49','PersonalField50','PersonalField51','PersonalField52','PersonalField53','PersonalField54','PersonalField55','PersonalField56','PersonalField57','PersonalField58','PersonalField59','PersonalField60','PersonalField61','PersonalField62','PersonalField63','PersonalField64','PersonalField65','PersonalField66','PersonalField67','PersonalField68','PersonalField69','PersonalField70','PersonalField71','PersonalField72','PersonalField73','PropertyField5','PropertyField9','PropertyField20','PropertyField22','PropertyField10','PropertyField6', 'GeographicField10A','GeographicField10B'], axis=1)
test = test.drop(['QuoteNumber','PersonalField7','PersonalField8','PersonalField13','PersonalField34','PersonalField35','PersonalField36','PersonalField37','PersonalField38','PersonalField39','PersonalField40','PersonalField41','PersonalField42','PersonalField43','PersonalField49','PersonalField50','PersonalField51','PersonalField52','PersonalField53','PersonalField54','PersonalField55','PersonalField56','PersonalField57','PersonalField58','PersonalField59','PersonalField60','PersonalField61','PersonalField62','PersonalField63','PersonalField64','PersonalField65','PersonalField66','PersonalField67','PersonalField68','PersonalField69','PersonalField70','PersonalField71','PersonalField72','PersonalField73','PropertyField5','PropertyField9','PropertyField20','PropertyField22','PropertyField10','PropertyField6', 'GeographicField10A','GeographicField10B'], axis=1)

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

clf = xgb.XGBClassifier(n_estimators=25,
                        nthread=-1,
                        max_depth=10,
                        learning_rate=0.025,
                        silent=True,
                        subsample=0.8,
                        colsample_bytree=0.8)
                        
xgb_model = clf.fit(train, y, eval_metric="auc")

preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('xgb_benchmark.csv', index=False)