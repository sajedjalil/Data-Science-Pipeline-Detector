import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

seed = 666

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
#train['yearday'] = train['Date'].apply(lambda x: int(x.strftime('%j')))

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek
#test['yearday'] = test['Date'].apply(lambda x: int(x.strftime('%j')))

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

#train['Custom_pers_fiel_10a_na'] = train['PersonalField10A'].apply(lambda x: 0 if(x == -1) else 1)
#test['Custom_pers_fiel_10a_na']  =  test['PersonalField10A'].apply(lambda x: 0 if(x == -1) else 1)

#train = train.drop(['GeographicField10A', 'PropertyField6'], axis=1)
#test  =  test.drop(['GeographicField10A', 'PropertyField6'], axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# 25 -> 95657
# 25 + pf10a na -> 95692
# 25 + day of year -> 95692
# 25 - (GeographicField10A + PropertyField6) -> 95677
# 25 + nb train quotes day -> 95666
clf = xgb.XGBClassifier(n_estimators=25,
                        nthread=-1,
                        max_depth=8,
                        learning_rate=0.03,
                        silent=True,
                        subsample=0.7,
                        colsample_bytree=0.7)
                        
xgb_model = clf.fit(train, y, eval_metric="auc")

preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('xgb_benchmark_2.csv', index=False)