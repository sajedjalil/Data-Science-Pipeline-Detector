import pandas as pd


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

seed = 260681


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

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
        lbl.fit(list(train[f].unique()) + list(test[f].unique()))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


X = train.ix[:, 0:299]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


clf = xgb.XGBClassifier(n_estimators=140,
                        nthread=-1,
                        max_depth=10,
                        learning_rate=0.025,
                        silent=False,
                        subsample=0.8,
                        colsample_bytree=0.8)
clf.fit(X_train, y_train, eval_metric="auc",eval_set=[(X_test,y_test)],early_stopping_rounds=30)

p1 = clf.predict_proba(train)
p2 = clf.predict_proba(test)
np.save('xgb_80_train.npy',p1)
np.save('xgb_80_test.npy',p2)