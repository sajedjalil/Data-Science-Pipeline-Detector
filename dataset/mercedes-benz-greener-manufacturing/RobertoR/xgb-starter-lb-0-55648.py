

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, ensemble

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



#Categorical to
categorical = ["X0",  "X1",  "X2", "X3", "X4",  "X5", "X6", "X8"]
for f in categorical:
        if train[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))


#thanks to Anokas

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)
        
   
    
usable_columns = list(set(train.columns) - set([ 'y']))

y_train = train['y'].values
id_test = test['ID'].values

x_train = train[usable_columns]
x_test = test[usable_columns]

   
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(x_test)




params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4


watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=10)

p_test = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = p_test
sub.to_csv('xgb.csv', index=False)




