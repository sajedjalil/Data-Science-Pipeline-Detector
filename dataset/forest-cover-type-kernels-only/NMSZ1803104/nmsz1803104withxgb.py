import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
train = pd.read_csv('../input/train.csv',  encoding='gbk')
test = pd.read_csv('../input/test.csv', encoding='gbk')
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
y = train["Cover_Type"]
train.drop(["Cover_Type"], inplace=True, axis=1)
x = train
feed = 42
test_size = 0.25
#X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=feed)
ss=StandardScaler()
#X_train=ss.fit_transform(X_train)
#X_test=ss.transform(X_test)
X_train = ss.fit_transform(x)
y_train = y
model = XGBClassifier(learning_rate=0.3, n_estimators=350, max_depth=9, objective='multi:softmax', num_class=7)
model.fit(X_train,y_train)
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print ('Accuracy:',accuracy)
X_sub=test
X_sub=ss.transform(X_sub)
pred=model.predict(X_sub)
pred = pd.Series(pred, name='Cover_Type')
sub = pd.read_csv('../input/sample_submission.csv')
sub.drop('Cover_Type', axis=1, inplace=True)
sub = pd.concat([sub, pred], axis=1)
sub.to_csv('./NMSZ1803104.csv', index=False)
print('Saved.')