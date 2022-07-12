# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
import gzip
import zipfile
from sklearn import linear_model
import xgboost as xgb
from sklearn.decomposition import PCA

number = preprocessing.LabelEncoder()
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'], index_col=False)
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col=False)

train = train.drop(['Descript', 'Resolution', 'Address'], axis = 1)
test = test.drop(['Address'], axis = 1)

def convert(data):
    number = preprocessing.LabelEncoder()
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['DayOfWeek'] = data['Dates'].dt.dayofweek
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
    data=data.fillna(0)
    return data

train=convert(train)
test=convert(test)
enc = LabelEncoder()
train['PdDistrict'] = enc.fit_transform(train['PdDistrict'])
category_encoder = LabelEncoder()
category_encoder.fit(train['Category'])
train['CategoryEncoded'] = category_encoder.transform(train['Category'])
enc = LabelEncoder()
test['PdDistrict'] = enc.fit_transform(test['PdDistrict'])


x_cols = list(train.columns[2:11].values)
x_cols.remove('Minute')


#pca = PCA(n_components=None)
#train[x_cols] = pca.fit_transform(train[x_cols])

train, validation = train_test_split(train, test_size = 0.33)

'''x = train[x_cols].as_matrix()
y = train['CategoryEncoded']
valid_x  = validation[x_cols].as_matrix()
valid_y = validation['CategoryEncoded']
gbm = xgb.XGBClassifier(objective='multi:softprob', max_depth=6, max_delta_step=1, learning_rate=1.0).fit(x, y, verbose=True, eval_metric='mlogloss')
valid_pred = gbm.predict_proba(valid_x)'''
x = train[x_cols].values
y = train['CategoryEncoded'].values
arr = np.random.permutation(x.shape[0])
x = x[arr,]
y = y[arr]
valid_x  = validation[x_cols].values
valid_y = validation['CategoryEncoded'].values

c = 0
for i in valid_y:
    if i==33:
        c=c+1
        
if c>0:
    print("Yes")
else:
    print("No")

params = {}
params["objective"] = "multi:softprob"
params["eta"] = 1.0
params["max_delta_step"] = 1
#params["booster"] = 'gbtree',
#params["min_child_weight"] = 10
#params["subsample"] = 0.7
params["scale_pos_weight"] = 0.9
params["silent"] = 1
params["max_depth"] = 8
params["eval_metric"] = 'mlogloss'
params["num_class"] = 39
params["early.stop.round"] = 1

plst = list(params.items())

#xgtrain = xgb.DMatrix(x,label = y)
#xgtest = xgb.DMatrix(valid_x)

num_rounds = 15

#model = xgb.train(plst, xgtrain, num_rounds)
#prediction = model.predict(xgtest)

#score1 = metrics.log_loss(valid_y, prediction)
#print(score1)

del train
del test

knn = KNeighborsClassifier(n_neighbors=150, weights='distance')
knn.fit(x, y)
prediction_knn = knn.predict_proba(valid_x)
score2 = metrics.log_loss(valid_y, prediction_knn)
print(score2)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.