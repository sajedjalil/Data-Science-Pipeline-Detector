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

# Import libraries
from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

train = pd.read_csv('../input/train.csv', encoding='ISO-8859-1')
test = pd.read_csv('../input/test.csv', encoding='ISO-8859-1')

train.head()


house = train[['idhogar',
                'male',
                'female' ,
                'estadocivil1',
                'estadocivil2',
                'estadocivil3',
                'estadocivil4',
                'estadocivil5',
                'estadocivil6',
                'estadocivil7',
                'parentesco2',
                'parentesco3',
                'parentesco4',
                'parentesco5',
                'parentesco6',
                'parentesco7',
                'parentesco8',
                'parentesco9',
                'parentesco10',
                'parentesco11',
                'parentesco12',
                'instlevel1',
                'instlevel2',
                'instlevel3',
                'instlevel4',
                'instlevel5',
                'instlevel6',
                'instlevel7',
                'instlevel8',
                'instlevel9'

                ]]
house.head()
house_agg =  house.groupby('idhogar').agg(sum)
house_agg.info()


house_age = train[['idhogar',
                'age'

                ]]
house_age.head()
house_agg_age =  house_age.groupby('idhogar').agg(pd.np.mean)
house_agg_age.info()

house_merge = pd.merge(house_agg, house_agg_age, on='idhogar', how='left')
house_merge.info()
house_merge = house_merge.add_prefix('h_')


train = train.loc[train['parentesco1'] == 1]
train.info()
del train['parentesco1']


train = pd.merge(train, house_merge, on='idhogar', how='left')
train.info()
train.head()


target = pd.DataFrame(train.Target)
target.info()
del train['Target']
train.head()

del train['Id']
del train['idhogar']
train.info()

one_hot_1 = train.loc[:, train.dtypes == object]
one_hot_1.head()

train = train.replace('yes',1)
train = train.replace('no',0)


# For Min/Max scaling
scaler = MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns = train.columns)
train.head()

train = pd.concat([train, target], axis=1)
train.head()

train['Target'].value_counts()

train.isnull().values.any()
train.isnull().sum(axis = 0).sort_values(ascending = False)


del train['rez_esc']
del train['v18q1']
del train['v2a1']

train = train.dropna()
train.head()


#Separate target & attributes
X, y = train.iloc[:,:-1],train.iloc[:,-1]
y.head()
y = y-1

X = X.apply(pd.to_numeric) # convert all columns of DataFrame


# Create matrix used for xgboost
data_dmatrix = xgb.DMatrix(data=X,label=y)


# Create train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


xg_reg = xgb.XGBRegressor(objective ='multi:softmax', num_class = 4, colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

# Fit model
xg_reg.fit(X_train,y_train)

# Predict
preds = xg_reg.predict(X_test)

# Look at RMSE
f1 = f1_score(y_test, preds, average='weighted')

print("F1 Score: %f" % (f1))


test.info()



house = test[['idhogar',
                'male',
                'female' ,
                'estadocivil1',
                'estadocivil2',
                'estadocivil3',
                'estadocivil4',
                'estadocivil5',
                'estadocivil6',
                'estadocivil7',
                'parentesco2',
                'parentesco3',
                'parentesco4',
                'parentesco5',
                'parentesco6',
                'parentesco7',
                'parentesco8',
                'parentesco9',
                'parentesco10',
                'parentesco11',
                'parentesco12',
                'instlevel1',
                'instlevel2',
                'instlevel3',
                'instlevel4',
                'instlevel5',
                'instlevel6',
                'instlevel7',
                'instlevel8',
                'instlevel9'

                ]]
house.head()
house_agg =  house.groupby('idhogar').agg(sum)
house_agg.info()


house_age = test[['idhogar',
                'age'

                ]]
house_age.head()
house_agg_age =  house_age.groupby('idhogar').agg(pd.np.mean)
house_agg_age.info()

house_merge = pd.merge(house_agg, house_agg_age, on='idhogar', how='left')
house_merge.info()
house_merge = house_merge.add_prefix('h_')


test = test.loc[test['parentesco1'] == 1]
test.info()
del test['parentesco1']


test = pd.merge(test, house_merge, on='idhogar', how='left')
test.info()
test.head()


ids = pd.DataFrame(test.Id)
ids.info()
del test['Id']
test.head()


del test['idhogar']
test.info()



test = test.replace('yes',1)
test = test.replace('no',0)


# For Min/Max scaling
scaler = MinMaxScaler()
test = pd.DataFrame(scaler.fit_transform(test), columns = test.columns)
test.head()



test.isnull().values.any()
test.isnull().sum(axis = 0).sort_values(ascending = False)


del test['rez_esc']
del test['v18q1']
del test['v2a1']

test = test.fillna(0)


x_test = test.apply(pd.to_numeric) # convert all columns of DataFrame




# Predict
preds = xg_reg.predict(x_test)

preds = pd.DataFrame(preds)
preds = preds + 1

final = pd.concat([ids, preds], axis=1)
final.columns = ['Id', 'Target']
final['Target'].value_counts()

final.to_csv('prediction.csv', header = True, index = False)
