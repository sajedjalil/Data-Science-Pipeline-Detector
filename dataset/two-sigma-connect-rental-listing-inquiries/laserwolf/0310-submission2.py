

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def scorer(estimator, X, y):
    return -log_loss(y, estimator.predict_proba(X))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df=pd.read_json('../input/train.json')
df['priceperbed']=(df['price'].clip(upper=7000)/df['bedrooms'].clip(lower=1))
df['created']=df['created'].astype(np.datetime64)
df['created_day']=np.array(df.created.values, dtype='datetime64[D]').astype(np.float32)%7
df['created_week']=np.array(df.created.values, dtype='datetime64[W]').astype(np.float32)
df['created_hour']=np.array(df.created.values, dtype='datetime64[h]').astype(np.float32)%24
df['desc_count']=df.description.apply(lambda x: len(x.split())).clip(upper=150)
df['features_count']=df.features.apply(lambda x: len(x))
df['photos_count']=df.photos.apply(lambda x: len(x))

df['features']=df['features'].apply(lambda x: list(map(str.lower, x)))
for feature in ['hardwood floors',  'laundry in building', 'cats_allowed', 'no fee']:
    df[feature]=df['features'].apply(lambda x: feature in x)

df['medium']=df['interest_level']=='medium'
df['low']=df['interest_level']=='low'
df['high']=df['interest_level']=='high'

df_test=pd.read_json('../input/test.json')
df_test['priceperbed']=(df_test['price'].clip(upper=7000)/df_test['bedrooms'].clip(lower=1))
df_test['created']=df_test['created'].astype(np.datetime64)
df_test['created_day']=np.array(df_test.created.values, dtype='datetime64[D]').astype(np.float32)%7
df_test['created_week']=np.array(df_test.created.values, dtype='datetime64[W]').astype(np.float32)
df_test['created_hour']=np.array(df_test.created.values, dtype='datetime64[h]').astype(np.float32)%24
df_test['desc_count']=df_test.description.apply(lambda x: len(x.split())).clip(upper=150)
df_test['features_count']=df_test.features.apply(lambda x: len(x))
df_test['photos_count']=df_test.photos.apply(lambda x: len(x))

df_test['features']=df_test['features'].apply(lambda x: list(map(str.lower, x)))
for feature in ['hardwood floors',  'laundry in building', 'cats_allowed', 'no fee']:
    df_test[feature]=df_test['features'].apply(lambda x: feature in x)

cols=['price', 'bathrooms', 'bedrooms', 'priceperbed', 'created_hour', 
          'desc_count', 'photos_count', 'features_count', 'hardwood floors',
          'laundry in building', 'cats_allowed', 'no fee']
clf=ExtraTreesClassifier(max_depth=3, n_estimators=100, min_samples_split=10, random_state=0)
clf.fit(df[cols].values, df['interest_level'])

print ("almost done")
y_pred=clf.predict_proba(df_test[cols])
df_y_pred=pd.DataFrame(y_pred, index=df_test['listing_id'], 
                       columns=['high', 'low', 'medium'])
df_y_pred.to_csv("submission.0310.6.csv.gz", compression='gzip')
print(check_output(["ls"]).decode("utf8"))
print ("done")
