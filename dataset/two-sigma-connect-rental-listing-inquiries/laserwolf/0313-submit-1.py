# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df_train=pd.read_json("../input/train.json")
df_train['priceperbed']=(df_train['price'].clip(upper=7000)/df_train['bedrooms'].clip(lower=1))
df_train['created']=df_train['created'].astype(np.datetime64)
df_train['created_day']=np.array(df_train.created.values, dtype='datetime64[D]').astype(np.float32)%7
df_train['created_week']=np.array(df_train.created.values, dtype='datetime64[W]').astype(np.float32)
df_train['created_hour']=np.array(df_train.created.values, dtype='datetime64[h]').astype(np.float32)%24
df_train['desc_count']=df_train.description.apply(lambda x: len(x.split())).clip(upper=150)
df_train['features_count']=df_train.features.apply(lambda x: len(x))
df_train['photos_count']=df_train.photos.apply(lambda x: len(x))

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df_train['manager_id'].values))
df_train['manager_id'] = lbl.transform(list(df_train['manager_id'].values))

feature_list=['no fee', 'hardwood floors', 'laundry in building']
df_train['features']=df_train['features'].apply(lambda x: list(map(str.lower, x)))
for feature in feature_list:
        df_train[feature]=df_train['features'].apply(lambda x: feature in x)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
vectorizer.fit(df_train.description.values)

df_test=pd.read_json("../input/train.json")
df_test['priceperbed']=(df_test['price'].clip(upper=7000)/df_test['bedrooms'].clip(lower=1))
df_test['created']=df_test['created'].astype(np.datetime64)
df_test['created_day']=np.array(df_test.created.values, dtype='datetime64[D]').astype(np.float32)%7
df_test['created_week']=np.array(df_test.created.values, dtype='datetime64[W]').astype(np.float32)
df_test['created_hour']=np.array(df_test.created.values, dtype='datetime64[h]').astype(np.float32)%24
df_test['desc_count']=df_test.description.apply(lambda x: len(x.split())).clip(upper=150)
df_test['features_count']=df_test.features.apply(lambda x: len(x))
df_test['photos_count']=df_test.photos.apply(lambda x: len(x))

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df_test['manager_id'].values))
df_test['manager_id'] = lbl.transform(list(df_test['manager_id'].values))

feature_list=['no fee', 'hardwood floors', 'laundry in building']
df_test['features']=df_test['features'].apply(lambda x: list(map(str.lower, x)))
for feature in feature_list:
        df_test[feature]=df_test['features'].apply(lambda x: feature in x)

temp = pd.concat([df_train.manager_id,pd.get_dummies(df_train.interest_level)], axis = 1
                ).groupby('manager_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = df_train.groupby('manager_id').count().iloc[:,1]
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
unranked_managers_ixes = temp['count']<20
ranked_managers_ixes = ~unranked_managers_ixes
mean_values = temp.loc[ranked_managers_ixes, [
    'high_frac','low_frac', 'medium_frac','manager_skill']].mean()
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

df_train = df_train.merge(temp.reset_index(),how='left', on='manager_id')
df_test = df_test.merge(temp.reset_index(),how='left', on='manager_id')
new_manager_ixes = df_test['high_frac'].isnull()
df_test.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill'
                            ]] = mean_values.values

derived_cols = ['derived_'+str(i) for i in range(5)]
cols=['price', 'bathrooms', 'bedrooms', 'latitude', 'longitude', 'priceperbed','created_hour', 
      'desc_count', 'photos_count', 'features_count', 'no fee', 'hardwood floors', 
      'laundry in building', 'manager_skill']

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
X_train = svd.fit_transform(vectorizer.transform(df_train.description))
X_train=np.hstack([X_train, df_train[cols].values])

X_test = svd.transform(vectorizer.transform(df_test.description))
X_test=np.hstack([X_test, df_test[cols].values])
clf=ExtraTreesClassifier(max_depth=23, n_estimators=1000,
                             min_samples_split=10, random_state=0) 
clf.fit(X_train, df_train['interest_level'])
y_pred=clf.predict_proba(X_train)
score=log_loss(df_train['interest_level'].values, y_pred)
print(score)

y_pred=clf.predict_proba(X_test)
df_y_pred=pd.DataFrame(y_pred, index=df_test['listing_id'], 
                       columns=['high', 'low', 'medium'])
df_y_pred.to_csv("submission.0313.1.csv.gz", compression='gzip')
print(check_output(["ls"]).decode("utf8"))
print ("done")
