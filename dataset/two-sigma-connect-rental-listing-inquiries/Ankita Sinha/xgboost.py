# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_json(open("../input/train.json", "r"))
test = pd.read_json(open("../input/test.json", "r"))
print(train.shape)
print(test.shape)
print(train.info())
df=pd.concat([train,test])
# feature engineering
ulimit = np.percentile(df.price.values, 99)
df['price'].ix[df['price']>ulimit] = ulimit
# price is right skewed so using log to create a gaussian pattern
df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
df["created_hour"] = df["created"].dt.hour
categorical = ["display_address", "manager_id", "building_id", "street_address"]

for f in categorical:
        if train[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

temp = pd.concat([df.manager_id,pd.get_dummies(train['interest_level'])], axis = 1).groupby('manager_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = df.groupby('manager_id').count().iloc[:,1]
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
unranked_managers_ixes = temp['count']<20
ranked_managers_ixes = ~unranked_managers_ixes
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
df = df.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')

temp = pd.concat([df.building_id,pd.get_dummies(train['interest_level'])], axis = 1).groupby('building_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = df.groupby('building_id').count().iloc[:,1]
temp['building_popularity'] = temp['high_frac']*2 + temp['medium_frac']
unranked_building_ixes = temp['count']<20
ranked_building_ixes = ~unranked_building_ixes
mean_values = temp.loc[ranked_building_ixes, ['high_frac','low_frac', 'medium_frac','building_popularity']].mean()
print(mean_values)
temp.loc[unranked_building_ixes,['high_frac','low_frac', 'medium_frac','building_popularity']] = mean_values.values
df = df.merge(temp.reset_index(),how='left', left_on='building_id', right_on='building_id')

df["no_of_bathrooms"]=df["bathrooms"].apply(lambda x:
    "%s_bathrooms"%int(x) if x<6 else "more_than_5_bed/bathrooms")
df['no_of_bathrooms'] = df['no_of_bathrooms'].apply(lambda x:
    "0_or_5_bathrooms" if x=="5_bathrooms" or x=="0_bathrooms" else x )
df["no_of_bedrooms"]=df["bedrooms"].apply(lambda x:
    "%s_bedrooms"%int(x) if x<6 else "more_than_5_bed/bathrooms")
df['t'] = df['no_of_bedrooms'].apply(lambda x:
    x if x=="more_than_5_bed/bathrooms" else "Remove")
df = pd.get_dummies(df,columns=['no_of_bathrooms','no_of_bedrooms','t'])
print(df.info())



train = df.head(49352)
test = df.tail(74659)
pred =["num_photos", "num_features", "num_description_words",
    "created_year", "created_month", "created_day", 
    "listing_id", "created_hour","bathrooms",
    "bedrooms", "latitude", "longitude", "price",
    "manager_skill","building_popularity",
    "no_of_bedrooms_more_than_5_bed/bathrooms"]
target = ['interest_level']
test_ID = ['listing_id']
X = train[pred]
y = train["interest_level"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=1000,max_depth=10)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)


print(log_loss(y_val, y_val_pred))

X = test[pred]
y = clf.predict_proba(X)
out_df = pd.DataFrame(y)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test.listing_id.values
out_df.to_csv("rf_starter2.csv", index=False)


# Any results you write to the current directory are saved as output.