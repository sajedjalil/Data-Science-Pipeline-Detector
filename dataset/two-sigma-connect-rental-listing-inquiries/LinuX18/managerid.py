import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
df = pd.read_json(open("../input/train.json", "r"))

df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day

features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
                   "num_photos", "num_features", "num_description_words",
                   "created_year", "created_month", "created_day"]
from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()
lbl.fit(list(df['manager_id'].values))
df['manager_id'] = lbl.transform(list(df['manager_id'].values))

features_to_use.append('manager_id')
X = df[features_to_use]
y = df["interest_level"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
# compute fractions and count for each manager
temp = pd.concat([X_train.manager_id,pd.get_dummies(y_train)], axis = 1).groupby('manager_id').mean()
temp.columns = ['high_frac','low_frac', 'medium_frac']
temp['count'] = X_train.groupby('manager_id').count().iloc[:,1]

# remember the manager_ids look different because we encoded them in the previous step 
print(temp.tail(10))
# compute skill
temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

# get ixes for unranked managers...
unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
print(temp.tail(10))

X_train = X_train.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
X_train.head()
X_val = X_val.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
new_manager_ixes = X_val['high_frac'].isnull()
X_val.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
X_val.head()
features_to_use.extend(['high_frac','low_frac', 'medium_frac','manager_skill'])
these_features = [f for f in features_to_use if f not in ['manager_id','high_frac','low_frac', 'medium_frac','manager_skill']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
print(log_loss(y_val, y_val_pred))
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')

# add manager_id
these_features = [f for f in features_to_use if f not in ['high_frac','low_frac', 'medium_frac','manager_skill']]
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
print(log_loss(y_val, y_val_pred))
# Let's visualize features importance
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')

# no manager_id, use fractions and skill instad
these_features = [f for f in features_to_use if f not in ['manager_id']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
print(log_loss(y_val, y_val_pred))
pd.Series(index = these_features, data = clf.feature_importances_).sort_values().plot(kind = 'bar')

# no manager_id, no skill, use fractions
these_features = [f for f in features_to_use if f not in ['manager_id','manager_skill']]

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train[these_features], y_train)
y_val_pred = clf.predict_proba(X_val[these_features])
log_loss(y_val, y_val_pred)


