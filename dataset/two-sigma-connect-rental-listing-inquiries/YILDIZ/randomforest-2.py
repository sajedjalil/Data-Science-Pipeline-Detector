# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, preprocessing, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#input data
train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]  #numeric features

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year

train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month

train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day

#train_df["created_hour"] = train_df["created"].dt.hour
#test_df["created_hour"] = test_df["created"].dt.hour

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words", "created_year", "created_month", "created_day", "listing_id"])

print("Starting building level...")
train_df.ix[train_df.building_id == '0', 'new_building_id'] = train_df['building_id'] + train_df['manager_id']
train_df.ix[train_df.building_id != '0', 'new_building_id'] = train_df['building_id']

a=[np.nan]*len(train_df)
building_level={}

for bid in train_df['new_building_id'].values:
    building_level[bid]=[0,0,0]
    
for j in range(train_df.shape[0]):
    rec=train_df.iloc[j]
    if rec['interest_level']=='low':
        building_level[rec['new_building_id']][0]+=1
    if rec['interest_level']=='medium':
        building_level[rec['new_building_id']][1]+=1
    if rec['interest_level']=='high':
        building_level[rec['new_building_id']][2]+=1
        
for j in range(train_df.shape[0]):    
        rec=train_df.iloc[j]
        occurance = sum(building_level[rec['new_building_id']])
        if occurance!=0:
            a[j]= (building_level[rec['new_building_id']][0]*0.0 + building_level[rec['new_building_id']][1]*1.0 \
                   + building_level[rec['new_building_id']][2]*2.0) / occurance

train_df['building_level']=a

test_df.ix[test_df.building_id == '0', 'new_building_id'] = test_df['building_id'] + test_df['manager_id']
test_df.ix[test_df.building_id != '0', 'new_building_id'] = test_df['building_id']

b=[]
for i in test_df['new_building_id'].values:
    if i not in building_level.keys():
        b.append(np.nan)
    else:
        occurance = sum(building_level[i])
        b.append((building_level[i][0]*0.0 + building_level[i][1]*1.0 \
                   + building_level[i][2]*2.0) / occurance)

test_df['building_level']=b

train_df = train_df.drop(['new_building_id'], axis=1)
test_df = test_df.drop(['new_building_id'], axis=1)

features_to_use.append('building_level')
print("building_level calculation...DONE")

# Create our imputer to replace missing values within test data:
imp=Imputer(missing_values="NaN", strategy="mean", axis=0)
imp.fit(test_df[["building_level"]])
test_df["building_level"]=imp.transform(test_df[["building_level"]]).ravel()


categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
print("Categorical features...DONE")

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=200)
tfidf.fit(list(train_df['features']) + list(test_df['features']))
tr_sparse = tfidf.transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape, train_y.shape)

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.33)


print("Training...n_estimators=1000")
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
print("Checking...n_estimators=1000")
y_val_pred = clf.predict_proba(X_val)
print(log_loss(y_val, y_val_pred))


# make prediction with classifier built with 1000 estimator:
y = clf.predict_proba(test_X)

sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, target_num_map[label]]
sub.to_csv("random_forest1.csv", index=False)