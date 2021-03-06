# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, preprocessing, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold



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

"""
#basic features
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

train_df["price_t"] =train_df[["price","room_sum"]].apply(lambda x: x[0]/x[1] if x[1]!=0 else x[0], axis=1)
test_df["price_t"] = test_df[["price","room_sum"]].apply(lambda x: x[0]/x[1] if x[1]!=0 else x[0], axis=1)

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))


features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price","num_photos", "num_features", "num_description_words","listing_id"]

"""

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


#Manager_level
#### Prepare train data
index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c

#### Prepare test data
a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')

#Categorical:
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
            
#Features:
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
tfidf = CountVectorizer(stop_words='english', max_features=200)
tfidf.fit(list(train_df['features']) + list(test_df['features']))
tr_sparse = tfidf.transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

#Impute:
fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
train_imputed = pd.DataFrame(fill_NaN.fit_transform(train_df[features_to_use]))
train_imputed.columns = train_df[features_to_use].columns
train_imputed.index = train_df.index

test_imputed = pd.DataFrame(fill_NaN.fit_transform(test_df[features_to_use]))
test_imputed.columns = test_df[features_to_use].columns
test_imputed.index = test_df.index

#Build final dataset:
train_X = sparse.hstack([train_imputed, tr_sparse]).tocsr()
test_X = sparse.hstack([test_imputed, te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape, train_y.shape)

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20)



print("Training...n_estimators=100")
clf = ExtraTreesClassifier(n_estimators=100, max_features='auto', min_samples_leaf=1)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
print(log_loss(y_val, y_val_pred))


print("Training...n_estimators=100")
rf = RandomForestClassifier(n_estimators=100, max_features='auto', min_samples_leaf=1)
rf.fit(X_train, y_train)
y_val_pred = rf.predict_proba(X_val)
print(log_loss(y_val, y_val_pred))





"""
# make prediction for test set - use the model for n_estimators=1000
y = clf.predict_proba(test_X)

sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, target_num_map[label]]
sub.to_csv("submission_rf_base.csv", index=False)
"""