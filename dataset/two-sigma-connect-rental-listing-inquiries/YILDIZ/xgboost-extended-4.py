# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np
import pandas as pd
import random
from scipy import sparse
from scipy.sparse import vstack

from sklearn import model_selection, preprocessing
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
    
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

features_for_xgb  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
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

train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

train_df["weekday"] = train_df["created"].dt.weekday
test_df["weekday"] = test_df["created"].dt.weekday

# adding all these new features to use list #
features_for_xgb.extend(["num_photos", "num_features", "num_description_words", "created_year", "created_month", "created_day", "listing_id", "created_hour", "weekday"])

# Building Level
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

features_for_xgb.append('building_level')


#####################################################################
# Manager Level
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

features_for_xgb.append('manager_level_low') 
features_for_xgb.append('manager_level_medium') 
features_for_xgb.append('manager_level_high')



categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_for_xgb.append(f)
            
            
train_df["room_sum"] = train_df["bedrooms"] + train_df["bathrooms"]
test_df["room_sum"] = test_df["bedrooms"] + test_df["bathrooms"]
train_df['room_sum'] = train_df.room_sum.clip(lower=1)
test_df['room_sum'] = test_df.room_sum.clip(lower=1)

train_df["logprice"] = np.log(train_df.price)
test_df["logprice"] = np.log(test_df.price)

train_df["price_per_room"] = train_df["price"] / train_df["room_sum"]
test_df["price_per_room"] = test_df["price"] / test_df["room_sum"]

train_df['half_bathrooms'] = ((np.round(train_df.bathrooms) - train_df.bathrooms)!=0).astype(float) # Half bathrooms? 1.5, 2.5, 3.5...
test_df['half_bathrooms'] = ((np.round(test_df.bathrooms) - test_df.bathrooms)!=0).astype(float) # Half bathrooms? 1.5, 2.5, 3.5...

train_df["price_sqft"] = (train_df.price / (1 + train_df.bedrooms.clip(1, 4) + 0.5*train_df.bathrooms.clip(0, 2)))
test_df["price_sqft"] = (test_df.price / (1 + test_df.bedrooms.clip(1, 4) + 0.5*test_df.bathrooms.clip(0, 2)))

features_for_xgb.extend(["room_sum", "logprice", "price_per_room", "half_bathrooms", "price_sqft"])



train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

#Train and test set for XGBoost
train_X = sparse.hstack([train_df[features_for_xgb], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_for_xgb], te_sparse]).tocsr()
#####################################################################
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

'''
cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break
'''

preds, model = runXGB(train_X, train_y, test_X, num_rounds=200)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgboost_extended_4.csv", index=False)