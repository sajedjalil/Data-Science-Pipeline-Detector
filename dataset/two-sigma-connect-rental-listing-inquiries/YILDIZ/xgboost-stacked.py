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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, classification_report

import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer


data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
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
features_to_use.extend(["num_photos", "num_features", "num_description_words", "created_month", "created_day", "created_hour", "weekday"])
features_for_xgb.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour"])

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

features_to_use.append('building_level')


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

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')

##Categorical - LabelEncoder
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)
            features_for_xgb.append(f)
            
            
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

#Train and test set for XGBoost
train_X = sparse.hstack([train_df[features_for_xgb], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_for_xgb], te_sparse]).tocsr()

target_num_map = {'high':2, 'medium':1, 'low':0}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))


#Prepare Train and test sets for Trees
fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
train_imputed = pd.DataFrame(fill_NaN.fit_transform(train_df[features_to_use]))
train_imputed.columns = train_df[features_to_use].columns
train_imputed.index = train_df.index

test_imputed = pd.DataFrame(fill_NaN.fit_transform(test_df[features_to_use]))
test_imputed.columns = test_df[features_to_use].columns
test_imputed.index = test_df.index

train_Xtree = train_imputed
test_Xtree = test_imputed

print("Printing final train and test sets...")
#Train and test set for XGBoost
print(train_X.shape, test_X.shape)

#Train and test set for trees
print(train_Xtree.shape, test_Xtree.shape)

#####################################
#Begin Stacking
#####################################
NFOLDS = 5
SEED = 0
y_train = train_y

ntrain = train_Xtree.shape[0]
ntest = test_Xtree.shape[0]
print("{},{}".format(ntrain, ntest))

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict_prb(self, x):
        return self.clf.predict(x)
    
    def predict_proba(self, x):
        return self.clf.predict_proba(x)
        
        
def get_oof(clf):
    oof_train = np.zeros((ntrain,3))
    oof_test = np.zeros((ntest,3))
    oof_test_skf = np.empty((NFOLDS, ntest, 3))

    i = 0
    for train_index, test_index in skf.split(x_train, y_train):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        rf1.train(x_tr, y_tr)

        oof_train[test_index]= rf1.predict_proba(x_te)
        oof_test_skf[i, :, :] = rf1.predict_proba(x_test)
        i += 1

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test
    
#Prepare train and test sets
train_test = pd.concat((train_Xtree, test_Xtree)).reset_index(drop=True)
x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])

print("{},{},{}".format(x_train.shape, y_train.shape, x_test.shape))

skf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED, shuffle=True)

rf1_params = {
    'n_jobs': 16,
    'n_estimators': 10,
    'criterion' : "entropy",
    'max_features': 0.5,
    'max_depth': 6,
    'min_samples_leaf': 2,
}

rf2_params = {
    'n_jobs': 16,
    'criterion' : "gini",
    'n_estimators': 1000,
    'max_features': None,
    'max_depth': 8,
    'min_samples_leaf': 1,
}

et1_params = {
    'n_jobs': 16,
    'n_estimators': 10,
    'max_features': "auto",
    'criterion' : "gini",
    'max_depth': 4,
    'min_samples_leaf': 2,
}

et2_params = {
    'n_jobs': 16,
    'n_estimators': 1000,
    'criterion' : "entropy",
    'max_depth': 12,
    'min_samples_leaf': 2,
    'max_features': 0.8,
}

#Level one models:
et1 = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et1_params)
et2 = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et2_params)

rf1 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf1_params)
rf2 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf2_params)

gbc1 = GradientBoostingClassifier(n_estimators = 10, max_depth = 4, subsample = 0.5,
          learning_rate = 0.1, min_samples_leaf = 2, random_state = 0)

gbc2 = GradientBoostingClassifier(n_estimators = 1000, max_depth = 8, 
          learning_rate = 0.5, min_samples_leaf = 1, random_state = 0)

et1_oof_train, et1_oof_test = get_oof(et1)
et2_oof_train, et2_oof_test = get_oof(et2)

rf1_oof_train, rf1_oof_test = get_oof(rf1)
rf2_oof_train, rf2_oof_test = get_oof(rf2)

gbc1_oof_train, gbc1_oof_test = get_oof(gbc1)
gbc2_oof_train, gbc2_oof_test = get_oof(gbc2)

print("ET1-CV: {}".format(log_loss(y_train, et1_oof_train)))
print("ET2-CV: {}".format(log_loss(y_train, et2_oof_train)))

print("RF1-CV: {}".format(log_loss(y_train, rf1_oof_train)))
print("RF2-CV: {}".format(log_loss(y_train, rf2_oof_train)))

print("GBC1-CV: {}".format(log_loss(y_train, gbc1_oof_train)))
print("GBC2-CV: {}".format(log_loss(y_train, gbc2_oof_train)))

## Prepare XGBoost
class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'eta': 0.1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': "mlogloss",
    'nrounds': 400
}

train_test = vstack([train_X, test_X]).toarray()
x_train = np.array(train_test[:ntrain,:])
x_test = np.array(train_test[ntrain:,:])

print("{},{},{}".format(x_train.shape, y_train.shape, x_test.shape))

xg = XgbWrapper(seed=SEED, params=xgb_params)
xg_oof_train, xg_oof_test = get_oof(xg)
print("XG-CV: {}".format(log_loss(y_train, xg_oof_train)))

## Level 2 stacking
x_train = np.concatenate((xg_oof_train, et1_oof_train, rf2_oof_train, gbc2_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et1_oof_test, rf2_oof_test, gbc2_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params_2 = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'eta': 0.1,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mlogloss',   
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)
             
best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params_2, dtrain, best_nrounds)
preds = gbdt.predict(dtest)

out_df = pd.DataFrame(preds)
out_df.columns = ["low", "medium", "high"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_stacked_1.csv", index=False)  
