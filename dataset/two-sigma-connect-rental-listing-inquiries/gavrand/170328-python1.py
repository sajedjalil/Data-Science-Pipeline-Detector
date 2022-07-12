
#renaca4.py

# 1677   0.5389  NO REPLACE NO BED CLEAN
# 1732   0.5391  with REPLACE NO  BED CLEAN

# 1777   0.5387  with REPLACE AND BED CLEAN

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection, preprocessing, ensemble
import random
from math import exp
from sklearn.metrics import log_loss

import xgboost as xgb

from sklearn.cluster import KMeans

random.seed(321)
np.random.seed(321)
print ('Reading...')
X_train = pd.read_json("../input/train.json")
X_test = pd.read_json("../input/test.json")

interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
X_train['interest_level'] = X_train['interest_level'].apply(lambda x: interest_level_map[x])
X_test['interest_level'] = -1
#yy = X_train['interest_level'].ravel()
yy = X_train['interest_level'].values

#add features
feature_transform = CountVectorizer(stop_words='english', max_features=50)
X_train['features'] = X_train["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
X_test['features'] = X_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
feature_transform.fit(list(X_train['features']) + list(X_test['features']))

train_size = len(X_train)
low_count = len(X_train[X_train['interest_level'] == 0])
medium_count = len(X_train[X_train['interest_level'] == 1])
high_count = len(X_train[X_train['interest_level'] == 2])

def find_objects_with_only_one_record(feature_name):
    temp = pd.concat([X_train[feature_name].reset_index(), 
                      X_test[feature_name].reset_index()])
    temp = temp.groupby(feature_name, as_index = False).count()
    return temp[temp['index'] == 1]

managers_with_one_lot = find_objects_with_only_one_record('manager_id')
buildings_with_one_lot = find_objects_with_only_one_record('building_id')
addresses_with_one_lot = find_objects_with_only_one_record('display_address')

lambda_val = None
k=5.0
f=1.0
r_k=0.01 
g = 1.0

def categorical_average(variable, y, pred_0, feature_name):
    def calculate_average(sub1, sub2):
        s = pd.DataFrame(data = {
                                 variable: sub1.groupby(variable, as_index = False).count()[variable],                              
                                 'sumy': sub1.groupby(variable, as_index = False).sum()['y'],
                                 'avgY': sub1.groupby(variable, as_index = False).mean()['y'],
                                 'cnt': sub1.groupby(variable, as_index = False).count()['y']
                                 })
                                 
        tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable) 
        del tmp['index']                       
        tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
        tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

        def compute_beta(row):
            cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
            return 1.0 / (g + exp((cnt - k) / f))
            
        if lambda_val is not None:
            tmp['beta'] = lambda_val
        else:
            tmp['beta'] = tmp.apply(compute_beta, axis = 1)
            
        tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred_0'],
                                   axis = 1)
                                   
        tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred_0']
        tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred_0']
        tmp['random'] = np.random.uniform(size = len(tmp))
        tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] *(1 + (row['random'] - 0.5) * r_k),
                                   axis = 1)
    
        return tmp['adj_avg'].ravel()
     
    #cv for training set 
    k_fold = StratifiedKFold(5)
    X_train[feature_name] = -999 
    for (train_index, cv_index) in k_fold.split(np.zeros(len(X_train)),
                                                X_train['interest_level'].ravel()):
        sub = pd.DataFrame(data = {variable: X_train[variable],
                                   'y': X_train[y],
                                   'pred_0': X_train[pred_0]})
            
        sub1 = sub.iloc[train_index]        
        sub2 = sub.iloc[cv_index]
        
        X_train.loc[cv_index, feature_name] = calculate_average(sub1, sub2)
    
    #for test set
    sub1 = pd.DataFrame(data = {variable: X_train[variable],
                                'y': X_train[y],
                                'pred_0': X_train[pred_0]})
    sub2 = pd.DataFrame(data = {variable: X_test[variable],
                                'y': X_test[y],
                                'pred_0': X_test[pred_0]})
    X_test.loc[:, feature_name] = calculate_average(sub1, sub2)                               


def transform_data(X):
    
    
    print('Identify bad geographic coordinates')
    X['bad_addr'] = 0
    mask = ~X['latitude'].between(40.5, 40.9)
    mask = mask | ~X['longitude'].between(-74.05, -73.7)
    bad_rows = X[mask]
    X.loc[mask, 'bad_addr'] = 1
    
    print('Create neighborhoods')
    # Replace bad values with mean
    mean_lat = X.loc[X['bad_addr']==0, 'latitude'].mean()
    X.loc[X['bad_addr']==1, 'latitude'] = mean_lat
    mean_long = X.loc[X['bad_addr']==0, 'longitude'].mean()
    X.loc[X['bad_addr']==1, 'longitude'] = mean_long
    # From: https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/unsupervised-and-supervised-neighborhood-encoding
    kmean_model = KMeans(42)
    loc_df = X[['longitude', 'latitude']].copy()
    standardize = lambda x: (x - x.mean()) / x.std()
    loc_df['longitude'] = standardize(loc_df['longitude'])
    loc_df['latitude'] = standardize(loc_df['latitude'])
    kmean_model.fit(loc_df)
    X['neighborhoods'] = kmean_model.labels_    
    
    
    
    #features
    X['features']=X['features'].str.replace("garden/patio","garden")
    X['features']=X['features'].str.replace("patio","garden")
    X['features']=X['features'].str.replace("residents_garden","garden")
    X['features']=X['features'].str.replace("common_garden","garden")
    
    X['features']=X['features'].str.replace("wifi_access","wifi")
    
    X['features']=X['features'].str.replace("24/7","24")
    X['features']=X['features'].str.replace("24-hour","24")
    X['features']=X['features'].str.replace("24hr","24")
    X['features']=X['features'].str.replace("concierge","doorman")
    X['features']=X['features'].str.replace("ft_doorman","doorman")
    X['features']=X['features'].str.replace("24_doorman","doorman")
    X['features']=X['features'].str.replace("24_hr_doorman","doorman")
    X['features']=X['features'].str.replace("doorman_service","doorman")
    X['features']=X['features'].str.replace("full-time_doorman","doorman")

    feat_sparse = feature_transform.transform(X["features"])
    vocabulary = feature_transform.vocabulary_
    del X['features']
    X1 = pd.DataFrame([ pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0]) ])
    X1.columns = list(sorted(vocabulary.keys()))
    X = pd.concat([X.reset_index(), X1.reset_index()], axis = 1)
    del X['index']
    
    #add NEW features    
    X["num_photos"] = X["photos"].apply(len)
    X['created'] = pd.to_datetime(X["created"])
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X['price_per_bed']  = X['price'] / X['bedrooms']    
    X['price_per_bath'] = X['price'] / X['bathrooms']
    X['price_per_room'] = X['price'] / (X['bathrooms'] + X['bedrooms'] )
    #X['Price_sqft proxy'] = (X['price'] / (2 + X['bedrooms'].clip(1, 4) + 0,5*X['bathrooms'].clip(0, 2))).values
    X['low'] = 0
    X.loc[X['interest_level'] == 0, 'low'] = 1
    X['medium'] = 0
    X.loc[X['interest_level'] == 1, 'medium'] = 1
    X['high'] = 0
    X.loc[X['interest_level'] == 2, 'high'] = 1
    
    X['display_address'] = X['display_address'].apply(lambda x: x.lower().strip())
    X['street_address']  = X['street_address'].apply(lambda x: x.lower().strip())
    
    X['pred0_low'] = low_count * 1.0 / train_size
    X['pred0_medium'] = medium_count * 1.0 / train_size
    X['pred0_high'] = high_count * 1.0 / train_size
    
    X.loc[X['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()), 
          'manager_id'] = "-1"
    X.loc[X['building_id'].isin(buildings_with_one_lot['building_id'].ravel()), 
          'building_id'] = "-1"
    X.loc[X['display_address'].isin(addresses_with_one_lot['display_address'].ravel()), 
          'display_address'] = "-1"


    X['building_id0'] = 0
    X.loc[X['building_id'] == 0, 'building_id0'] = 1
    X['half_bathrooms'] = ((np.round(X.bathrooms) - X.bathrooms)!=0).astype(float) # Half bathrooms? 1.5, 2.5, 3.5...
    X['bedrooms_clipped'] = X['bedrooms'].clip(1, 4)
    X['bathrooms_clipped'] = X['bedrooms'].clip(0, 2)
    X['Price_sqft proxy'] = X['price'] / (2+ X['bedrooms_clipped'] +0.5*X['bathrooms_clipped'] )
    #mm_clip = lambda x, l, u: max(l, min(u, x))
    #X['Price_sqft proxy'] = (X['price'] / (2 + mm_clip(X['bedrooms'],1, 4) + 0,5*mm_clip(X['bathrooms'],0.2)))
    #X['Price_sqft proxy'] = (X.price / (2 + X.bedrooms.clip(1, 4) + 0,5*X.bathrooms.clip(0, 2))).values
    #X['Price_sqft proxy'] = (X.price / (2 + X['bedrooms'].apply(lambda x: x.clip(1, 4)) + 0,5*X['bathrooms'].apply(lambda x: x.clip(0, 2))))
    #X['Price_sqft proxy'] = (X.price / (2 + X['bedrooms'].apply(lambda x: max(1, min(4, x))) + 0,5*X['bathrooms'].apply(lambda x: max(0, min(2, x)))))

    
    #Beds cleaning(0>>1)
    nora=0
    beds=np.array(X[["bedrooms","description"]])
    new_beds=np.zeros(X.shape[0])
    for rr in range(X.shape[0]):
        a1=beds[rr][0]
        a2=str(beds[rr][1]).lower()
        if (a2.find('one bedroom') != -1  or a2.find('1 bedroom') != -1) and a1==0:
            new_beds[rr]=1
            nora+=1
        else:
            new_beds[rr]=a1
    X['bedrooms']=new_beds   
    print (nora,' bedrooms changed')

    return X

def normalize_high_cordiality_data():
    high_cardinality = ["building_id", "manager_id"]
    for c in high_cardinality:
        categorical_average(c, "medium", "pred0_medium", c + "_mean_medium")
        categorical_average(c, "high", "pred0_high", c + "_mean_high")

def transform_categorical_data():
    categorical = ['building_id', 'manager_id', 
                   'display_address', 'street_address']
                   
    for f in categorical:
        encoder = LabelEncoder()
        encoder.fit(list(X_train[f]) + list(X_test[f])) 
        X_train[f] = encoder.transform(X_train[f].ravel())
        X_test[f] = encoder.transform(X_test[f].ravel())
                  

def remove_columns(X):
    columns = ["photos", "pred0_high", "pred0_low", "pred0_medium",
               "description", "low", "medium", "high",
               "interest_level", "created"]
    for c in columns:
        del X[c]
#===============================================================================
#===============================================================================
print("Starting transformations")        
X_train = transform_data(X_train)    
X_test = transform_data(X_test) 
y = X_train['interest_level'].ravel()

print("Normalizing high cordiality data...")
normalize_high_cordiality_data()
transform_categorical_data()

print ("train-test transforming...")


remove_columns(X_train)
remove_columns(X_test)

print ('TRAIN features...',X_train.shape)
#===============================================
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.03
param['max_depth'] = 4
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 321
param['nthread'] = 8
num_rounds = 2000

param2 = {}
param2['objective'] = 'multi:softprob'
param2['eta'] = 0.03
param2['max_depth'] = 6
param2['silent'] = 1
param2['num_class'] = 3
param2['eval_metric'] = "mlogloss"
param2['min_child_weight'] = 1
param2['subsample'] = 0.8
param2['colsample_bytree'] = 0.7
param2['seed'] = 321
param2['nthread'] = 8
num_rounds2 = 2000
#===============================================
print ('CV...')
#CV...
cv=0
if cv :
        cv_scores = []
        cv_scores2 = []
        cv_scoresx = []

        noa=0
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
        for dev_index, val_index in kf.split(range(X_train.shape[0])):
                XX=np.array(X_train)
                print (dev_index)
                noa+=1
                print ('Training ..',noa)
                dev_X, val_X =  XX[dev_index,:], XX[val_index,:]
                dev_y, val_y =  yy[dev_index],  yy[val_index]
                
                xgtrain = xgb.DMatrix(dev_X, label=dev_y)
                xgtest = xgb.DMatrix(val_X, label=val_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                
                model = xgb.train(param, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
                pred_test_xgb  = model.predict(xgtest)
                
                #model2 = xgb.train(param2, xgtrain, num_rounds2, watchlist, early_stopping_rounds=20)
                #pred_test_xgb2  = model2.predict(xgtest)
                               
                score1=log_loss(val_y,pred_test_xgb)
                #score2=log_loss(val_y,pred_test_xgb2)
                
                #pred_test_xgbx=(pred_test_xgb+pred_test_xgb2)/2
                #scorex=log_loss(val_y,pred_test_xgbx)
                
                cv_scores.append( round(log_loss(val_y, pred_test_xgb),4))
              #  cv_scores2.append(round(log_loss(val_y, pred_test_xgb2),4))
               # cv_scoresx.append(round(log_loss(val_y, pred_test_xgbx),4))
                break
                
               
        print ('**** AVERAGE 1...',cv_scores,np.mean(cv_scores))      
      #  print ('**** AVERAGE 2...',cv_scores2,np.mean(cv_scores2))      
      #  print ('**** AVERAGE x...',cv_scoresx,np.mean(cv_scoresx))      

#===============================================
print("Start fitting...")

xgtrain = xgb.DMatrix(X_train, label=y)
clf = xgb.train(param, xgtrain, num_rounds)

print("Fitted")

def prepare_submission(model):
    xgtest = xgb.DMatrix(X_test)
    
    preds = model.predict(xgtest) 
  #  preds2 = model2.predict(xgtest) 
    
    xpreds=(preds)
    sub = pd.DataFrame(data = {'listing_id': X_test['listing_id'].ravel()})
    sub['low']    = xpreds[:, 0]
    sub['medium'] = xpreds[:, 1]
    sub['high']   = xpreds[:, 2]
    sub.to_csv("submission.csv", index = False, header = True)

prepare_submission(clf)