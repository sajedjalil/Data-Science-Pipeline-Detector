import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#input data
train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')

#basic features
train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
test_df["price_t"] = test_df["price"]/test_df["bedrooms"] 
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))


features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price","price_t","num_photos", "num_features", "num_description_words","listing_id"]

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.03
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
    
df_train = train_df.copy()
#df_train.reset_index(inplace=True)

index=list(range(df_train.shape[0]))

random.seed(0)
random.shuffle(index)

df_train['manager_level_low'] = np.nan
df_train['manager_level_medium'] = np.nan
df_train['manager_level_high'] = np.nan

df_train['manager_price_low'] = np.nan
df_train['manager_price_medium'] = np.nan
df_train['manager_price_high'] = np.nan

for i in range(5):
    test_index = index[int((i*df_train.shape[0])/5):int(((i+1)*df_train.shape[0])/5)]
    train_index = list(set(index).difference(test_index)) 

    cv_train = df_train.iloc[train_index]
    cv_test  = df_train.iloc[test_index]

    for m in cv_train.groupby('manager_id'):
        test_subset = cv_test[cv_test.manager_id == m[0]].index

        df_train.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 'low').mean()
        df_train.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 'medium').mean()
        df_train.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 'high').mean()
        df_train.loc[test_subset, 'manager_price'] = m[1].price.mean()
        df_train.loc[test_subset, 'manager_count'] = m[1].size
        
# now for the test data

df_test = test_df.copy()

df_test['manager_level_low'] = np.nan
df_test['manager_level_medium'] = np.nan
df_test['manager_level_high'] = np.nan

for m in df_train.groupby('manager_id'):
    test_subset = df_test[df_test.manager_id == m[0]].index

    df_test.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 'low').mean()
    df_test.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 'medium').mean()
    df_test.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 'high').mean()
    df_test.loc[test_subset, 'manager_price'] = m[1].price.mean()
    df_test.loc[test_subset, 'manager_count'] = m[1].size
    
train_df=df_train
test_df=df_test

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')
features_to_use.append('manager_price') 
features_to_use.append('manager_count')

categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_df["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

preds, model = runXGB(train_X, train_y, test_X, num_rounds=1050)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("sub.csv", index=False)
