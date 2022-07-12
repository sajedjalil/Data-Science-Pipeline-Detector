import os
import sys
import operator
import math
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def runXGB(train_X, train_y, test_X, num_rounds):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 5
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = 8088
    num_rounds = num_rounds
    
    
    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    watchlist = [ (xgtrain,'train')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=25,
                     verbose_eval=25)#False)
    xgtest = xgb.DMatrix(test_X)
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

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

train_df["price_per_bed"] = train_df["price"]/train_df["bedrooms"].clip(lower=1)
test_df["price_per_bed"] = test_df["price"]/test_df["bedrooms"].clip(lower=1)
train_df['created_date']=np.array(train_df.created.values, dtype='datetime64[D]'
                                 ).astype(np.float32)
test_df['created_date']=np.array(test_df.created.values, dtype='datetime64[D]'
                                 ).astype(np.float32)

train_df['display_address'] = train_df['display_address'].apply(lambda x: x.lower().strip())
test_df['display_address'] = test_df['display_address'].apply(lambda x: x.lower().strip())
train_df['street_address'] = train_df['street_address'].apply(lambda x: x.lower().strip())
test_df['street_address'] = test_df['street_address'].apply(lambda x: x.lower().strip())

# adding all these new features to use list #
features_to_use.extend(["num_photos", "num_features", "num_description_words",
                        "created_month", "listing_id", "created_hour", 
                        "price_per_bed", "created_date"])
                        
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
tfidf.fit(train_df["features"])

train_df_tr=train_df
tr_sparse_tr = tfidf.transform(train_df_tr["features"])
te_sparse = tfidf.transform(test_df["features"])

def manager(train_df_tr, test_df):
    temp = pd.concat([train_df_tr.manager_id,pd.get_dummies(train_df_tr.interest_level)], axis = 1
                    ).groupby('manager_id').mean()
    temp.columns = ['high_frac','low_frac', 'medium_frac']
    temp['count'] = train_df_tr.groupby('manager_id').count().iloc[:,1]

    temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
    unranked_managers_ixes = temp['count']<20
    ranked_managers_ixes = ~unranked_managers_ixes
    mean_values = temp.loc[ranked_managers_ixes, [
        'high_frac','low_frac', 'medium_frac','manager_skill']].mean()
    temp.loc[unranked_managers_ixes,[
        'high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

    train_df_tr = train_df_tr.merge(temp.reset_index(),how='left', on='manager_id')

    test_df = test_df.merge(temp.reset_index(),how='left', on='manager_id')
    new_manager_ixes = test_df['high_frac'].isnull()
    test_df.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill'
                                ]] = mean_values.values

    return (train_df_tr, test_df)

features_sparse = ['features_'+str(i) for i in range(tr_sparse_tr.shape[1])]
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y_tr = np.array(train_df_tr['interest_level'].apply(lambda x: target_num_map[x]))

p_high=(train_y_tr==0).sum()/train_y_tr.shape[0]
p_medium=(train_y_tr==1).sum()/train_y_tr.shape[0]
print(p_high)
print(p_medium)

def cardinal(train_df_tr, test_df, f, one=False):
    temp=pd.concat([train_df_tr[f], 
                    train_df_tr['interest_level'].apply(lambda x: target_num_map[x])], axis=1)

    if one:
        tempc=temp.groupby(f)['interest_level'].agg(np.size)
        tempc=tempc[tempc==1]
        train_df_tr.loc[train_df_tr[f].isin(tempc.index.ravel()), 
              f] = -1
        test_df.loc[test_df[f].isin(tempc.index.ravel()), 
              f] = -1

        temp=pd.concat([train_df_tr[f], 
                        train_df_tr['interest_level'].apply(lambda x: target_num_map[x])], axis=1)
    
    temp['high']=(temp['interest_level']==0).astype(np.int)
    temp['medium']=(temp['interest_level']==1).astype(np.int)

    temp2=temp.groupby(f)['high', 'medium'].agg(np.average)
    temp2['count']=temp.groupby(f)['interest_level'].agg(np.size)
    temp2['l']=temp2['count'].apply(lambda x: 1/(1+math.e**(-(min(x, 670)-20))))
    r_k=0.01
    temp2[f+'_high']=temp2['high']*temp2['l'] + p_high*(1-temp2['l']
        )# * np.random.uniform(1 - r_k, 1 + r_k, len(temp2))
    temp2[f+'_medium']=temp2['medium']*temp2['l'] + p_medium*(1-temp2['l']
        )# * np.random.uniform(1 - r_k, 1 + r_k, len(temp2))
    train_df_tr=train_df_tr.reset_index().merge(temp2[[f+'_high', f+'_medium']], how='left', 
                                  left_on=[f], right_index=True).set_index('index')
    test_df=test_df.reset_index().merge(temp2[[f+'_high', f+'_medium']], how='left', 
                                  left_on=[f], right_index=True).set_index('index')
    test_df.loc[test_df[f+'_high'].isnull(), f+'_high'] = p_high
    test_df.loc[test_df[f+'_medium'].isnull(), f+'_medium'] = p_medium
    
    return (train_df_tr, test_df)

features_to_use=['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos', 
                 'num_features', 'num_description_words', 'created_month', 'listing_id', 
                 'created_hour', 'price_per_bed', 'created_date', 'display_address', 'manager_id', 
                 'building_id', 'street_address', 'manager_skill']

features_to_use2=features_to_use# + ['building_id_high', 'building_id_medium']

train_df_tr=train_df_tr.reset_index(drop=True)

tr=train_df_tr
te=test_df

(tr, te) = manager(tr, te)

(tr, te)=cardinal(tr, te, 'building_id')

train_X_tr = sparse.hstack([tr[features_to_use2], 
                            tr_sparse_tr]).tocsr()
test_X = sparse.hstack([te[features_to_use2], te_sparse]).tocsr()
preds, model = runXGB(train_X_tr, train_y_tr, test_X, 1200)
print("%d  %.6f"%(model.best_iteration, model.best_score))

out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("sub.csv", index=False)
