# coding=utf8
# Based on yibo's R script

import pandas as pd
import math
import numpy as np
import xgboost as xgb
import time
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.metrics import log_loss
##In this script, we develop a manual App2Vec encoding, which is based 
#on the crosstab of classes in the training data. The new feature is 
#added to the xgb-analysis performed by JianXiao. 

#maximum number of noisy observations injected  per column to the crosstab-feature
NOISE_CT = 2

#upper bound for the popularity of an app 
POP_BOUND = 100

#number of classes to be predicted
NCLASSES = 12

#seed for randomness
SEED = 1747

#only apps used by at least MIN_COUNT users are considered
MIN_COUNT = 10

NROWS = 500000000


#print('read data')
start = time.clock()
train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str}).set_index('device_id')['group']
test = pd.read_csv("../input/gender_age_test.csv",  dtype={'device_id': np.str}).set_index('device_id')
events = pd.read_csv("../input/events.csv", index_col = 0, dtype={'device_id': np.str}, nrows = NROWS)[['device_id']]
app_ev = pd.read_csv("../input/app_events.csv", index_col = 0,  nrows = NROWS)[['app_id']]
print('{0} data loaded'.format(time.clock() - start))

#join train and events data
train_test = np.hstack([train.index, test.index])
events = events[events['device_id'].isin(train_test)]

dec = LabelEncoder().fit(train_test)
train_test = dec.transform(train_test).astype(np.int32)
train.index = dec.transform(train.index).astype(np.int32)
test.index = dec.transform(test.index).astype(np.int32)
events['device_id'] = dec.transform(events['device_id']).astype(np.int32)
app_ev['app_id'] = LabelEncoder().fit_transform(app_ev['app_id']).astype(np.int16)


device_apps_inner = events.merge(app_ev, 'inner', left_index = True, right_index = True).set_index('device_id')
print('{0} merge performed'.format(time.clock() - start))

# filter out rarely used apps
apps_train = device_apps_inner[device_apps_inner.index.isin(train.index)]['app_id']
apps_test = device_apps_inner[device_apps_inner.index.isin(test.index)]['app_id']

train_counts = apps_train.value_counts()
test_counts = apps_test.value_counts()

relevant_train = train_counts[train_counts > MIN_COUNT].index
relevant_test = test_counts[test_counts > MIN_COUNT].index
relevant_apps = np.intersect1d(relevant_train, relevant_test)

device_apps_filtered = device_apps_inner[device_apps_inner['app_id'].isin(relevant_apps)]

del events
del device_apps_inner
del app_ev

#In order to avoid distortion of validation scores, we select a hold-out set. Moreover, we split the data into two blending halves.
train_indices = np.intersect1d(train.index, device_apps_filtered.index)

train_mask = train.index.isin(train_indices)
labels = LabelEncoder().fit_transform(train[train_mask])

#first validation set
train_ids_filtered, _, y_train, _ = train_test_split(train_indices, labels, stratify = labels, train_size = 0.8, random_state = SEED)

#second split in stacking halves
train_ids_0, train_ids_1, y_train_0, y_train_1 = train_test_split(train_ids_filtered, y_train,
                                                          stratify = y_train, train_size = 0.5, random_state = SEED)
blend_ids = [train_ids_0,  train_ids_1]
blend_labels = [y_train_0, y_train_1]



##################
#CROSSTAB FEATURE
##################
#Our goal is to embed the bags of apps and labels into a lower dimensional space. Instead of training a neural network, we use an ad hoc approach, where each app is encoded by its histogram. 

class CrossTabEncoder(BaseEstimator, TransformerMixin):
    """CrossTabEncoder
    A CrossTabEncoder characterizes a feature by its crosstab dataframe.
    """
    def __init__(self):
        self.crosstab_total = None
        self.crosstabs = None
        self.ids_pair = None

    def fit(self, data, ids_pair):
        """For each class of the considered feature, the empirical histogram for the prediction classes is computed. 
        
        Parameters
        ----------
        data : feature column used for the histogram computation
        ids_pair : pair of ids used to split the training ids
        """
        self.ids_pair = ids_pair
        np.random.seed(SEED)
        merged_data = [train[train.index.isin(ids)].to_frame().merge(data,
                                                'inner', left_index = True, right_index = True) for ids in  ids_pair]
        data_total = pd.concat(merged_data, axis = 0)

        self.crosstabs = [pd.crosstab(mdata.iloc[:, 1], mdata.iloc[:, 0]).apply(compute_log_probs, axis = 1).round(3).astype(np.float16)
                          for mdata in merged_data]
        self.crosstab_total = pd.crosstab(data_total.iloc[:, 1], data_total.iloc[:, 0]).apply(compute_log_probs, axis = 1).round(3).astype(np.float16)

        return self

    def transform(self, data):
        """The precomputed histograms are joined as features to the given data set.
        
        Parameters
        ----------
        data : data that will be augmented by the crosstab feature
        
        Returns
        -------
        Transformed dataset.
        """
        feat_name = data.columns[1]
        data_merge = [pd.Series(ids, name = 'device_id').to_frame().merge(data, 'inner') for ids in self.ids_pair]
        data_ct = pd.concat([mdata.merge(crosstab, 'left', left_on = feat_name, right_index = True).drop(feat_name, axis = 1)
            for mdata, crosstab in zip(data_merge, self.crosstabs[::-1])], axis = 0)
            
        del data_merge
        
        data_ct_total = data.merge(self.crosstab_total, 'left', left_on = feat_name, right_index = True).drop(feat_name, axis = 1)
        return  [data_ct, data_ct_total]

def compute_log_probs(row):
    """
    helper function for computing regularized log probabilities
    """
    row = row + np.random.randint(1, NOISE_CT, len(row))#.apply(lambda x: max(x, MIN_COUNT_CT))

    #compute the log ratios of class probabilities and the popularity of the feature 
    row_sum = row.sum()
    row = ((row/row_sum).apply(lambda y: math.log(y) - math.log(1.0/NCLASSES))*1000).astype(np.int16)

    return row
    


ct = CrossTabEncoder().fit(device_apps_filtered, blend_ids)
print('{0} crosstab computed'.format(time.clock() - start))
[d_0,d_1] = ct.transform(device_apps_filtered.reset_index())
del device_apps_filtered
device_ct_0 = d_0.groupby('device_id').mean()
del d_0
device_ct_1 = d_1.groupby('device_id').mean()
del d_1
print('{0} crosstab joined'.format(time.clock() - start))

device_ct = device_ct_0.combine_first(device_ct_1)
del device_ct_0
del device_ct_1
stacked_device_ct = device_ct.stack().reset_index()
del device_ct
print('{0} dataframes combined'.format(time.clock() - start))




# Create bag-of-apps in character string format
# first by event
# then merge to generate larger bags by device

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
# remove duplicates(app_id)
app_ev = app_ev.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
events = events[events['device_id'].isin(dec.classes_)]
events["app_id"] = events["event_id"].map(app_ev)
events['device_id'] = dec.transform(events['device_id'])
events = events.dropna()

del app_ev

events = events[["device_id", "app_id"]]

# remove duplicates(app_id)
events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# expand to multiple rows
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("../input/phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)
pbd['device_id'] = dec.transform(pbd['device_id'])


##################
#  Train and Test
##################
print("# Generate Train and Test")

train = pd.read_csv("../input/gender_age_train.csv",
                    dtype={'device_id': np.str})
train.drop(["age", "gender"], axis=1, inplace=True)
train['device_id'] = dec.transform(train['device_id'])
train = train.sort_values('device_id')
print(train)

test = pd.read_csv("../input/gender_age_test.csv",
                   dtype={'device_id': np.str})
test["group"] = np.nan
test['device_id'] = dec.transform(test['device_id'])


split_len = len(train)

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)
device_id = test["device_id"]

# Concat
Df = pd.concat((train, test), axis=0, ignore_index=True)

Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(
    lambda x: "device_model:" + str(x))


###################
#  Concat Feature
###################
f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model
f3 = events[["device_id", "app_id"]]    # app_id
#####################
#NEW FEATURE
#####################
f4 = stacked_device_ct[['device_id', 'level_1']]
f4.columns =  ['device_id', 'feature']
#####################
del Df
f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f3.columns.values[1] = "feature"
FLS = pd.concat((f1, f2, f3, f4), axis=0, ignore_index=True)
FLS['feature'] = LabelEncoder().fit_transform(FLS["feature"])
###################
# User-Item Feature
###################
print("# User-Item-Feature")
device_ids = FLS["device_id"].unique()
feature_cs = len(FLS["feature"].unique()) + NCLASSES
data_0 = np.ones(len(FLS)-f4.shape[0])
data = pd.Series(np.hstack([data_0, stacked_device_ct.iloc[:, 2]]))

del stacked_device_ct

train_mask = FLS['device_id'].isin(train.index)
test_mask = FLS['device_id'].isin(test.index)

FLS_train = FLS[train_mask]
FLS_test= FLS[test_mask]



data_train = data[train_mask]
del FLS
train_sp = sparse.csr_matrix(
(data_train, (FLS_train['device_id'], FLS_train['feature'])), shape=(len(train.index), feature_cs))
feat_mask = train_sp.getnnz(0) > 0
train_sp = train_sp[:, feat_mask]


data_test = data[test_mask]
test_sp = sparse.csr_matrix(
(data_test, (FLS_test['device_id'], FLS_test['feature'])), shape=(len(test.index), feature_cs))
test_sp = test_sp[:, feat_mask]


##################
#      Data
##################
del data
del data_0
del data_train
del data_test
del device_ids
del y_train_0
del y_train_1
del events
del pbd
del apps_train
del apps_test
del relevant_train
del relevant_test
del train_ids_0
del train_ids_1
del blend_labels

print('data')
print(locals().keys())
#train_row = train.index
#train_sp = sparse_matrix[train_row, :]

print('testt_data')
#test_row = dec.transform(test["device_id"])
#test_sp = sparse_matrix[test_row, :]

#del sparse_matrix

print('split')
X_train, X_val, y_train, y_val = train_test_split(
    train_sp, Y, train_size=.90, random_state=10)

##################
#   Feature Sel
##################
print("# Feature Selection")
selector = SelectPercentile(f_classif, percentile=23)

selector.fit(X_train, y_train)

X_train = selector.transform(X_train)
X_val = selector.transform(X_val)

train_sp = selector.transform(train_sp)
test_sp = selector.transform(test_sp)

print("# Num of Features: ", X_train.shape[1])

##################
#  Build Model
##################

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gblinear",
    "max_depth": 6,
    "eval_metric": "mlogloss",
    "eta": 0.07,
    "silent": 1,
    "alpha": 3,
}

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 800, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)

print("# Train")
dtrain = xgb.DMatrix(train_sp, Y)
gbm = xgb.train(params, dtrain, 800, verbose_eval=True)
y_pre = gbm.predict(xgb.DMatrix(test_sp))

# Write results
result = pd.DataFrame(y_pre, columns=lable_group.classes_)
result["device_id"] = dec.inverse_transform(device_id)
result = result.set_index("device_id")
result.to_csv('fine_tune.gz', index=True,
              index_label='device_id', compression="gzip")
