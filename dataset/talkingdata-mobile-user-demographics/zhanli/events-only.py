
# Bag of apps categories
# Bag of labels categories
# Include phone brand and model device

print("Initialize libraries")

import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn import ensemble
from sklearn.decomposition import PCA
import os
import gc
from scipy import sparse
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split


#------------------------------------------------- Write functions ----------------------------------------

def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0
            


#------------------------------------------------ Read data from source files ------------------------------------

seed = 7
np.random.seed(seed)
datadir = '../input'

print("### ----- PART 1 ----- ###")

# Data - Events data
# Bag of apps
print("# Read app events")
app_events = pd.read_csv(os.path.join(datadir,'app_events.csv'), dtype={'device_id' : np.str})
app_events.head(5)
app_events.info()
#print(rstr(app_events))

# remove duplicates(app_id)
app_events= app_events.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))
app_events.head(5)

print("# Read Events")
events = pd.read_csv(os.path.join(datadir,'events.csv'), dtype={'device_id': np.str})
events.head(5)
events["app_id"] = events["event_id"].map(app_events)
events = events.dropna()
del app_events

events = events[["device_id", "app_id"]]
events.info()
# 1Gb reduced to 34 Mb

# remove duplicates(app_id)
events.loc[:,"device_id"].value_counts(ascending=True)

events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# expand to multiple rows
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']
events.head(5)
f3 = events[["device_id", "app_id"]]    # app_id

print("#Part1 formed")

##################
#   App labels
##################

print("### ----- PART 2 ----- ###")

print("# Read App labels")
app_labels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
label_cat = pd.read_csv(os.path.join(datadir,'label_categories.csv'))
app_labels.info()
label_cat.info()
label_cat=label_cat[['label_id','category']]

app_labels=app_labels.merge(label_cat,on='label_id',how='left')
app_labels.head(3)
events.head(3)
#app_labels = app_labels.loc[app_labels.smaller_cat != "unknown_unknown"]

#app_labels = app_labels.groupby("app_id")["category"].apply(
#    lambda x: ";".join(set("app_cat:" + str(s) for s in x)))
app_labels = app_labels.groupby(["app_id","category"]).agg('size').reset_index()
app_labels = app_labels[['app_id','category']]
print("# App labels done")


# Remove "app_id:" from column
print("## Handling events data for merging with app lables")
events['app_id'] = events['app_id'].map(lambda x : x.lstrip('app_id:'))
events['app_id'] = events['app_id'].astype(str)
app_labels['app_id'] = app_labels['app_id'].astype(str)
app_labels.info()

print("## Merge")

events= pd.merge(events, app_labels, on = 'app_id',how='left').astype(str)
#events['smaller_cat'].unique()

# expand to multiple rows
print("#Expand to multiple rows")
#events= pd.concat([pd.Series(row['device_id'], row['category'].split(';'))
#                    for _, row in events.iterrows()]).reset_index()
#events.columns = ['app_cat', 'device_id']
#events.head(5)
#print(events.info())

events= events.groupby(["device_id","category"]).agg('size').reset_index()
events= events[['device_id','category']]
events.head(10)
print("# App labels done")

f5 = events[["device_id", "category"]]    # app_id
# Can % total share be included as well?
print("# App category part formed")

##################
#   Phone Brand
##################
print("### ----- PART 3 ----- ###")

print("# Read Phone Brand")
pbd = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'),
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")

dd = events.device_id.unique()

#train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'))
#test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'))

# seperate data based on weather they have events
#train_events_index = train.device_id.intersection(dd)

#gatrain_No_event = gatrain[~gatrain.index.isin(gatrain_events_index)]

train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),dtype={'device_id': np.str})
train= train[train.device_id.isin(dd)]
train.drop(["age", "gender"], axis=1, inplace=True)





test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),dtype={'device_id': np.str})
test["group"] = np.nan
test= test[test.device_id.isin(dd)]
#test_events_index = test.device_id.intersection(dd)




split_len = len(train)

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)
device_id = test["device_id"]
device_id.reset_index(drop=True,inplace=True)

# Concat
Df = pd.concat((train, test), axis=0, ignore_index=True)

print("### ----- PART 4 ----- ###")

Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(
    lambda x: "device_model:" + str(x))


###################
#  Concat Feature
###################

print("# Concat all features")

f1 = Df[["device_id", "phone_brand"]]   # phone_brand
f2 = Df[["device_id", "device_model"]]  # device_model

events = None
Df = None

f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f5.columns.values[1] = "feature"
f3.columns.values[1] = "feature"

FLS = pd.concat((f1, f2, f3, f5), axis=0, ignore_index=True)

FLS.info()

###################
# User-Item Feature
###################
print("# User-Item-Feature")

device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()

data = np.ones(len(FLS))
len(data)

dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))
sparse_matrix.shape
sys.getsizeof(sparse_matrix)

sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]
print("# Sparse matrix done")

del FLS
del data
f1 = [1]
f5 = [1]
f2 = [1]
f3 = [1]

events = [1]

##################
#      Data
##################

print("# Split data")
train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row, :]

test_row = dec.transform(test["device_id"])
test_sp = sparse_matrix[test_row, :]

X_train, X_val, y_train, y_val = train_test_split(
    train_sp, Y, train_size=.98, random_state=10)

##################
#   Feature Sel
##################
print("# Feature Selection")
#selector = SelectPercentile(f_classif, percentile=53)

#selector.fit(X_train, y_train)
#X_train.shape
#X_train = selector.transform(X_train)
#X_train.shape
#X_val = selector.transform(X_val)
#X_val.shape

# Selection using chi-square
# selector = SelectKBest(chi2, k=11155).fit(X_train, y_train)
# X_train.shape
# X_train = selector.transform(X_train)
# X_train.shape
# X_val = selector.transform(X_val)
# X_val.shape

print("# Num of Features: ", X_train.shape[1])
print("# Num of data: ", X_train.shape[0])



##################
#  Build Model
##################


#act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model

model=baseline_model()

fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=15,
                         samples_per_epoch=22824,
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )

# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
print('logloss val {}'.format(log_loss(y_val, scores_val)))

print("# Final prediction")
scores = model.predict_generator(generator=batch_generatorp(test_sp, 800, False), val_samples=test_sp.shape[0])
result = pd.DataFrame(scores , columns=lable_group.classes_)
result["device_id"] = device_id
print(result.head(1))
result = result.set_index("device_id")

#result.to_csv('./sub_bagofapps7_keras_10_50_pt2_10epoch.csv', index=True, index_label='device_id')
#Drop out 0.2
#Validation 2.3017
result.to_csv('keras_with_event.csv', index=True, index_label='device_id')


print("Done")
