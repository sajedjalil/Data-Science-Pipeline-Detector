'''


Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.cluster import KMeans


#load train and test
print("reading")
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)
print("DONE reading")

labels = train.Hazard

print("dropping train")
train.drop('Hazard', axis=1, inplace=True)
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)
print("dropping test")
test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)
print("DONE dropping")

columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
print("pre-processing")
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])


print("DONE pre-processing")

print("clustering")
km = KMeans(init='k-means++', n_clusters=6, n_init=10, tol = 0.0001, verbose = 1)
km.fit(train)
train_clust = km.labels_
test_clust = km.predict(test)

train = np.column_stack((train, train_clust))
test = np.column_stack((test, test_clust))
print("DONE clustering")

train = train.astype(float)
test = test.astype(float)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 8

plst = list(params.items())

#Using 5000 rows for early stopping. 
offset = 5000

num_rounds = 2000
xgtest = xgb.DMatrix(test)

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
preds1 = model.predict(xgtest)


#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = np.log(labels[::-1])

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
preds2 = model.predict(xgtest)


#combine predictions
#since the metric only cares about relative rank we don't need to average
preds = preds1*2.6+ preds2*7.4

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark_kmeans.csv')