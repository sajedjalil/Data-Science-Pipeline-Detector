import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

                                 
# print('preprocessing...')
# from sklearn import preprocessing 
# for f in traindf.columns: 
#     if traindf[f].dtype=='object': 
#         lbl = preprocessing.LabelEncoder() 
#         lbl.fit(list(traindf[f].values)) 
#         traindf[f] = lbl.transform(list(traindf[f].values))

# for f in testdf.columns: 
#     if testdf[f].dtype=='object': 
#         lbl = preprocessing.LabelEncoder() 
#         lbl.fit(list(testdf[f].values)) 
#         test[f] = lbl.transform(list(testdf[f].values))

# traindf.fillna((-999), inplace=True) 
# testdf.fillna((-999), inplace=True)

# traindf=np.array(traindf) 
# testdf=np.array(testdf) 
# traindf= traindf.astype(float) 
# testdf = testdf.astype(float)

print("preprocessing...")
#log transform the target:
observation.train["y"] = np.log1p(observation.train["y"])

#log transform skewed numeric features:
# numeric_feats = observation.dtypes[observation.dtypes != "object"].index

# skewed_feats = observation.train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index

# observation[skewed_feats] = np.log1p(observation[skewed_feats])

# observation = pd.get_dummies(observation)
observation = observation.train.fillna(observation.train.mean())


print('traintestsplit...')


traindf, testdf = train_test_split(observation.train.drop(axis=1, labels=["id", "timestamp"]).dropna(),
                                  train_size=0.8,
                                  test_size=0.2)
#if numpy array
# Y_train = traindf[-1]
# X_train = np.delete(traindf,1,0)

# Y_test = testdf[-1]
# X_test = np.delete(testdf,1,0)

#if df
Y_train = traindf["y"]
X_train = traindf.drop(axis=1, labels=["y"])

Y_test = testdf["y"]
X_test = testdf.drop(axis=1, labels=["y"])

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'reg:linear' }
num_round = 10

print ('training')
bst = xgb.train(param, X_train, num_round)
# make prediction
preds = bst.predict(X_test)

print (preds)
observation.target.y = preds

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

while True:
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break