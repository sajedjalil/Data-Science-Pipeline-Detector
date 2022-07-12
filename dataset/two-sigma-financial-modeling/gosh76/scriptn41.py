import kagglegym
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import GradientBoostingRegressor
env = kagglegym.make()
observation = env.reset()
traindf = observation.train
train1 = traindf.drop(['id'],axis=1)
train1 = train1.fillna(value=train1.median())
train1.fillna(value=0,inplace=True)
train2 = train1.set_index('timestamp')
trf = np.array(train2)
transformer = SelectKBest(f_regression, k=8).fit(trf[:,0:trf.shape[1]-1], trf[:,trf.shape[1]-1])
X_new = transformer.transform(trf[:,0:trf.shape[1]-1])
listy = list(trf[:,trf.shape[1]-1])
train3 = X_new[1:]
train3 = np.c_[train3,listy[0:len(listy)-1]]
target = trf[1:,trf.shape[1]-1]
print("Train length is {}".format(len(train3)))
model = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=20,subsample=0.8,criterion='friedman_mse',min_samples_split=10,min_samples_leaf=3,\
max_depth=3,random_state=69,alpha=0.9,verbose=1)
model = model.fit(train3,target)
i = 0
while True:
    i = i + 1
    print("i is {}".format(i))
    test = observation.features
    test1 = test.drop(['id','timestamp'],axis=1)
    print("Test1 length is {}".format(len(test1)))
    test2 = test1.fillna(value=test1.median())
    test2.fillna(value=0,inplace=True)
    tes = np.array(test2)
    tes1 = transformer.transform(tes)
    tes2 = np.c_[tes1,np.zeros(len(tes1))]
    pred = model.predict(tes2)
    observation.target.loc[:,'y'] = list(pred)
    observation, reward, done, info = env.step(observation.target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
