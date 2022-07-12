#
# Solution: A neural network with one hidden later
#   This was largely an learning exercise to apply MLP to a regression problem -
#   Categorical features are converted to boolean flags (one-hot encoding) 
#   Result is comparable to random forest baseline - if I had more time I would have 
#   tried feature combinations, as well as building a separate classification model 
#   to detect high dollar claims (>$20k, top 1%) separately
#   
#

import pandas as pd
import numpy as np


from math import *
from sklearn.neural_network.multilayer_perceptron import MLPRegressor

TRAIN_FILENAME="../input/train.csv"
TEST_FILENAME="../input/test.csv"

#transform target
mapv   = np.vectorize(lambda x:log(x))
unmapv = np.vectorize(lambda x:exp(max(min(x,12),-12)))

print("Reading file %s ..."%(TRAIN_FILENAME))
train=pd.read_csv(TRAIN_FILENAME)
test =pd.read_csv(TEST_FILENAME)


# cat1 to cat116, cont1 to cont14
features         =[f for f in train.columns if f not in ["id","loss"]]
cont_features    =[f for f in features if f.startswith("cont")]
cat_features     =[f for f in features if f.startswith("cat")]

## convert categoricals with 2 values (A/B) to boolean flags 
cat_features_2vals=[c for c in cat_features if len(train[c].unique())==2]
for c in cat_features_2vals: 
  boolname=c.replace("cat","bool") # cat2->bool2
  train[boolname]=(train[c]=="A").astype(int) 
  test[boolname] =(test[c]=="A").astype(int)
boolean_features=[f for f in train.columns if f.startswith("bool")]

## apply one-hot encoding for other categoricals
cat_features_3plus=[c for c in cat_features if len(train[c].unique())>=3]
#train:
df2=pd.get_dummies(train[cat_features_3plus])
train=pd.concat([train,df2],axis=1)

#test:
df3=pd.get_dummies(test[cat_features_3plus])
test=pd.concat([test,df3],axis=1)

# save list of dummy flags in both test and train 
dummy_features=[f for f in df2.columns if f in df3.columns]
del df2,df3


## cat2="A"  is equivalent to cat101="A"
boolean_features.remove("bool2")

model_features = cont_features+dummy_features+boolean_features

train_X = train[model_features]
test_X  = test[model_features]

# transform Y
train_Y = pd.Series(map(mapv,train["loss"]))

print("Training...")
model = MLPRegressor(solver='sgd', alpha=0.00001, hidden_layer_sizes=(50), random_state=1,verbose=1,learning_rate="adaptive",max_iter=200,momentum=0.001)
model.fit(train_X,train_Y)

print("trained MLP  model:",model)

test_Yhat=model.predict(test_X)

## unmap log transform 
test_Yhat=pd.Series(map(unmapv,test_Yhat))

# write output
OUT=open("submit_v4.txt","w")
OUT.write("id,loss\n")
for idx,val in enumerate(test_Yhat):
  OUT.write("%d,%.2f\n"%(test["id"][idx],val))
OUT.close()
