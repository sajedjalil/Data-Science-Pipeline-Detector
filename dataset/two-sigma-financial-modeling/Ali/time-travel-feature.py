# Ali Ajouz

import kagglegym
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import math
import gc

env = kagglegym.make()
o = env.reset()
gc.collect()

##### general 
def _reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)


#### vars to use
excl = ["id", "sample", "y", "timestamp"]
selected=["technical_20","technical_30","technical_21", "technical_2","technical_17","technical_19","technical_11","technical_12"]
col = [c for c in selected if c not in excl]

### mini data
train = o.train.loc[:,col]

### target
y=o.train.loc[:,"y"]

### id 
train["id"]=o.train.loc[:,"id"]

###
ymedian = dict(o.train.groupby(["id"])["y"].median())

### 
o.train=[]
gc.collect()

### new var
train["next_technical_30_technical_20"] = train.groupby('id')["technical_20"].shift(-1) - train.groupby('id')["technical_30"].shift(-1)
train = train.drop('id', 1)


### save medians
d_median= train.median(axis=0)

#### Fill NA
train.fillna(d_median,inplace=True)

model_20_30 = LinearRegression(n_jobs=-1)
model_20_30.fit(np.array(train.loc[:,selected]), train.loc[:,"next_technical_30_technical_20"])
#model_20_30.fit(train.loc[:1000,selected], train.loc[:1000,"next_technical_30_technical_20"])
train["next_technical_30_technical_20"]=model_20_30.predict(np.array(train.loc[:,selected]))

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.086093
high_y_cut = 0.093497
y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)




#### model 2
model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(train.loc[y_is_within_cut,"next_technical_30_technical_20"]).reshape(-1, 1), y[y_is_within_cut])


#####
i = 0; reward_=[];kk=0
models_names=["based_on_train", "based_on_test"]
gc.collect()

##### 
train = []
gc.collect()


#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.97 * y + 0.03 * ymedian[id] if id in ymedian else y

while True:
    
    ## test data
    test = o.features[col]
    
    ### (id, target) data
    pred = o.target
    
    ### Fill Na
    test.fillna(d_median, inplace=True)

    ### predict next_technical_30_technical_20
    test["next_technical_30_technical_20"] = model_20_30.predict(  np.array(test.loc[:, selected]) )
    
    ### re-train based on test 
    test_model_20_30 = LinearRegression(n_jobs=-1)
    test_model_20_30.fit(test.loc[:,selected], test.loc[:,"next_technical_30_technical_20"])
    test_next_technical_30_technical_20 = test_model_20_30.predict(  np.array(test.loc[:, selected]) )
    
    ### check the diff
    # r_diff_pred= _reward(test["next_technical_30_technical_20"], test_next_technical_30_technical_20)

    ## choose based on kk
    test["next_technical_30_technical_20"]= np.abs(kk)*test["next_technical_30_technical_20"] + np.abs(1-kk)*test_next_technical_30_technical_20

    ### predict y
    pred['y']  = model2.predict(  np.array(test["next_technical_30_technical_20"] ).reshape(-1, 1)  )

    
    ### 
    test=[]
    pred['y'] = pred.apply(get_weighted_y, axis = 1)
    
    
    #### env.step
    o, reward, done, info = env.step(pred[['id','y']])
    
    reward_.append(reward)
    #if i % 100 == 0:
    print(reward, np.mean(np.array(reward_)))

   ##### set kk
    try:
        if reward < -0.15:
            print("switching models")
            kk = np.abs(1-kk)
            print("current model is {}".format(models_names[kk]))
    except: pass

    gc.collect()
    
    i += 1
    if done:
        print("el fin ...", info["public_score"])
        break