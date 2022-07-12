import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import cross_validation

train = pd.read_csv('./data/train_set.csv')
test = pd.read_csv('./data/test_set.csv')


train_labels = train.cost.values
train = train.drop(['cost'], axis = 1)

#omitted pre processing steps 

train = np.array(train)
test = np.array(test)

#omitted pre processing steps 

train = train.astype(float)
test = test.astype(float)

from sklearn import cross_validation

# number of folds
k=5

cv= cross_validation.KFold(len(train),n_folds=k,indices=False)
cv=list(cv)

#The input will your prediction followed by your ground truth 
#the output will be a string followed by your error value
def rmsle_eval(y, y0):
    
    y0=y0.get_label()    
    assert len(y) == len(y0)
    return 'error',np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
    

def cvtest(i):
    
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.1
    params["min_child_weight"] = 10
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.8
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 7
    params["nthread"] = 4
    
    plst = list(params.items())


    num_rounds=20000    

    # pass the indexes to your training and validation data
    xgtrain = xgb.DMatrix(train[cv[i][0]], label=train_labels[cv[i][0]])
    xgval=xgb.DMatrix(train[cv[i][1]],label=train_labels[cv[i][1]])
    
    # define a watch list to observe the change in error f your training and holdout data
     watchlist  = [ (xgtrain,'train'),(xgval,'eval')]

 
    model = xgb.train(plst, 
                      xgtrain, 
                      num_rounds,
                      watchlist,
                      feval=rmsle_eval,  # this is your custom evaluation function
                      early_stopping_rounds=50)   # stops 50 iterations after marginal improvements or drop in performance on your hold out set
    
    pred_train=model.predict(xgval)
    print('best ite:',model.best_iteration)
    print('best score:',model.best_score)

    res=rmsle(pred_train,train_labels[cv[i][1]])
    print(res)     
    return(res,model.best_iteration)
    
    
import time
results=np.repeat(0.0,k)
trees=np.repeat(0.0,k)  

start = time.clock()
for i in range (5):
    results[i],trees[i]=cvtest(i)

print('mean score: ', np.mean(results))
end = time.clock()
print ('runtime: ',end - start)
print(np.mean(results))
print(np.mean(trees))