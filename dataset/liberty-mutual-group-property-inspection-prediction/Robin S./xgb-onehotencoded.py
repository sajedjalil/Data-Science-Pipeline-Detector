'''
Some changes to the XGBoost benchmark by Soutik

Author: Robin
'''

import pandas as pd
import numpy as np 


import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


def one_hot_dataframe(data, cols, replace=False):
    """
    Performs a one hot encoding of the categorical columns (given by argument 'cols')
    """
    vec = DictVectorizer()
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    
    return (data, vecData, vec)


def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 9
    
    
    plst = list(params.items())

    #Using 5000 rows for early stopping. 
    offset = 4000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices 
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    print("Train first model")
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    print("Train second model")
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    preds = preds1*1.4 + preds2*8.6
    return preds










if __name__ == '__main__':
    #load train and test 
    train  = pd.read_csv('../input/train.csv', index_col=0)
    test  = pd.read_csv('../input/test.csv', index_col=0)
    
    
    labels = train.Hazard
    train.drop('Hazard', axis=1, inplace=True)

    
    preprocessing_columns = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 
                             'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 
                             'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']

    columns = train.columns
    test_ind = test.index
    
    
    #Perform encoding
    
    #print(train.columns) #To show the encoding
    train, _, _ = one_hot_dataframe(train, preprocessing_columns, replace=True)
    test, _, _ = one_hot_dataframe(test, preprocessing_columns, replace=True)
    #print(train.columns)
    
    train_s = np.array(train)
    test_s = np.array(test)
    
    train_s = train_s.astype(float)
    test_s = test_s.astype(float)
    
    
    preds = xgboost_pred(train_s,labels,test_s)

    
    #generate solution
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    print('Write output')
    preds.to_csv('xgboost_onehot.csv')
    print('Finished')
                    