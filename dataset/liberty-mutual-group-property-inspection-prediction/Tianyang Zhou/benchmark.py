'''


Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

from datetime import *
import pandas as pd
import numpy as np 
import random
import itertools
from sklearn import preprocessing
import xgboost as xgb

def gini(solution, submission):                                                 
    df = sorted(zip(solution, submission), key=lambda x : (x[1], x[0]),  reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]                
    totalPos = np.sum([x[0] for x in df])                                       
    cumPosFound = np.cumsum([x[0] for x in df])                                     
    Lorentz = [float(x)/totalPos for x in cumPosFound]                          
    Gini = [l - r for l, r in zip(Lorentz, random)]                             
    return np.sum(Gini)                                                         


def normalized_gini(solution, submission):                                      
    normalized_gini = gini(solution, submission)/gini(solution, solution)       
    return normalized_gini                                                      

def param_table_gen(params_list):
    for param in itertools.product(*params_list.values()):
        yield dict(zip(params_list.keys(), param))

start_time = datetime.now()

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)


columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)
random.seed(20150720)
samp_id = np.array([random.randint(0,4) for i in range(train.shape[0])])
print([sum(samp_id==i) for i in range(5)])

# label encode the categorical variables
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)




#Using 5000 rows for early stopping. 


xgtest = xgb.DMatrix(test)

#mode = "tune" or "prod" or "tune_prod"
mode = "tune_prod"

if mode=="tune":
    results_all_output=[]
    
    params_list = {}
    params_list["objective"] = ["reg:linear"]
    params_list["eta"] = [0.01]
    params_list["min_child_weight"] = [5]
    params_list["subsample"] = [0.5]
    params_list["colsample_bytree"] = [0.9]
    params_list["scale_pos_weight"] = [1.0]
    params_list["silent"] = [1]
    params_list["max_depth"] = [6]
    
    for params_use in param_table_gen(params_list):
        plst_use = list(params_use.items())
    
        num_rounds_list = np.array([100,200,300,400,500,600])
        num_rounds_use = np.max(num_rounds_list)
        
        results_all = {}
        for num_rounds in num_rounds_list: results_all[num_rounds] = []
        for samp_id_block in range(5):
            build_ind = np.array([samp_id[i]==samp_id_block for i in range(len(samp_id))])
            valid_ind = np.array([samp_id[i]!=samp_id_block for i in range(len(samp_id))])
            #create a train and validation dmatrices 
            xgtrain = xgb.DMatrix(train[build_ind,:], label=labels[build_ind])
            xgval = xgb.DMatrix(train[valid_ind,:], label=labels[valid_ind])
            
            #train using early stopping and predict
            watchlist = [(xgtrain, 'train'),(xgval, 'val')]
            # model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
            model = xgb.train(plst_use, xgtrain, num_rounds_use, watchlist, early_stopping_rounds=num_rounds_use)
            for num_rounds in num_rounds_list:
                preds1 = model.predict(xgval, ntree_limit = int(num_rounds))
                results_all[num_rounds] += [normalized_gini(solution=labels[valid_ind], submission=preds1)]
            #print(results_all)
        results_all_output += [{'params':params_use, 'results':results_all}]
    
    for results_all in results_all_output:
        print(sorted(results_all['params']))
        for num_rounds in num_rounds_list:
            results = results_all['results'][num_rounds]
            results = np.array(results)
            print(num_rounds, results, np.mean(results), np.std(results)/np.sqrt(5))
    
elif mode=="prod":
    
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 5
    params["subsample"] = 0.5
    params["colsample_bytree"] = 0.9
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 6
    
    num_rounds = 500
    
    plst = list(params.items())
    
    xgtrain = xgb.DMatrix(train, label=labels)

    model = xgb.train(plst, xgtrain, num_rounds)
    preds = model.predict(xgtest)
    
    #generate solution
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    preds.to_csv('xgboost_depth6_colsamp09.csv')
    
elif mode=="tune_prod":
    
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 5
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.9
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8
    
    num_rounds = 500
    
    plst = list(params.items())
    preds = None
    results = []
    
    for samp_id_block in range(5):
        build_ind = np.array([samp_id[i]==samp_id_block for i in range(len(samp_id))])
        valid_ind = np.array([samp_id[i]!=samp_id_block for i in range(len(samp_id))])
        
        xgtrain = xgb.DMatrix(train[build_ind,:], label=labels[build_ind])
        xgval = xgb.DMatrix(train[valid_ind,:], label=labels[valid_ind])
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
        new_preds_test = model.predict(xgtest)
        preds = new_preds_test if preds is None else preds+new_preds_test/5
        results += [normalized_gini(solution=labels[valid_ind], submission=model.predict(xgval))]
        
    print(plst)
    print(results, np.mean(results), np.std(results)/np.sqrt(5))
    
    #generate solution
    preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
    preds = preds.set_index('Id')
    preds.to_csv('xgb_d8_samp08X09_tune2.csv')    

print("total time", datetime.now()-start_time)