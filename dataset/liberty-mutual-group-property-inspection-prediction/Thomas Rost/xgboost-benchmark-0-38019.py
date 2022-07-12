'''
This benchmark uses xgboost and early stopping to achieve a score of 0.38019
In the liberty mutual group: property inspection challenge

Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb

#load train and test 
train  = pd.read_csv('../input/train.csv', index_col=0)
test  = pd.read_csv('../input/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)


columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

def prepare_data():
    # load train data
    train    = pd.read_csv('../input/train.csv')
    test     = pd.read_csv('../input/test.csv')
    
    
    
    
    
    

    labels   = train.Hazard
    test_ind = test.ix[:,'Id']
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn import preprocessing
    df_train = train# pd.read_csv(path+"\\train.csv")
    df_test = test#pd.read_csv(path+"\\test.csv")
   #example = pd.read_csv(path+"\\sample_submission.csv")
    y = df_train['Hazard']
    X = df_train.drop(['Hazard'],axis=1)
    X.index = X.index + len(df_test)

    complete = df_test.append(X)
    var_list = [var for var in df_train.columns if var in df_test.columns]

    for var in var_list:
     if complete[var].dtypes == 'object':
            dummies = pd.get_dummies(complete[var],prefix=var)

            complete = complete.drop([var],axis=1)  
            complete = complete.join(dummies)
     else: 
          print('notnecessary')

    test = complete[:len(df_test)]
    X = complete[len(df_test):]

    return np.array(X).astype(float), labels, np.array(test).astype(float), test_ind
    
train, labels, test,test_ind  = prepare_data()
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 4
params["subsample"] = 0.65
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 10

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
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
preds1 = model.predict(xgtest)

#reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = labels[::-1]

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
preds2 = model.predict(xgtest)

#combine predictions
#since the metric only cares about relative rank we don't need to average
preds = preds1 + preds2

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark.csv')