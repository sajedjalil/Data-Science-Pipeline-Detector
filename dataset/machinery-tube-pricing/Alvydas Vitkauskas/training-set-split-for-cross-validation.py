import os
__author__ = 'xie'

#updated with some commen, 14/07/2015
#This script is created to set up a cross validation scheme to allow train/test split within the training dataset provided. 
#This allow us to benchmark our algorithm with labelled data 

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb
import time
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor



def xgb_benchmark_data():
    #data handling, take the input data, and merge them accordingly
    #this is the original data handling routine of the xgb benchark script shared by Gilberto Titericz Junior
    train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
    test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
    tube_data = pd.read_csv('../input/tube.csv')
    bill_of_materials_data = pd.read_csv('../input/bill_of_materials.csv')
    specs_data = pd.read_csv('../input/specs.csv')



    train = pd.merge(train, tube_data, on ='tube_assembly_id')
    train = pd.merge(train, bill_of_materials_data, on ='tube_assembly_id')
    test = pd.merge(test, tube_data, on ='tube_assembly_id')
    test = pd.merge(test, bill_of_materials_data, on ='tube_assembly_id')



    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month
    #train['dayofyear'] = train.quote_date.dt.dayofyear
    #train['dayofweek'] = train.quote_date.dt.dayofweek
    #train['day'] = train.quote_date.dt.day

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month
    #test['dayofyear'] = test.quote_date.dt.dayofyear
    #test['dayofweek'] = test.quote_date.dt.dayofweek
    #test['day'] = test.quote_date.dt.day

    # drop useless columns and create labels
    idx = test.id.values.astype(int)
    test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
    labels = train.cost.values

    #'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x',
    #  'end_a', 'end_x'
    #for some reason material_id cannot be converted to categorical variable
    train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

    train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    test['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    for i in range(1,9):
        column_label = 'component_id_'+str(i)
        # print(column_label)
        train[column_label].replace(np.nan,' ', regex=True, inplace= True)
        test[column_label].replace(np.nan,' ', regex=True, inplace= True)

    train.fillna(0, inplace = True)
    test.fillna(0, inplace = True)


    # convert data to numpy array
    train = np.array(train)
    test = np.array(test)

    # label encode the categorical variables
    for i in range(train.shape[1]):
        if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
            print(i,list(train[1:5,i]) + list(test[1:5,i]))
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])

    return train, test, idx, labels


# pickle data routine in case you saved the data in a local environment
def load_data(pickle_file):
    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    return  data

# xgb learner, inline with the xgb benchmark script shared by Gilberto Titericz Junior
def xgb_learning(labels, train, test):
    label_log = np.log1p(labels)
    # fit a random forest model
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.1
    params["min_child_weight"] = 6
    params["subsample"] = 0.87
    params["colsample_bytree"] = 0.50
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 7

    plst = list(params.items())

    xgtrain = xgb.DMatrix(train, label=label_log)
    xgtest = xgb.DMatrix(test)

    num_rounds = 120
    # model = xgb.train(plst, xgtrain, num_rounds)
    # preds = model.predict(xgtest)

    model = xgb.train(plst, xgtrain, num_rounds)
    preds1 = model.predict(xgtest)
    preds = np.expm1(preds1)

    # I have commented out the follownig line for fast run time 
    # preds = model.predict(xgtest)
    # n=1
    # for loop in range(n):
    #     model = xgb.train(plst, xgtrain, num_rounds)
    #     preds1 = preds1 + model.predict(xgtest)
    # preds = np.expm1( preds1/(n+1))
    return  preds

# sklearn LinearRegression
def linear_learning(labels, train, test):
    label_log=np.log1p(labels)
    linear=LinearRegression()
    model=linear.fit(train, label_log)
    preds1=model.predict(test)
    preds=np.expm1(preds1)
    return  preds

# sklearn svm regression 
def svm_learning(labels, train, test):
    label_log=np.log1p(labels)
    clf=svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.0,
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    model=clf.fit(train, label_log)

    preds1=model.predict(test)
    preds=np.expm1(preds1)
    return  preds
# sklearn random forest regression
def random_learning(labels, train, test):
    label_log=np.log1p(labels)
    clf=RandomForestRegressor(n_estimators=50, n_jobs=3)
    model=clf.fit(train, label_log)
    preds1=model.predict(test)
    preds=np.expm1(preds1)
    return  preds

if __name__ == '__main__':
    start_time=time.time()
    test_run=False
    train, test, idx, labels=xgb_benchmark_data()

    # if test run, then perform the cross validation
    if test_run:
        print("perform cross validation")
        rmse=[]
        rnd_state=np.random.RandomState(1234)
        for run in range(1, 11):
            train_i, test_i = train_test_split(np.arange(train.shape[0]), train_size = 0.8, random_state = rnd_state )
            tr_train=train[train_i]
            tr_test=train[test_i]
            tr_train_y=labels[train_i]
            tr_test_y=labels[test_i]

            # you can switch on/off each learninger as you wish by comment/uncomment
            tr_preds=xgb_learning(tr_train_y, tr_train, tr_test)
            #tr_preds=linear_learning(tr_train_y, tr_train, tr_test)
            # tr_preds=svm_learning(tr_train_y, tr_train, tr_test)
            #tr_preds=random_learning(tr_train_y, tr_train, tr_test)

            rmse_score = (np.sum((np.log1p(tr_preds)-np.log1p(tr_test_y))**2)/len(test_i))**0.5
            
            #output test score with both real value and predicted price, this allow you to have a visual understand
            #how close/far they are from each other
            
            compare=pd.DataFrame({"tr_test_id":test_i, "cost_real":tr_test_y, "cost_pred":tr_preds})
            header=["tr_test_id", "cost_real", "cost_pred"]
            compare.to_csv('compare.csv', columns=header, index=False)
            rmse.append(rmse_score)
            print ("logistic regression score for test run %i is %.6f" %(run, rmse_score))
        print ("Mean logistic regression RMSE is %.6f:" %np.mean(rmse))
    else:
        preds=xgb_learning(labels, train, test)
        preds = pd.DataFrame({"id": idx, "cost": preds})
        preds.to_csv('xgb_test.csv', index=False)

    end_time=time.time()
    duration=end_time-start_time
    print ("it takes %.3f seconds"  %(duration))