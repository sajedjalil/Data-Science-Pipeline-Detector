'''
Author: Mark Chatters
Title:  XGBRegressor with Bagging and "BRAGGing" (Bootstrap Robust AGGregation)

Acknowledge Jared Turkewitz for transforming Loss using log(x+200)
'''
import pandas as pd
import numpy as np
import csv as csv
import datetime
import sys
from shutil import copyfile
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

#turn off the warnings!
pd.options.mode.chained_assignment = None  # default='warn'

#shift value
SHIFT = 200

#script uses this to name the output file and the copy of the script that runs it
global_submission_id = 'xgb3_'+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def df_cleaner(df_train, df_test):
    
    #keep the number of training and test rows
    ntrain = df_train.shape[0]
    ntest =  df_test.shape[0]
    
    # create df which is a concat of the 2
    df = pd.concat((df_train, df_test)).reset_index(drop=True)
        
    #create a new dataframe with one hot encode the cat values
    df_cat = pd.get_dummies(df.filter(regex="^cat"))
        
    #scale the "cont" values (inplace)
    scale = StandardScaler()
    df[['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14']] = scale.fit_transform(df[['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14']].as_matrix())
    
    #concatenate id, loss and the cont values from the original df + the df_cat for the one hot encoded values
    df = pd.concat([df[['id','loss','cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14']], df_cat], axis=1)

    # split back into test and training
    df_out_train = df.iloc[:ntrain, :]
    df_out_test  = df.iloc[ntrain:, :]

    #drop columns in the training set where all zeros
    df_out_train = df_out_train.loc[:, (df_out_train != 0).any(axis=0)]
    
    #get data columns
    data_columns = list(df_out_train)
    data_columns.remove('id')
    data_columns.remove('loss')
    
    return df_out_train, df_out_test, data_columns

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds)-SHIFT, np.exp(labels)-SHIFT)

def main():

    # Read in the training data.
    train = pd.read_csv('../input/train.csv', header=0)
    
    # Read in the training data.
    test = pd.read_csv('../input/test.csv', header=0)
    
    # Clean the Training dataset up (one hot encoding and scaling)
    train, test, features = df_cleaner(train,test)

    x_test  = test[:][features]
    
    #Create new column (transform loss using log(x+200)) - works better for some spooky reason...
    train['loss_logshift'] = np.log(train['loss'] + SHIFT)

    number_of_bagging_iterations = 10
    max_number_of_rounds         = 1500
    early_stopping_rounds        = 20
    
    #the results of each iteration get added to work_dataframe (one column per iteration)
    work_dataframe           = test[['id']]
    
    for i in range(0, number_of_bagging_iterations):
        
        #Use modulus on the id to split the data into batches
        train_slice = train[train.id % number_of_bagging_iterations != i]
        val_slice   = train[train.id % number_of_bagging_iterations  == i]
    
        #build the x and y training features and output variable
        x_train = train_slice[:][features]
        y_train = train_slice[:]['loss_logshift']
    
        #build the x and y validation features and output variable
        x_val = val_slice[:][features]
        y_val = val_slice[:]['loss_logshift']
    
        #create the model
        model = xgb.XGBRegressor(max_depth=12,colsample_bytree=0.5, min_child_weight=1,subsample=0.8,gamma=1,n_estimators=max_number_of_rounds, learning_rate=0.1)

        #train the model
        model.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_set=[(x_train, y_train), (x_val, y_val)],eval_metric=evalerror)

        #predict the test results
        this_iteration_predictions = model.predict(x_test).astype(float)
        
        #Add the predictions as a new column to work_dataframe (after changing back the original target value)
        temp_series = pd.Series(np.exp(this_iteration_predictions) - SHIFT)
        work_dataframe['round'+str(i)] = temp_series.values


    #Work Dataframe now contains results of each iteration
    #create new columnds with the mean and median of the results
    work_dataframe['mean_values']     = work_dataframe.filter(regex="^round").mean(axis=1)
    work_dataframe['median_values']   = work_dataframe.filter(regex="^round").median(axis=1)
    work_dataframe['std_values']      = work_dataframe.filter(regex="^round").std(axis=1)

    #save the dataframe for additional analysis
    work_dataframe.to_csv('work_dataframe.csv',index=False,float_format='%.2f')
    
    #create 2 submissions files- one using mean(bagging) the other median(bragging)
    outfile_name = global_submission_id+'_mean.csv'
    work_dataframe[['id','mean_values']].to_csv(outfile_name,index=False,float_format='%.2f',header=['id','loss'])
        
    outfile_name = global_submission_id+'_median.csv'
    work_dataframe[['id','median_values']].to_csv(outfile_name,index=False,float_format='%.2f',header=['id','loss'])

    copyfile(sys.argv[0], global_submission_id + '_' + sys.argv[0])

if __name__ == '__main__':
    main()