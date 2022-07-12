import numpy as np 
import pandas as pd 
import os
import xgboost as xgb
        
def timeEngineering(data, col_names):
   for col_name in col_names:
         data[col_name] = pd.to_datetime(data[col_name], errors='coerce')
         data[col_name + '_year'] = data[col_name].dt.year
         data[col_name + '_month'] = data[col_name].dt.month
         data[col_name + '_week'] = data[col_name].dt.week
         data[col_name + '_day'] = data[col_name].dt.day
         data[col_name + '_hour'] = data[col_name].dt.hour
   return data

def dataForML(data_train, data_test):
   #data_train = data_train[0:100000]
   #data_train = data_train
   #data_test = data_test
   print ('feature engineering...')
   colnames = ['date_time', 'srch_ci', 'srch_co']
   data_train = timeEngineering(data_train, colnames)
   data_test = timeEngineering(data_test, colnames)
   features_test = set(data_train.columns)
   features_train = set(data_test.columns)
   common_features = list(set(features_test).intersection(features_train)-set(['date_time', 'srch_ci', 'srch_co']))
   #print common_features
   data_train = data_train.fillna(0)
   data_test = data_test.fillna(0)
   data_train_x = data_train[common_features]
   data_train_y = data_train['hotel_cluster']
   data_test_x = data_test[common_features]
   data_test_y = []

   data = dict(x_train=data_train_x,
               x_test=data_test_x,
               y_train=data_train_y,
               y_test=data_test_y)
   print ('finished feature engineering')
   return data

def predict(data):
   classes = dict(zip(list(set(data['y_train'])),range(len(list(set(data['y_train']))))))
   #print (list(data['x_train'].columns))
   xg_train = xgb.DMatrix(data['x_train'], label=data['y_train'])
   xg_test = xgb.DMatrix(data['x_test'], label=data['y_test'])
   print ('matrices for training the model created')
   # setup parameters for xgboost
   param = {}
   # use softprob multi-class classification
   param['objective'] = 'multi:softprob'
   param['eta'] = 0.1
   #param['eta'] = 0.8
   param['max_depth'] = 6
   param['silent'] = 1
   param['nthread'] = 4
   param['num_class'] = len(classes)
   print ('total classes:', len(classes))
   #param['max_delta_step'] = 1
   num_round = 10
   print ('building a model...')
   bst = xgb.train(param, xg_train, num_round)
   print ('model built')
   print ('predicting...')
   pred = bst.predict(xg_test)
   print ('prediction done')
   #format the prediction
   df_pred = pd.DataFrame(pred)
   df_pred.columns = classes
   return df_pred

def formatForSubmission(df_test, df_pred): 
   #df_pred = df_pred.reindex_axis(sorted(df_pred.columns), axis=1)
   df_pred['0_'], df_pred['1_'], df_pred['2_'], df_pred['3_'], df_pred['4_'] = zip(*df_pred.apply(lambda x: df_pred.columns[x.argsort()[::-1][:5]].tolist(), axis=1))
   df_pred = df_pred[['0_','1_','2_', '3_', '4_']]
   df_submit = pd.DataFrame()
   df_submit['hotel_cluster'] = df_pred[['0_','1_','2_','3_','4_']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
   df_submit = pd.concat([df_test['id'], df_submit['hotel_cluster']], axis=1)
   return df_submit

def runML(train, test):
    index = 0
    for chunk in train:
        #filename = 'data/training_' + str(index) + '.csv'
        #df = pd.read_csv('data/training_' + str(index) + '.csv')
        #chunk = chunk.sort_values(by=['date_time'], ascending=True)
        data = dataForML(chunk, test)
        df_pred = predict(data)
        df_submit = formatForSubmission(test, df_pred)
        print ('saving the results...')
        df_submit.to_csv('submission.csv', index=False)
        index = index + 1
        break
    
train = pd.read_csv('../input/train.csv', chunksize=100000)    
test = pd.read_csv('../input/test.csv')
runML(train, test)