"""
Facebook Challenge 5 - Predicting Check-in

Notebook 4: Improving the models

A number of ways can be used to further improve the accuracy of the machine learning models. Here in this notebook, two way to improve the model is outlined.

Majority Voting/averaging Ensembles

Majority Voting or Averaging ensembles involves creating a weighted sum of the probability of each class that is predicted by the model for each data point to return a scaled probability. Bagging works by the fact that all training and predictions are statistical, and that each model being trained on a different set of data or different technique would predict slightly differently, so by combining the probabilities predicted by the multiple models, the accuracy of the prediction improves. The script below shows how the XGB and kNN models are combined to obtain an improved result. 

2) Redoing models of low map@3 score

By collecting the map@3 data per grid, we can further analysis the places in the models that hinders the accuracy. One thing that is immediately observed is that despite most grids have a similar map@3 of ~0.67 for XGB and ~0.60 for kNN (validation scores), there are grids which return very low map@3 values. This can be improved by using a different data split in the same grid. It was found that (not shown here), that the validation score can be improved by ~0.06 by rerunning these the sub-optimal grids.

"""

import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def KNN_Prep(X_grid, X, weights):
    X.x = (X.x - np.mean(X_grid.x))/np.std(X_grid.x)
    X.y = (X.y - np.mean(X_grid.y))/np.std(X_grid.y)
    X.hours = (X.hours - np.mean(X_grid.hours))/np.std(X_grid.hours)
    X.time_to_midnight = (X.time_to_midnight  - np.mean(X_grid.time_to_midnight))/np.std(X_grid.time_to_midnight)
    X.day_of_week = (X.day_of_week  - np.mean(X_grid.day_of_week))/np.std(X_grid.day_of_week)
    X.days_to_sunday = (X.days_to_sunday  - np.mean(X_grid.days_to_sunday))/np.std(X_grid.days_to_sunday)
    X.month = (X.month  - np.mean(X_grid.month))/np.std(X_grid.month)
    X.accuracy = np.log10(X.accuracy)
    
    X.x = weights[0]*X.x
    X.y = weights[1]*X.y
    X.hours = weights[2]*X.hours
    X.time_to_midnight = weights[2]*X.time_to_midnight
    X.day_of_week = weights[3]*X.day_of_week
    X.days_to_sunday = weights[3]*X.days_to_sunday
    X.month = X.month*weights[4]
    X.accuracy = X.accuracy*weights[5]

def predict_df(Y_val, pred_labels, xgb_true = True, test = False):
    if not test:    
        compare = pd.DataFrame(columns = ['actual', 'predict1', 'predict2', 'predict3'])
        compare.actual = Y_val
        if xgb_true:
            compare.predict1 = pred_labels[:,2]
            compare.predict2 = pred_labels[:,1]
            compare.predict3 = pred_labels[:,0]
        
        else:
            compare.predict1 = pred_labels.transpose()[0,:]
            compare.predict2 = pred_labels.transpose()[1,:]
            compare.predict3 = pred_labels.transpose()[2,:]
    
        compare['map3'] = ((compare.actual == compare.predict1)/1.0 + (compare.actual == compare.predict2)/2.0 + (compare.actual == compare.predict3)/3.0)
        mpk_3 = compare.map3.mean()
        print('mpk@3 is ', mpk_3)
        return compare, mpk_3
        
    else:
        compare = pd.DataFrame(columns = ['row_id', 'predict1', 'predict2', 'predict3'])
        compare.row_id = Y_val
        if xgb_true:
            compare.predict1 = pred_labels[:,2]
            compare.predict2 = pred_labels[:,1]
            compare.predict3 = pred_labels[:,0]
        
        else:
            compare.predict1 = pred_labels.transpose()[0,:]
            compare.predict2 = pred_labels.transpose()[1,:]
            compare.predict3 = pred_labels.transpose()[2,:]
        
        compare_str = pd.DataFrame(compare, dtype = str)
        compare_str['place_id'] = compare_str.predict1 + ' ' + compare_str.predict2 + ' ' + compare_str.predict3
        return compare_str, 0
        
        
# Load in training and test data
training_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#Create Empty submission
submission_combine = pd.DataFrame(columns = ['row_id', 'place_id'])

#Process time features
training_data['hours'] = (np.mod(training_data.time, (60*24)))/60.0
training_data['time_to_midnight'] = np.minimum(abs(training_data.hours - 0), abs(training_data.hours - 24))
training_data['day_of_week'] = np.mod(training_data.time, (60*24*7))/(60*24)
training_data['days_to_sunday'] = np.minimum(abs(training_data.day_of_week - 0), abs(training_data.day_of_week - 7))
training_data['month'] = np.mod(np.floor(np.divide(training_data.time, (60*24*7*4))),12)


test_data['hours'] = (np.mod(test_data.time, (60*24)))/60.0
test_data['time_to_midnight'] = np.minimum(abs(test_data.hours - 0), abs(test_data.hours - 24))
test_data['day_of_week'] = np.mod(test_data.time, (60*24*7))/(60*24)
test_data['days_to_sunday'] = np.minimum(abs(test_data.day_of_week - 0), abs(test_data.day_of_week - 7))
test_data['month'] = np.mod(np.floor(np.divide(test_data.time, (60*24*7*4))),12)

print('time columns created')

# Drop 'Time' 

training_data = training_data.drop('time', 1)
test_data = test_data.drop('time', 1)


for x in range(0,1000,25):
    for y in range(0,1000,25): 
    #Select a grid of size 0.25*0.25
        print('training model at ', x , ',', y)
        
        #Create query string to select the correct data
        start_x = x/100
        start_y = y/100
        end_x = start_x + 0.25    
        end_y = start_y + 0.25
        if (end_x < 10) and (end_y < 10):
            query_string = 'x >= ' + str(start_x) + ' and x < ' + str(end_x) + ' and y >= ' + str(start_y) + ' and y < ' + str(end_y)
        elif (end_x < 10) and (end_y == 10):
            query_string = 'x >= ' + str(start_x) + ' and x < ' + str(end_x) + ' and y >= ' + str(start_y) + ' and y <= ' + str(end_y)
        elif (end_x == 10) and (end_y < 10):
            query_string = 'x >= ' + str(start_x) + ' and x <= ' + str(end_x) + ' and y >= ' + str(start_y) + ' and y < ' + str(end_y)
        else:
            query_string = 'x >= ' + str(start_x) + ' and x <= ' + str(end_x) + ' and y >= ' + str(start_y) + ' and y <= ' + str(end_y)
        
        
        #Drop data with place_id that occured less than 5 times. This is to manage the size of the model
        X_grid = training_data.query(query_string)
        X_test = test_data.query(query_string)
        X_grid = X_grid.groupby("place_id").filter(lambda x: len(x) > 4)
        Y_grid = X_grid.place_id
        X_grid = X_grid.drop('place_id' , 1)
        
        lb = preprocessing.LabelEncoder()
        lb.fit(Y_grid)
        Y_grid = lb.transform(Y_grid)
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_grid, Y_grid, test_size = 0.3)
        
        #Train XGB
        print('Training XGB')
        xgb_model = XGBClassifier(learning_rate = 0.1, n_estimators = 100, max_depth = 5, min_child_weight = 5, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, objective = 'multi:softmax', nthread = 3, scale_pos_weight = 1, reg_alpha = 0.1)
        xgb_model.fit(X_train, Y_train, eval_metric = 'map@3')

        
        Y_xgb_pred = xgb_model.predict_proba(X_val)
        Y_xgb_test = xgb_model.predict_proba(X_test)
        
        
        
        #Train kNN
        w = [2.8e8, 7e8, 7e7, 2.8e7, 1.0, 0.1]
        
        #Prep Data for kNN
        KNN_Prep(X_grid, X_train, w)
        KNN_Prep(X_grid, X_test, w)
        KNN_Prep(X_grid, X_val, w)
        
       
        print('Training kNN')
        
        knn = KNeighborsClassifier(n_neighbors = 50, n_jobs = 3, weights = 'distance', metric = 'manhattan')
        knn.fit(X_train, Y_train)
        
        Y_kNN_pred = knn.predict_proba(X_val)
        Y_kNN_test = knn.predict_proba(X_test)
        
        
        #Prepare for predictions
        
        
        #Combined Predictions - note that the format of predict_proba from XGB is different, hence the need of matrix manipulation in order to add the probabilities together
        Y_xgb_pred_inv = 1 - Y_xgb_pred
        Y_xgb_test_inv = 1 - Y_xgb_test
        Y_xgb_pred_inv = Y_xgb_pred_inv.transpose()
        Y_xgb_test_inv = Y_xgb_test_inv.transpose()
        Y_xgb_pred_inv = Y_xgb_pred_inv[0:len(X_val), :]
        Y_xgb_test_inv = Y_xgb_test_inv[0:len(X_test), :]
        #The weights here have been tested which seems to maximise the map@3 score for the validation data
        Y_combined_pred = (Y_kNN_pred**2 + 2.0*Y_xgb_pred_inv)
        Y_combined_test = (Y_kNN_test**2 + 2.0*Y_xgb_test_inv)
        
        combined_pred_labels = lb.inverse_transform(np.argsort(Y_combined_pred, axis=1)[:,::-1][:,:3]) 
        combined_test_labels = lb.inverse_transform(np.argsort(Y_combined_test, axis=1)[:,::-1][:,:3]) 
        
        
        #evaluation
        Y_val = lb.inverse_transform(Y_val)
        
        compare_comb, comb_map3 = predict_df(Y_val, combined_pred_labels, xgb_true = False)
        
        df_out, temp = predict_df(X_test.row_id, combined_test_labels, xgb_true = False, test = True)
        df = df_out[['row_id', 'place_id']]
        submission_combine = pd.concat([submission_combine, df])
        
        
        
submission_combine = submission_combine.sort("row_id")
submission_combine = submission_combine.drop_duplicates(subset = "row_id")
submission_combine.to_csv('submit_combine.csv', Columns = {'row_id', 'place_id'}, index = False)