# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#COMPETITION NAME (the 'Competition'): Facebook V: Predicting Check Ins 
#COMPETITION SPONSOR: Facebook
#COMPETITION WEBSITE: https://www.kaggle.com/c/facebook-v-predicting-check-ins

'''Partially based on several scripts:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts
And Kaggle Overfitting Community :)
'''

import numpy as np
import pandas as pd

#print('	Reading train.csv')
df_train = pd.read_csv('../input/train.csv',usecols=['x','y','time','place_id','accuracy'])
#print('	Reading test.csv')
df_test = pd.read_csv('../input/test.csv',usecols=['x','y','time','accuracy'])

#print('Feature Augmentation')
minute=df_train['time']%60
df_train['hour'] = df_train['time']//60
df_train.drop(['time'], axis=1, inplace=True)
df_train['weekday'] = df_train['hour']//24
df_train['month'] = df_train['weekday']//30
df_train['year'] = (df_train['weekday']//365+1)*10.0
df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)*4.0
pd.options.mode.chained_assignment = None
add_data = df_train[df_train.hour<10]# add data for periodic time that hit the boundary
add_data.hour = add_data.hour+96
add_data2 = df_train[df_train.hour>90]
add_data2.hour = add_data2.hour-96
df_train = df_train.append(add_data)
df_train = df_train.append(add_data2)
del add_data,add_data2
df_train['weekday'] = (df_train['weekday']%7+1)*3.12
df_train['month'] = (df_train['month']%12+1)*2.12
df_train['accuracy'] = np.log10(df_train['accuracy'])*10.0

minute = df_test['time']%60
df_test['hour'] = df_test['time']//60
df_test.drop(['time'], axis=1, inplace=True)
df_test['weekday'] = df_test['hour']//24
df_test['month'] = df_test['weekday']//30
df_test['year'] = (df_test['weekday']//365+1)*10.0
df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*4.0
del minute
df_test['weekday'] = (df_test['weekday']%7+1)*3.12
df_test['month'] = (df_test['month']%12+1)*2.12
df_test['accuracy'] = np.log10(df_test['accuracy'])*10.0

print('Generating wrong models. They are just useful to get this job :) ... done')
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def calculate_distance(distances):
    return distances ** -2

def process_one_cell(df_cell_train, df_cell_test):
    
    #Working on df_train
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= 5).values
    df_cell_train = df_cell_train.loc[mask]
    
    #Working on df_test
    row_ids = df_cell_test.index
    
    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= 462.0
    df_cell_train.loc[:,'y'] *= 975.0
    df_cell_test.loc[:,'x'] *= 462.0
    df_cell_test.loc[:,'y'] *= 975.0
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values
    
    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=np.floor((np.sqrt(y.size)/5.3)).astype(int), 
                             weights=calculate_distance,metric='manhattan',n_jobs=2)
    clf.fit(X, y)
    y_pred = clf.predict_proba(df_cell_test.values)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]) 
    
    return pred_labels, row_ids
   
def process_grid(df_train, df_test):
    """
    Iterates over all grid cells, aggregates the results
    """
    size = 10.0
    x_step = 0.5
    y_step = 0.25
    
    x_border_augment = 0.027
    y_border_augment = 0.015
    
    preds = np.zeros((df_test.shape[0], 3), dtype=int)

    for i in range((int)(size/x_step)):
        
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4) 
        if x_max == size:
            x_max = x_max + 0.001
            
        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)   
            if y_max == size:
                y_max = y_max + 0.001
                
            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]
            
            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_cell_train, df_cell_test)

            #Updating predictions
            preds[row_ids] = pred_labels
    
    return preds

def generate_sub(preds):
    out = open('sample_submission.csv', "w")
    out.write("row_id,place_id\n")
    rows = ['']*8607230
    for num in range(0,8607230):
        rows[num]='%d,%d %d %d\n' % (num,preds[num,0],preds[num,1],preds[num,2])
    out.writelines(rows)
    out.close()

preds=process_grid(df_train, df_test)

generate_sub(preds)