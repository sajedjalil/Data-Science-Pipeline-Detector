import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================

'''Author: CHG, and partially based on other kaggle 
scripts (Sandro, anokas). This code reads in train/test 
data to a pandas dataframe, splits up data into grid 
cells and then cleans the data. Many different options 
are then included to add features to data before 
perfoming cross-validation. Specific options added to
modify train/test data for use with LIBSVM.''' 

# =================================================
# Define some functions to split/clean/featurize data

# Function 1 : Define smaller grid cells
# Train data small grids

def smallGrid(data,ss,zero,xy):
    if xy == 1:
        df_small = data.loc[(data['x'] < ss) & (data['x'] >= zero)]
    else:
        df_small = data.loc[(data['y'] < ss) & (data['y'] >= zero)]

    u_P = str(len(list(set(df_small['place_id'].values.tolist()))))
    counts_P = df_small['place_id'].value_counts()
    np.set_printoptions(precision=3)
    print('Number of unique places/shape in small grid...')
    print (u_P,df_small.shape)

    return df_small

# Test data small grids

def smallGridt(data,ss,zero,xy):
    if xy == 1:
        df_small_test = data.loc[(data['x'] < ss) & (data['x'] >= zero)]
    else:
        df_small_test = data.loc[(data['y'] < ss) & (data['y'] >= zero)]

    return df_small_test

# Function 2 : Delete any place_id's with < th examples...

'''Will test this during cross-validation, a trade-off of
over-fitting, code speed and highest possible accuracy'''
 
def removeLTth(data,th):
    counts = data.place_id.value_counts()
    mask = counts[data.place_id.values] >= th
    data = data.loc[mask.values]
    
    u_P = str(len(list(set(data['place_id'].values.tolist()))))
    np.set_printoptions(precision=3)
    print ('Number of unique places/shape after LTth...')
    print (u_P,data.shape)
    return data
    
# Function 3 : For each place_id, compare all x and y..
# values with mean + 2 sigma and delete all outlier rows

def removeOutliers(data):
    
    data['u_x'] = 0
    data['u_y'] = 0
    data['sigma_x'] = 0
    data['sigma_y'] = 0
    
    data = data.sort_values(by='place_id')
    u_P = len(list(set(data['place_id'].values.tolist())))
    counts = data.place_id.value_counts(sort=False)
    counts = counts.sort_index()
    i = 0
    j = counts.iloc[0]
    for n in range(0,u_P-1):
        data['u_x'].iloc[i:j] = data['x'].iloc[i:j].mean()
        data['u_y'].iloc[i:j] = data['y'].iloc[i:j].mean()
        data['sigma_x'].iloc[i:j] = data['x'].iloc[i:j].std()
        data['sigma_y'].iloc[i:j] = data['y'].iloc[i:j].std()
        i = i + counts.iloc[n]
        j = j + counts.iloc[n+1]

    mask = data[data['x'] <= data['u_x'] + 2*data['sigma_x']]
    data = mask
    mask = data[data['x'] >= data['u_x'] - 2*data['sigma_x']]    
    data = mask
    
    mask = data[data['y'] <= data['u_y'] + 2*data['sigma_y']]
    data = mask
    mask = data[data['y'] >= data['u_y'] - 2*data['sigma_y']]
    data = mask

    u_P = str(len(list(set(data['place_id'].values.tolist()))))
    print ('Number of unique places after 2 sigma removal...')
    print (u_P,data.shape)
    
    return data

# Function 4 : Determine lots of possible features to use:
# x,y,x^n,y^n,x*y,x/y,time,hour,weekday...

def addFeatures(data):
    inf = 0.0000001

    # Higher Order x,y features
    degree = 2
    k = 0
    for i in range(2,degree+1):
        for j in range (0,i+1):
            k = k + 1
    k = 0
    for i in range(2,degree+1):
        for j in range (0,i+1):
            k = k + 1
            data['HO_xy%d' % k] = data.x.values**(i-j) * data.y.values**j

    data['x_y'] = data.x.values / (data.y.values+inf)
    data['y_x'] = data.y.values / (data.x.values+inf)
    data['logx'] = np.log(data.x.values+0.01)
    data['logy'] = np.log(data.y.values+0.01)

    minutes = data.time.values
    hours = data.time.values // 60
    days = hours // 24
    data['hour_of_day'] = hours - days*24
    weeks = days // 7
    data['weekday'] = days - weeks*7

    # Higher Order Time Features
    k = 0
    for i in range(2,degree+1):
        for j in range (0,i+1):
            k = k + 1
            data['HO_time%d' % k] = data.hour_of_day.values**(i-j) * data.weekday.values**j

    return data

# Function 5 : Normalize the selected features

def featureNormalize(data):
    print('Normalizing the data: (X - mean(X)) / std(X) ...')
    cols = ['x', 'y', 'xy', 'xdy', 'ydx', 'lx', 'ly', 'u_x', 'u_y', 'hour', 'weekday', 'day', 'month', 'year']
    for cl in cols:
        ave = data[cl].mean()
        std = data[cl].std()
        data[cl] = (data[cl].values - ave ) / std

    return data

# Function 6 : Modify y to #'s 0 to nlabels
# Helped when using LIBSVM/LIBLINEAR

def modYtrain(Y):
    m = Y.shape[0]
    labels = np.unique(Y)
    num_labels = len(labels)
    Ymod = np.zeros((m,1))
    Ymod[0] = 0
    j = 0
    for i in range(m-1):
        if Y[i+1] == Y[i]:
            Ymod[i+1] = j
        else:
            Ymod[i+1] = j + 1
            j = j + 1

    return Ymod

# =================================================
# Main
if __name__ == '__main__':
    
    # =================================================
    # Read in the train/test data to dataframe

    df_train = pd.read_csv('../input/train.csv',usecols=['row_id','x','y','time','place_id'],index_col = 0)
    df_test = pd.read_csv('../input/test.csv',usecols=['row_id','x','y','time'],index_col = 0)

    print('Training data shape: ' + str(df_train.shape))
    print('Testing data shape:  ' + str(df_test.shape))
    print('Number of training unique places: ' + str(len(list(set(df_train['place_id'].values.tolist())))))
  
    # Use defined functions to split/clean/featurize data
    k = 0

    # Set the ranges to 20 and 20 to cover full grid

    for i in range(1):
        for j in range(1):
            split_size_y = np.linspace(0.5,10,20)
            zeroy = np.linspace(0,9.5,20)
            xy = 2
            df_small = smallGrid(df_train.sort_values(by='y'),split_size_y[i],zeroy[i],xy)
            df_small_test = smallGridt(df_test.sort_values(by='y'),split_size_y[i],zeroy[i],xy)

            xy = 1
            split_size_x = np.linspace(0.5,10,20)
            zerox = np.linspace(0,9.5,20)
            df_small = smallGrid(df_small.sort_values(by='x'),split_size_x[j],zerox[j],xy)
            df_small_test = smallGridt(df_small_test.sort_values(by='x'),split_size_x[j],zerox[j],xy)

            # Choose threshold value
            th = 3
            df_small = removeLTth(df_small,th)
            df_small = removeOutliers(df_small)

            # Add features to train/test data 
            df_small = addFeatures(df_small)
            df_small_test = addFeatures(df_small_test)

            print('Shape of training data after featurizing...')
            print (df_small.shape)

            print('Shape of test data after featurizing...')
            print (df_small_test.shape)

            # Optional: Feature normalize
            #df_small = featureNormalize(df_small,u,sigma))
            #df_small_test = featureNormalize(df_small_test,u,sigma))

            # Optional: Modify data for LIBSVM/LIBLINEAR format
        
            # Put train data into numpy array
            train_y = np.zeros((df_small.shape[0],1))
            train_x = df_small.drop(['place_id','u_x','u_y','sigma_x','sigma_y'],axis=1).values
            train_y[:,0] = df_small['place_id'].values.astype(int)
            data_lib = np.append(train_y,train_x,axis=1)
        
            # Sort by place_id
            data_lib = data_lib[data_lib[:,0].argsort()]

            # Store the original and modified labels
            labels_uq = np.unique(data_lib[:,[0]])
            labels = np.zeros((len(labels_uq),1))
            labels[:,0] = labels_uq
            data_lib[:,[0]] = modYtrain(data_lib[:,0])
            labels_uq = np.unique(data_lib[:,[0]])
            labels_mod = np.zeros((len(labels_uq),1))
            labels_mod[:,0] = labels_uq
            labels = np.append(labels,labels_mod,axis=1)

            # Put test data into numpy array
            test_y = np.zeros((df_small_test.shape[0],1))
            row_ids = np.zeros((df_small_test.shape[0],1))
            test_x = df_small_test.values
            row_ids[:,0] = df_small_test.index.values

            data_lib_test = np.append(test_y,test_x,axis=1)
            data_lib_test[:,[0]] = modYtrain(data_lib_test[:,0])

            # =================================================
            # Write out LIBSVM/LIBLINEAR data to .csv
        
            write1 = 'training%d.csv' % (k)
 
            with open(write1, 'w') as mycsvfile:
                thedatawriter = csv.writer(mycsvfile, dialect='excel')
                for row in data_lib:
                    thedatawriter.writerow(row)

            write2 = 'testing%d.csv' % (k)

            with open(write2, 'w') as mycsvfile:
                thedatawriter = csv.writer(mycsvfile, dialect='excel')
                for row in data_lib_test:
                    thedatawriter.writerow(row)

            write3 = 'labels%d.csv' % (k)

            with open(write3, 'w') as mycsvfile:
                thedatawriter = csv.writer(mycsvfile, dialect='excel')
                for row in labels:
                    thedatawriter.writerow(row)
  
            write4 = 'row_ids%d.csv' % (k)

            with open(write4, 'w') as mycsvfile:
                thedatawriter = csv.writer(mycsvfile, dialect='excel')
                for row in row_ids:
                    thedatawriter.writerow(row)
        
            k = k + 1

        # =================================================
        # Output for other ML libraries
        #df_small.to_csv('training.csv', index=True, header=True, index_label='row_id')
        #df_small_test.to_csv('testing.csv', index=True, header=True, index_label='row_id')
