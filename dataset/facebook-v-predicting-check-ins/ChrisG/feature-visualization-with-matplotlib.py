import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

# =================================================

'''This script is for feature selection. Function #6
plots combinations of features. Current setup plots first
50 classes on 50m X 50m grid cell.
Functions for data processing partially from Sandro'''

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

# Function 5 : Modify place_id's to 0,1,2... 

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
    
# Function 6 : Plot sets of features

def plotFeatures(data):   
    # Plot only first 50 classes
    data = data[0:6035,:]
    u_P = np.unique(data[:,0],return_index=False)
    print ('Number of unique places to plot, shape of data...')
    print (len(u_P),data.shape)

    # Plot some features in matplotlib subplots
    i = 0
    j = 0
    k = 0
    jj = 1
    c = ['r','b','g','k','m','y','c']
    marker = ['.','o','x','+','*','s','d']

    for i in range(1,data.shape[0]):
        if data[i,0] == data[i-1,0]:
            jj = 1
            for ii in range(1,15):
                jj = jj + 1
                plt.subplot(3,5,ii)
                if ii == 3 or ii == 8:
                    plt.plot(data[i,2],data_lib[i,jj],c=c[j],marker=marker[k])
                else:
                    plt.plot(data[i,1],data[i,jj],c=c[j],marker=marker[k])
        else:
            j = j + 1
            jj = 1
            for ii in range(1,15):
                jj = jj + 1
                plt.subplot(3,5,ii)
                if ii == 3 or ii == 8:
                    plt.plot(data[i,2],data[i,jj],c=c[j],marker=marker[k])
                else: 
                    plt.plot(data[i,1],data[i,jj],c=c[j],marker=marker[k])    
        if j == 6:
            j = 0
            k = k + 1
        if k == 7:
            k = 0

    y_axis = ['y','time','x^2','x*y','y^2','x/y','y/x','logx','logy','hour of day','weekday','hour^2','hour*day','day^2']

    for i in range (1,15):
        plt.subplot(3,5,i)
        plt.xlabel('x')
        if i == 3 or i == 8:
            plt.xlabel('y')
        plt.ylabel(y_axis[i-1])
        plt.grid(True)

    plt.suptitle('Feature Selection')
    plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.40,hspace=0.25)
    fig = plt.gcf()
    fig.set_size_inches(30, 15)
    fig.savefig('featureSelect.png', dpi=200)

# =================================================
# Main
if __name__ == '__main__':
    
    # =================================================
    # Read in the train/test data to dataframe

    df_train = pd.read_csv('../input/train.csv',usecols=['row_id','x','y','time','place_id'],index_col = 0)
  
    # Use defined functions to split/clean/featurize data
    i = 0
    j = 0

    split_size_y = np.linspace(0.5,10,20)
    zeroy = np.linspace(0,9.5,20)
    xy = 2
    df_small = smallGrid(df_train.sort_values(by='y'),split_size_y[i],zeroy[i],xy)

    xy = 1
    split_size_x = np.linspace(0.5,10,20)
    zerox = np.linspace(0,9.5,20)
    df_small = smallGrid(df_small.sort_values(by='x'),split_size_x[j],zerox[j],xy)

    # Choose threshold value
    th = 3
    df_small = removeLTth(df_small,th)
    #df_small = removeOutliers(df_small)

    # Add features to train/test data 
    df_small = addFeatures(df_small)

    print('Shape of training data after featurizing...')
    print (df_small.shape)
        
    # Put train data into numpy array
    train_y = np.zeros((df_small.shape[0],1))
    #train_x = df_small.drop(['place_id','u_x','u_y','sigma_x','sigma_y'],axis=1).values
    train_x = df_small.drop(['place_id'],axis=1).values
    train_y[:,0] = df_small['place_id'].values.astype(int)
    data_lib = np.append(train_y,train_x,axis=1)
    
    # Sort by place_id
    data_lib = data_lib[data_lib[:,0].argsort()]

    # Store the original and modified labels
    data_lib[:,[0]] = modYtrain(data_lib[:,0])
    
    # Plot conbinations of features
    plotFeatures(data_lib)
    
