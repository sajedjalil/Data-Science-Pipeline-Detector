# Getting started
# Use numeric features only (in train_set.csv) for training
# Linear Regression 

#-------------------------------------
#define the evaluation metric (RMSLE)
import numpy as np
from numpy import sqrt,mean,log
def rmsle(p,a): # p,a : numpy arrays
    return sqrt(mean( (log(p+1)-log(a+1))**2 ))
#-------------------------------------
#read from csv
import pandas as pd
path="../input/" # path to data
train_df = pd.read_csv(path+"train_set.csv",header=0,index_col=None)
test_df = pd.read_csv(path+"test_set.csv",header=0,index_col=None)
#-------------------------------------
#find numeric columns
# s = train_df.dtypes # pandas Series
# def isnumeric(x): return x!=np.dtype('object')
# b = s.map(isnumeric)
# numeric_cols = list(s[b].index)
# numeric_features = [f for f in numeric_cols if f!='cost']

#or just do it by inspection
numeric_features = ['annual_usage','min_order_quantity','quantity']
target = train_df['cost']
target = np.log(1+target)
features = train_df[numeric_features].as_matrix()
#----------------------------------------
# train model 
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(features, target)
predicted = clf.predict(features)
predicted = np.exp(predicted)-1
#----------------------------------------
# plt.scatter(target, predicted)
# plt.plot([0, 1000], [0, 1000], '--k')
# plt.axis('tight')
# plt.xlabel('True cost')
# plt.ylabel('Predicted cost')
#----------------------------------------
actual = train_df['cost']
err = rmsle(p=predicted,a=actual) # 0.756066
print("RMSLE: %f" %(err) )
#----------------------------------------
# predict
features = test_df[numeric_features].as_matrix()
predicted = clf.predict(features)
predicted = np.exp(predicted)-1
predicted[np.where(predicted<0)] = 1 # set negative price values to 1 
                                     # because log(negative) = -infinity
test_df['cost'] = predicted
#----------------------------------------
# write to csv
submit = test_df[['id','cost']]
submit.to_csv("mysubmit-traincsv-numonly.csv",index=False)