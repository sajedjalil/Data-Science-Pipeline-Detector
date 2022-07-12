
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


number_of_data_points = 4000000         # Increase this to have a better model
train_df_total = pd.read_csv("../input/train_V2.csv") # Change the path accordingly 
test_df = pd.read_csv("../input/test_V2.csv")

print(train_df_total.shape)
print(test_df.shape[0])

train_df = train_df_total.sample(n=number_of_data_points)    # Sampling 10,000 rows at random

train_df = train_df.drop(['Id','matchId','groupId','matchId','matchType'],axis=1)

test_Id = test_df[['Id']]

test_df = test_df.drop(['Id','matchId','groupId','matchId','matchType'],axis=1)

ID = test_Id.values
print(ID)

target = train_df[['winPlacePerc']]
train_df = train_df.drop(['winPlacePerc'],axis=1)



std = train_df.std(axis=0)
mean = train_df.mean(axis=0)
col_names = list(train_df.columns.values)
for j in range(len(col_names)):
    train_df[col_names[j]] = (train_df[col_names[j]] - mean[col_names[j]])/std[col_names[j]] 
    
std1 = test_df.std(axis=0)
mean1 = test_df.mean(axis=0)
col_names1 = list(test_df.columns.values)
for j in range(len(col_names1)):
    test_df[col_names1[j]] = (test_df[col_names1[j]] - mean[col_names1[j]])/std[col_names1[j]] 
    


from numpy import cov            # To calculate covraiance matrix
from numpy.linalg import eig     # To calculate eigenvalues and eigenvectors

train_array = train_df.values   # Returns numpy array of values of the dataframe
test_array = test_df.values

cov_matrix = cov(train_array.T)

Sig = number_of_data_points*cov_matrix

x,y=eig(cov_matrix)

eig_val = -np.sort(-x)

eig_vec = y[:, (-x).argsort()]

s= 0
eig_val
for col in range(len(eig_val)):
    s = s + eig_val[col]
    if s/(np.sum(eig_val)) > 0.9:
        break
print(col)



new_eig_vec = eig_vec[:,0:col]
reduced_data = np.dot(train_array,new_eig_vec)

new_data_pca = np.ones((number_of_data_points,col+1))
new_data_pca[:,1:col+1] = reduced_data 

# For Non PCA data
new_data = np.ones((number_of_data_points,train_array.shape[1]+1))
new_data[:,1:train_array.shape[1]+1] = train_array 


newtest = np.ones((test_df.shape[0],test_array.shape[1]+1))
newtest[:,1:test_array.shape[1]+1] = test_array

print(newtest.shape)



target_np = target.values   # converting pandas df to numpy array

a = int(number_of_data_points*1)

train_x_pca = new_data_pca[0:a]      # Train data for PCA Data
train_y = target_np[0:a]
train_x = new_data[0:a]              # Train data for the non PCA Data
print(train_x.shape)   
   
print(a)


def cost(w,x,y):
    J = (1/(2*len(y)))*np.dot((((np.dot(x,w))-y).T),((np.dot(x,w))-y))
    return J
    
def grad_desc(alpha,x,y,num_itr):
    w=np.zeros((x.shape[1],1))
    for n in range(num_itr):
        de = (1/len(y))*np.dot((x.T),((np.dot(x,w))-y))
        w = w - alpha*(de)
    return w
    
ans3=grad_desc(0.1,train_x,train_y,500)

print(ans3.shape)
print(newtest.shape)

ans = np.dot(newtest,ans3)
print(ans.shape,ID.shape)

sub = pd.DataFrame({'Id':ID[:,0],'winPlacePerc':ans[:,0]})
#print(sub)
sub.to_csv('submission.csv',index= False)
    


# Any results you write to the current directory are saved as output.

