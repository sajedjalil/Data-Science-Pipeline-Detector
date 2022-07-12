# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd
import numpy  as  np
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
#more about kinetic features  developed  by Daia Alexandru    here  on the next  blog  please  read  last article :
#https://alexandrudaia.quora.com/

##############################################creatinng   kinetic PCA features for  train   BINDED WITH TEST#####################################################
 


first_kin_names=[col for  col in train.columns  if '_ind_' in col]
 


# In[12]:


trainvec=train[first_kin_names]
testvec=test[first_kin_names]


# In[14]:


trainvec=np.array(trainvec)
testvec=np.array(testvec)


# In[15]:


data=np.concatenate((trainvec,testvec))
print(data.shape)


# In[16]:


data=data.astype(int)
data


# In[29]:


import numpy as np
import pandas   as pd
 
import  numpy as np
def  kin_energy(random_vec):
    """return    kinetic  energy  of   random vector represented   of   (1,) dimmensional  array"""
    freq=np.unique(random_vec,return_counts=True)
    prob=freq[1]/random_vec.shape[0]
    energy=np.sum(prob**2)
    return  energy
def  ic(vector1,vector2):
    """return  information  coefficient   IC  for  2  random  variables 
    -defined as   dot product of   probabilities  corresponding to  each class
    
    """
    a=vector1
    b=vector2
    # get the probs  in order  to    do     dot product with  them 
    prob1=np.unique(a,return_counts=True)[1]/a.shape[0]
    prob2=np.unique(b,return_counts=True)[1]/b.shape[0]
    p1=list(prob1)
    p2=list(prob2)
    diff=len(p1)-len(p2)
    if diff>0:
        for elem in range(diff):
            p2.append(0)
    if diff<0:
        for  elem in range((diff*-1)):
            p1.append(0)
    ic=np.dot(np.array(p1),np.array(p2))
    return ic
    
 
def  o(vector1,vector2):
    """return onicescu   information   correlation   based on kinetic energy """
    i_c=ic(vector1,vector2)
    o=i_c/np.sqrt(kin_energy(vector1)*kin_energy(vector2))
    return o


# In[30]:


rows=data.shape[1]
rows


# In[31]:


matrix= np.zeros((rows,rows))


# In[32]:


for i in range(rows):
    for j in  range(i, rows):
        cor=o(data[:,i],data[:,j])
  
        matrix[i,j]=cor
        matrix[j,i]=cor


# In[33]:


corr_matrix=matrix


# In[22]:


class kineticPCA(object):
    def __init__(self,kinetic_components):
        self.kinetic_components=kinetic_components
        self.eigenvalues=None
        self.eigenvectors=None
    def fit_transform_Kinetic(self,matrix,data):
        self.eigenvalues,self.eigenvectors=np.linalg.eig(matrix)
        ordered=np.argsort(self.eigenvalues)
        components=self.eigenvectors[:,ordered[-self.kinetic_components:]]
        return data.dot(components)


# In[35]:


kPCA=kineticPCA(4)


# In[36]:


new_features=kPCA.fit_transform_Kinetic(corr_matrix,data)


# In[37]:


new_features.shape


# In[38]:


train['kineticPCA1']=new_features[0:train.shape[0],0]
train['kineticPCA2']=new_features[0:train.shape[0],1]
train['kineticPCA3']=new_features[0:train.shape[0],2]
train['kineticPCA4']=new_features[0:train.shape[0],3]


# In[49]:


test['kineticPCA1']=new_features[train.shape[0]:data.shape[0],0]
test['kineticPCA2']=new_features[train.shape[0]:data.shape[0],1]
test['kineticPCA3']=new_features[train.shape[0]:data.shape[0],2]
test['kineticPCA4']=new_features[train.shape[0]:data.shape[0],3]


# In[50]:


second_kin_names= [col for  col in test.columns  if '_car_' in col and col.endswith('cat')]


# In[53]:


trainvec=train[second_kin_names]
testvec=test[second_kin_names]
trainvec=np.array(trainvec)
testvec=np.array(testvec)
data=np.concatenate((trainvec,testvec))
print(data.shape)
data=data.astype(int)
data
rows=data.shape[1]
print('nr  of features :',rows)
matrix= np.zeros((rows,rows))
for i in range(rows):
    for j in  range(i, rows):
        cor=o(data[:,i],data[:,j])
  
        matrix[i,j]=cor
        matrix[j,i]=cor
corr_matrix=matrix
kPCA=kineticPCA(4)
new_features=kPCA.fit_transform_Kinetic(corr_matrix,data)
train['kineticPCA11']=new_features[0:train.shape[0],0]
train['kineticPCA21']=new_features[0:train.shape[0],1]
train['kineticPCA31']=new_features[0:train.shape[0],2]
train['kineticPCA41']=new_features[0:train.shape[0],3]
test['kineticPCA11']=new_features[train.shape[0]:data.shape[0],0]
test['kineticPCA21']=new_features[train.shape[0]:data.shape[0],1]
test['kineticPCA31']=new_features[train.shape[0]:data.shape[0],2]
test['kineticPCA41']=new_features[train.shape[0]:data.shape[0],3]


test.to_csv('kineticPCA_test.csv.gz', index=False,compression = "gzip")
train.to_csv('kineticPCA_train.csv.gz', index=False,compression = "gzip")
# In[57]:

"""
from sklearn import *
import xgboost as xgb
col = [c for c in train.columns if c not in ['id','target']]

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_xgb(preds, y):
    y = y.get_label()
    return 'gini', gini(y, preds) / gini(y, y)

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['target'], test_size=0.25, random_state=99)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
test[['id','target']].to_csv('kineticPCAXGBOOST.csv', index=False, float_format='%.5f')
"""