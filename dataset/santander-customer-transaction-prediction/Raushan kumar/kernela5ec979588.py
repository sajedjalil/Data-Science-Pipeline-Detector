# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import np_utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
traindata=pd.read_csv("../input/train.csv")
testdata=pd.read_csv("../input/test.csv")
print(traindata.shape)
print(testdata.shape)
traindata.head()

# Any results you write to the current directory are saved as output.
y=traindata.loc[:,'target']
traindata=traindata.iloc[:,2:]
testid = testdata.iloc[:,0]
testdata = testdata.iloc[:,1:]
total = traindata.append(testdata,sort=False)
total.head()

#plotting histograms yo see skewness
m=1
plt.figure(figsize=(15,15))
for i in total.columns[:5]:
    plt.subplot(3,4,m)
    sns.distplot(total[i],kde = True)
    m = m+1
    
from scipy.stats import skew
def skew_operation(data):
    data_x=data.copy()
    col = data_x.columns
    skewed_f = data_x[col].apply(lambda x: skew(x.dropna()))
    skewed_f = skewed_f[skewed_f > 0.75]
    skewed_f = skewed_f.index
    data_x[skewed_f] = np.log1p(data_x[skewed_f])
    
    return data_x
    
    
total_sk = skew_operation(total)
total_sk.head()

#Undersampling
#reducing y=0 labels from training data
total = total.reset_index(drop=True)
y=y.reset_index(drop=True)

#Shuffle the training data
n_train = int(traindata.shape[0])
train_data = total.iloc[:n_train,:]
remove_n = int(n_train*0.6)
drop_indices = np.random.choice(y[y==0].index, remove_n, replace=False)
print('Shape of training data before dropping rows having 0 labels: ', train_data.shape)
train_data = train_data.drop(drop_indices, axis=0)
y1 = y.copy()
y1 = y1.drop(drop_indices)
print('Shape of training data after dropping rows having 0 labels: ',train_data.shape)

y1.value_counts(normalize=True)
test_data = total.iloc[n_train: ,:]
rs = RobustScaler()
rs.fit(train_data)
train_data = rs.transform(train_data)
test_data = rs.transform(test_data)

#Splitting into training and testing dataset
from sklearn.model_selection import train_test_split

#  split X between training and testing set
x_train, x_test, y_train, y_test = train_test_split(train_data,y1, test_size=0.25, shuffle=True)
y_train.value_counts(normalize=True)

#Lightgbm model
import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)
params = {'num_leaves': 9,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 16,
         'learning_rate': 0.0123,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}
clf = lgb.train(params, d_train, 15000)
lgb_pred = clf.predict(x_test) #output will be probabilties


roc_auc_score(y_test, lgb_pred)
y_pred = clf.predict(test_data)


sub = pd.DataFrame(data = testid,columns =['ID_code'])
sub['target'] = y_pred
sub.to_csv('submission.csv', index=False)
