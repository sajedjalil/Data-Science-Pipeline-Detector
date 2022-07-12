# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import copy 
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_file = '../input/train.csv'
test_file = '../input/test.csv'
train_dataset=pd.read_csv(train_file)
test_dataset=pd.read_csv(test_file)
y_train = train_dataset.iloc[:,-1].values
a = test_dataset.iloc[:,0].values
del train_dataset['Id']
del test_dataset['Id']
del train_dataset['Cover_Type']
indices = train_dataset.columns.tolist()
x_train = train_dataset.values
x_test =test_dataset.values
 
#data = pd.read_csv(r"C:\Users\huzhipeng_sx\Desktop\data",header = None,sep = '\t')
 
y_train = y_train
X_train = x_train
#X_train = pd.DataFrame(np.delete(X_train,-5,axis=1))
 
features_name = indices
 
 
rf = RandomForestClassifier(n_estimators=200,oob_score=True)
rf.fit(X_train,y_train)
features_imp = rf.feature_importances_
 
X_train = np.array(X_train)  

#feature selection
 
def select_combine(X_train,y_train,features_name,features_imp,select_num):
    oob_result = []
    fea_result = []
    features_imp = list(features_imp)
    iter_count = X_train.shape[1] - select_num  #
    if iter_count < 0:
        print("select_nume must less or equal X_train columns")
    else:
        features_test  = copy.deepcopy(features_imp)   
        features_test.sort()
        features_test.reverse() 
        
        while iter_count >= 0:
            iter_count -= 1
            train_index = [features_imp.index(j) for j in features_test[:select_num]]
            train_feature_name = [features_name[k] for k in train_index][0]
            train_data = X_train[:,train_index]
            rf.fit(train_data,y_train)
            acc = rf.oob_score_
            print(acc)
            oob_result.append(acc)
            fea_result.append(train_index)
            if select_num < X_train.shape[1]:
                select_num += 1
            else:
                break
    return max(oob_result),oob_result,fea_result[oob_result.index(max(oob_result))]
 
select_num = 25
max_result, oob_result, fea_result = select_combine(X_train,y_train,features_name,features_imp,select_num)

rf1 = RandomForestClassifier(n_estimators= 140, max_depth=23, min_samples_split=20,
                                  min_samples_leaf=1,max_features=20 ,oob_score=True, random_state=10)
rf1.fit(X_train[:,fea_result[:select_num]],y_train)
p_test = rf1.predict(x_test[:,fea_result[:select_num]])

# Output



ceil_final_predictions = pd.DataFrame({'Id': a.tolist(), 'Cover_Type': p_test.tolist()})

print(ceil_final_predictions.head())

# Files

ceil_final_predictions.to_csv('output_ceil.csv', index=False)
