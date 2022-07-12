# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
import csv
import pandas as pd
from xgboost import XGBClassifier

# 这个函数的功能就是把测试的结果保存在一个csv文件中
def saveResult(result,filename):
    path = '%s.csv' % (filename)
    headers = ['id', 'label']
    #with open(r'filename','w') as myFile:
    with open(path, 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(headers)
        x=1
        for i in result:
            tmp=[x]
            tmp.append(i)#行号之后是预测的数据
            myWriter.writerow(tmp)
            x += 1  

Train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
Test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

X_train = Train_data.ix[:,1:]
Y_train = Train_data.ix[:,[0]]
X_test = Test_data.ix[:,1:]

XGBoost_model = XGBClassifier(learning_rate=0.01,
                      n_estimators=10,           # 树的个数-10棵树建立xgboost
                      max_depth=4,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=1,               # 所有样本建立决策树
                      colsample_btree=1,         # 所有特征建立决策树
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27,           # 随机数
                      slient = 0
                      )
XGBoost_model.fit(X_train,Y_train)
y_pred_XGBoost = XGBoost_model.predict(X_test)
saveResult(y_pred_XGBoost, filename='submission')


