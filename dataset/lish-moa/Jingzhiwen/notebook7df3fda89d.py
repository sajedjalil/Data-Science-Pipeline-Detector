# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# 首先导入需要的库
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from IPython.display import display

import seaborn as sns


from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

#获取训练特征
resource = pd.read_csv(r'/kaggle/input/lish-moa/train_features.csv')

#将字符串特征用数字表示
resource.loc[resource['cp_dose'] == 'D1','cp_dose'] = 1
resource.loc[resource['cp_dose'] == 'D2','cp_dose'] = 0
resource.loc[resource['cp_type'] == 'trt_cp','cp_type'] = 1
resource.loc[resource['cp_type'] == 'ctl_vehicle','cp_type'] = 0

print(resource)

#提取特征
train = resource.iloc[0:11905,1:]
test = resource.iloc[11906:,1:]

#读取标签CSV
text = pd.read_csv(r'/kaggle/input/lish-moa/train_targets_scored.csv')

#读取预测特征
pre = pd.read_csv(r'/kaggle/input/lish-moa/test_features.csv')

#将字符串特征用数字表示
pre.loc[pre['cp_dose'] == 'D1','cp_dose'] = 1
pre.loc[pre['cp_dose'] == 'D2','cp_dose'] = 0
pre.loc[pre['cp_type'] == 'trt_cp','cp_type'] = 1
pre.loc[pre['cp_type'] == 'ctl_vehicle','cp_type'] = 0


#打开结果CSV
with open(r"/kaggle/working/submission.csv", "w", newline=''):
    pass

#拿到预测目标数量
num = text.shape[1]

#在结果CSV中写入第一列
dic = pd.DataFrame({'sig_id':pre.iloc[:,0]})
dic.to_csv(r"/kaggle/working/submission.csv", sep=',',index=False)



#再读取结果CSV
sample = pd.read_csv(r"/kaggle/working/submission.csv")

train_text = text.iloc[0:11905, 1:] #训练标签
test_text = text.iloc[11906:, 1:] #测试标签



model = MLPClassifier([10, 6], learning_rate_init=0.001, activation='relu', \
                        solver='adam', alpha=0.0001, max_iter=30000)  # 神经网络

X_shuffle, y_shuffle = shuffle(train, train_text)



#训练
model.fit(X_shuffle,y_shuffle)
#预测
pred_Y = model.predict(pre.iloc[:, 1:])

#结果矩阵转置
pred_Y = pred_Y.T


pred_Y = list(pred_Y)


#按列写入结果
for i in range(1,num):
    colum_name = list(text)[i] #很神奇，拿到了训练标签的每列的列名
    print(colum_name,type(colum_name))
    print(i-1)
    sample[colum_name] = pred_Y[i-1] #为每列的列头提供数据

    sample.to_csv(r"/kaggle/working/submission.csv",index=False)


print(list(text))





