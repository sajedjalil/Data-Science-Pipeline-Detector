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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier as xgb
from sklearn.ensemble import RandomForestClassifier as rfc
'''读取train数据并划分'''
train_data=pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')
train_data.drop(columns='Id', inplace=True)
print("训练集数据个数为 %i " % train_data.shape[0])
print("特征个数为  %i " % train_data.shape[1])
#保存训练集label以进行后续整体特征处理
train_label=train_data['Cover_Type']
train_data.drop(columns='Cover_Type', inplace=True)
'''读取测试集数据'''
test_inidata=pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')
test_inidata.drop(columns='Id', inplace=True)
test_data=test_inidata
test_inidata=pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')
'''合并test&train data 特征工程'''
all_data=pd.concat([train_data, test_data], axis=0, ignore_index=True)

'''观察训练集数据特征'''
plt.figure(figsize=(15, 12))
train_corr = train_data.corr()
sns.heatmap(train_corr, square=True, vmax=0.8, cmap='RdBu')
#soil_Type7和soil_Type15 nan 删除
#与最近地表水的垂直和水平距离高度相关应该合并
all_data.drop(['Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
classes = np.array(list(train_label.values))
'''增加变量'''
#合并地表水特征
all_data['distance_to_hydrology']=(all_data['Horizontal_Distance_To_Hydrology']**2+all_data['Vertical_Distance_To_Hydrology'])**(1/2)
#山体阴影
all_data['mean_hillshade'] =  (all_data['Hillshade_9am']+all_data['Hillshade_Noon']+ all_data['Hillshade_3pm'] ) / 3
#3个水平距离（水文、火店、公路）取和差
all_data['HorizontalHydrology_HorizontalFire'] = (all_data['Horizontal_Distance_To_Hydrology']+all_data['Horizontal_Distance_To_Fire_Points'])
all_data['Neg_HorizontalHydrology_HorizontalFire'] = (all_data['Horizontal_Distance_To_Hydrology']-all_data['Horizontal_Distance_To_Fire_Points'])
all_data['HorizontalHydrology_HorizontalRoadways'] = (all_data['Horizontal_Distance_To_Hydrology']+all_data['Horizontal_Distance_To_Roadways'])
all_data['Neg_HorizontalHydrology_HorizontalRoadways'] = (all_data['Horizontal_Distance_To_Hydrology']-all_data['Horizontal_Distance_To_Roadways'])
all_data['HorizontalFire_Points_HorizontalRoadways'] = (all_data['Horizontal_Distance_To_Fire_Points']+all_data['Horizontal_Distance_To_Roadways'])
all_data['Neg_HorizontalFire_Points_HorizontalRoadways'] = (all_data['Horizontal_Distance_To_Fire_Points']-all_data['Horizontal_Distance_To_Roadways'])
#垂直距离取和差
all_data['Neg_Elevation_Vertical'] =all_data['Elevation']-all_data['Vertical_Distance_To_Hydrology']
all_data['Elevation_Vertical'] =all_data['Elevation']+all_data['Vertical_Distance_To_Hydrology']

all_data['Mean_Fire_Hydrology_Roadways']=(all_data['Horizontal_Distance_To_Fire_Points'] + all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Roadways']) / 3

all_data['Neg_EHyd'] = all_data.Elevation-all_data.Horizontal_Distance_To_Hydrology*0.2
#垂直水文距离取绝对值
all_data["Vertical_Distance_To_Hydrology"] = abs(all_data['Vertical_Distance_To_Hydrology'])

plt.figure(figsize=(10,5))
sns.distplot(all_data['distance_to_hydrology'])
plt.xticks(rotation=30)

#将处理完的data还原成训练集和测试集（所求），训练集中再划分
x=np.array(all_data)[:train_data.shape[0],:].reshape([train_data.shape[0],-1])
y=np.array(train_label).reshape([-1,1])
test_data=np.array(all_data)[train_data.shape[0]:,:]
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=100)

#归一化处理
scaler =  MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_data=scaler.transform(test_data)
#检查分类样本个数是否比较平衡
unique, count= np.unique(y_train, return_counts=True)
print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )
'''模型测试'''
#随机森林
rfc_model= rfc(n_estimators = 200,criterion = 'gini',random_state = 0).fit(x_train,y_train)
y_pred_rfc=rfc_model.predict(x_test)
accurracy_rfc=accuracy_score(y_test,y_pred_rfc)

y_test_pred =rfc_model.predict(test_data)
sub = pd.DataFrame({"Id": test_inidata['Id'],'Cover_Type':y_test_pred}) 
sub.to_csv('submission.csv', index=False)
print('end!')