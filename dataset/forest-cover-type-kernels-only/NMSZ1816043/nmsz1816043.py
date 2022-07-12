import pandas as pd
import numpy as np
import gc
from sklearn.ensemble import ExtraTreesClassifier
import warnings
import math
warnings.filterwarnings("ignore")

def change_feature_data(data_df):
    del data_df['Id'],data_df['Aspect'],data_df['Hillshade_3pm']
    water_elevation = pd.DataFrame(data_df['Elevation']+data_df['Vertical_Distance_To_Hydrology'])
    data_df['Water_Elevation'] = water_elevation
    tp_sum = pd.DataFrame(data_df['Horizontal_Distance_To_Hydrology']+data_df['Horizontal_Distance_To_Roadways']+data_df['Horizontal_Distance_To_Fire_Points'])
    data_df['Tp_Sum'] = tp_sum
    tp_alt = pd.DataFrame(((data_df['Horizontal_Distance_To_Hydrology']-data_df['Tp_Sum']/3).map(lambda x: x**2))+((data_df['Horizontal_Distance_To_Roadways']-data_df['Tp_Sum']/3).map(lambda x: x**2))+((data_df['Horizontal_Distance_To_Fire_Points']-data_df['Tp_Sum']/3).map(lambda x: x**2)))
    data_df['Tp_alt'] = tp_alt
    data_df['Tp_alt'] = data_df['Tp_alt'].apply(lambda x: math.sqrt(x))
    return data_df
train_df = pd.read_csv('../input/train.csv')
validation_df = pd.read_csv('../input/test.csv')
arr = validation_df['Id']
n_columns = len(train_df.columns)
feature_column_names = train_df.columns[0:n_columns - 1]
label_column_name = train_df.columns[n_columns - 1]
train_feature_df = train_df[feature_column_names]
train_X = change_feature_data(train_feature_df)
validation_X = change_feature_data(validation_df)
X_train = train_X
Y_train = train_df[label_column_name]
X_dv = X_train.values
Y_dv = Y_train.values
for i in range(3):
    X_dv = np.vstack((X_dv,X_dv))
    Y_dv = np.hstack((Y_dv,Y_dv))

clf = ExtraTreesClassifier(n_estimators=150,random_state=14,max_features=17)#0.868716 150 14 17
s = clf.fit(X_dv,Y_dv)
y = s.predict(validation_X)
y_df = pd.DataFrame(y,columns=['Cover_Type'])
result = pd.concat([arr,y_df],axis=1, join='inner')

result.to_csv('result.csv',index=False,header=True)