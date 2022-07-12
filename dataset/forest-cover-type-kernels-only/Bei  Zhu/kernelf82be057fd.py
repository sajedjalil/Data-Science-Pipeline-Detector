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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance
from sklearn.metrics import accuracy_score
#load trainset
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.head())
print(train_df.info())
print(train_df.shape)
print(test_df.shape)
print(train_df.isnull().sum())
plt.figure(figsize = (12,5))
plt.title('Count of forest categories')
sns.countplot(x = 'Cover_Type', data = train_df, palette = 'Reds_d')

plt.figure(figsize = (30,30))
plt.title('The correlation between features and lables')
sns.heatmap(train_df.corr(), cmap="YlGnBu", square = True, annot = True, fmt = '.1f')

train_df = train_df.drop(['Soil_Type7','Soil_Type15'], axis = 1)
test_df = test_df.drop(['Soil_Type7','Soil_Type15'], axis = 1)
#print(train_df.head())
print(train_df.shape)
print(test_df.shape)

def add_features(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['ele_vert'] = df.Elevation-df.Vertical_Distance_To_Hydrology

    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    return df

train_df = add_features(train_df)
test_df =  add_features(test_df)
print(train_df.shape)
print(test_df.shape)

#columns = train_df.columns
#feature_columns = columns.delete(len(columns)-1)

X_train = train_df.drop(['Cover_Type'], axis = 1)
y_train = train_df['Cover_Type']
print(train_df.shape)
print(test_df.shape)

X_test = test_df
#dtrain = xgb.DMatrix(X_train,y_train)
#dtest = xgb.DMatrix(X_test)


#clf = XGBClassifier(params = params)
#clf.fit(X_train, y_train)
model = XGBClassifier(learning_rate=0.3,
                      n_estimators=350,
                      max_depth=9,
                      objective='multi:softmax',
                      num_class=7)
model.fit(X_train,
          y_train)


 
### make prediction for test data
preds = model.predict(X_test)
print('over!!')
dataframe = pd.DataFrame({'Id':test_df['Id'],'Cover_Type':preds})
dataframe.to_csv("sample_submission.csv",index=False)
print('over!!!!')