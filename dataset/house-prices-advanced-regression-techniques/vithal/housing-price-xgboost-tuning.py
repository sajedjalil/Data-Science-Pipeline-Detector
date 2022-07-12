

# IMPORTING THE PACKAGES
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

#Loading the data
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
dt = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#y_train  = (df['SalePrice'])
#print(df.shape)
#print(dt.shape)
#submission = pd.read_csv('sample_submission.csv')

# Feature Engineering
df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis = 1,inplace = True)

dt.drop(['Alley','PoolQC','Fence','MiscFeature'],axis = 1,inplace = True)

#print(df['BsmtFinSF1'].value_counts())
df['FireplaceQu']=df['FireplaceQu'].fillna((df['FireplaceQu'].mode()[0]))
dt['FireplaceQu']=dt['FireplaceQu'].fillna((dt['FireplaceQu'].mode()[0]))

df['LotFrontage'] =df['LotFrontage'].fillna((df['LotFrontage'].mean()))
df['LotFrontage'] =df['LotFrontage'].astype(int)

dt['LotFrontage'] =dt['LotFrontage'].fillna((dt['LotFrontage'].mean()))
dt['LotFrontage'] =dt['LotFrontage'].astype(int)
df['BsmtExposure']=df['BsmtExposure'].fillna((df['BsmtExposure'].mode()[0]))

dt['BsmtExposure']=dt['BsmtExposure'].fillna((dt['BsmtExposure'].mode()[0]))
#print(df['BsmtQual'].value_counts())
df['BsmtQual']=df['BsmtQual'].fillna((df['BsmtQual'].mode()[0]))
dt['BsmtQual']=dt['BsmtQual'].fillna((dt['BsmtQual'].mode()[0]))


df['GarageType']=df['GarageType'].fillna((df['GarageType'].mode()[0]))
dt['GarageType']=dt['GarageType'].fillna((dt['GarageType'].mode()[0]))
#print(df.isnull().sum())
df['GarageYrBlt'] =df['GarageYrBlt'].fillna((df['GarageYrBlt'].mean()))
dt['GarageYrBlt'] =dt['GarageYrBlt'].fillna((dt['GarageYrBlt'].mean()))
#print(df.isnull().sum())

df['GarageFinish']=df['GarageFinish'].fillna((df['GarageFinish'].mode()[0]))
dt['GarageFinish']=dt['GarageFinish'].fillna((dt['GarageFinish'].mode()[0]))
df['GarageQual']=df['GarageQual'].fillna((df['GarageQual'].mode()[0]))
dt['GarageQual']=dt['GarageQual'].fillna((dt['GarageQual'].mode()[0]))
df['GarageCond']=df['GarageCond'].fillna((df['GarageCond'].mode()[0]))
dt['GarageCond']=dt['GarageCond'].fillna((dt['GarageCond'].mode()[0]))

dt['GarageCars']=dt['GarageCars'].fillna((dt['GarageCars'].mode()[0]))
#print(dt['GarageArea'].value_counts())
#print(dt.isnull().sum())
dt['GarageArea'] =dt['GarageArea'].fillna((dt['GarageArea'].mean()))
#print(dt.isnull().sum())
#print(dt.isnull().sum())

#print(dt['Functional'].value_counts())
dt['Functional']=dt['Functional'].fillna((dt['Functional'].mode()[0]))
#print(dt.isnull().sum())

dt['Utilities']=dt['Utilities'].fillna((dt['Utilities'].mode()[0]))
dt['MasVnrType']=dt['MasVnrType'].fillna((dt['MasVnrType'].mode()[0]))
#print(dt['MasVnrType'].value_counts())
dt['GarageArea'] =dt['GarageArea'].fillna((dt['GarageArea'].mean()))
df['MasVnrArea'] =df['MasVnrArea'].fillna((df['MasVnrArea'].mean()))
df['MasVnrArea']=df['MasVnrArea'].astype(int)
dt['MasVnrArea'] =dt['MasVnrArea'].fillna((dt['MasVnrArea'].mean()))

df['GarageArea'] =df['GarageArea'].fillna((df['GarageArea'].mean()))
df['GarageYrBlt']=df['GarageYrBlt'].astype(int)
dt['GarageYrBlt']=dt['GarageYrBlt'].astype(int)

df['BsmtFinSF1'] =df['BsmtFinSF1'].fillna((df['BsmtFinSF1'].mean()))
dt['BsmtFinSF1'] =dt['BsmtFinSF1'].fillna((dt['BsmtFinSF1'].mean()))
df['BsmtFinSF2'] =df['BsmtFinSF2'].fillna((df['BsmtFinSF2'].mean()))
df['BsmtUnfSF'] =df['BsmtUnfSF'].fillna((df['BsmtUnfSF'].mean()))
df['TotalBsmtSF'] =df['TotalBsmtSF'].fillna((df['TotalBsmtSF'].mean()))
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])

df['Exterior1st']=df['Exterior1st'].fillna((df['Exterior1st'].mode()[0]))
df['Exterior2nd']=df['Exterior2nd'].fillna((df['Exterior2nd'].mode()[0]))
df['MSZoning']=df['MSZoning'].fillna((df['MSZoning'].mode()[0]))
df['BsmtFullBath']=df['BsmtFullBath'].fillna((df['BsmtFullBath'].mode()[0]))
df['BsmtHalfBath']=df['BsmtHalfBath'].fillna((df['BsmtHalfBath'].mode()[0]))
df['BsmtHalfBath']=df['BsmtHalfBath'].fillna((df['BsmtHalfBath'].mode()[0]))
df['SaleType']=df['SaleType'].fillna((df['SaleType'].mode()[0]))


dt['TotalBsmtSF'] =dt['TotalBsmtSF'].fillna((dt['TotalBsmtSF'].mean()))
dt['BsmtFinSF2'] =dt['BsmtFinSF2'].fillna((dt['BsmtFinSF2'].mean()))
dt['BsmtUnfSF'] =dt['BsmtUnfSF'].fillna((dt['BsmtUnfSF'].mean()))

dt['KitchenQual']=dt['KitchenQual'].fillna((dt['KitchenQual'].mode()[0]))
dt['BsmtFinType2']=dt['BsmtFinType2'].fillna(dt['BsmtFinType2'].mode()[0])
dt['Exterior1st']=dt['Exterior1st'].fillna((dt['Exterior1st'].mode()[0]))
dt['Exterior2nd']=dt['Exterior2nd'].fillna((dt['Exterior2nd'].mode()[0]))
dt['MSZoning']=dt['MSZoning'].fillna((dt['MSZoning'].mode()[0]))
dt['BsmtFullBath']=dt['BsmtFullBath'].fillna((dt['BsmtFullBath'].mode()[0]))
dt['BsmtHalfBath']=dt['BsmtHalfBath'].fillna((dt['BsmtHalfBath'].mode()[0]))
dt['BsmtHalfBath']=dt['BsmtHalfBath'].fillna((dt['BsmtHalfBath'].mode()[0]))
dt['SaleType']=dt['SaleType'].fillna((dt['SaleType'].mode()[0]))
df['BsmtCond']=df['BsmtCond'].fillna((df['BsmtCond'].mode()[0]))
dt['BsmtCond']=dt['BsmtCond'].fillna((dt['BsmtCond'].mode()[0]))
dt['KitchenQual']=dt['KitchenQual'].fillna((dt['KitchenQual'].mode()[0]))


df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
dt['BsmtFinType1']=dt['BsmtFinType1'].fillna(dt['BsmtFinType1'].mode()[0])

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
dt['BsmtFinType2']=dt['BsmtFinType2'].fillna(dt['BsmtFinType2'].mode()[0])


df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
dt['Electrical']=dt['Electrical'].fillna(dt['Electrical'].mode()[0])

dt['BsmtFullBath']=dt['BsmtFullBath'].astype(int)
dt['BsmtHalfBath']=dt['BsmtHalfBath'].astype(int)
dt['TotalBsmtSF']=dt['TotalBsmtSF'].astype(int)
dt['TotalBsmtSF']=dt['TotalBsmtSF'].astype(int)
dt['BsmtFinSF2']=dt['BsmtFinSF2'].astype(int)

#print(dt.info())

#print(df['BsmtFinSF1'].isnull().sum())
#print(df.dropna(inplace = True))
#print(dt.dropna(inplace = True))
#print(df.shape)
#print(dt.shape)
#col = ['KitchenQual','SaleType']
#df['KitchenQual']=df['KitchenQual'].astype('category')
#df['KitchenQual']=df['KitchenQual'].cat.codes
#(df.dropna(inplace = True))
#(dt.dropna(inplace = True))
#print(df.shape)
#print(dt.shape)
ds=pd.concat([df, dt], axis=0)
d_cat = ds[['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
         'Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure',
         'BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
        'Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual',
         'GarageCond','PavedDrive']]
ds.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
         'Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure',
         'BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
        'Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual',
         'GarageCond','PavedDrive'],axis=1,inplace=True)

#print(d_cat)
#print(ds.shape)
le = LabelEncoder()

d_cat=d_cat.apply(le.fit_transform)


d_final =pd.concat([ds,d_cat],axis=1)
#print(d_final.shape)
# split the train and test


df_Train=d_final.iloc[:1460,:]
df_Test=d_final.iloc[1460:,:]
#print(df_Test)
x_test = df_Test.drop(['SalePrice'],axis=1)
#print(x_test)

# Split Features and Labes
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
#print(y_train)

# Model with XGBOOST , SELECT THE PERFECT PARAMETERS 

clf_xgb = xgb.XGBRegressor()

params = {"n_estimators" : [100,500, 700],
          "learning_rate" : [0.001,.01, 0.1],
          'colsample_bytree': [0.3, 0.7],
          "max_depth" : [5, 7,9],
          
          "gamma" : [0.5, 0.6, 0.7],
         }

gsc_xgb = GridSearchCV(clf_xgb, params )


gsc_xgb = gsc_xgb.fit(X_train,y_train)

#print(gsc_xgb.best_estimator_)
clf_xgb = gsc_xgb.best_estimator_
clf_xgb.fit(X_train,y_train)

#print("")
#print("Accuracy Score: " + str(round(clf_xgb.score(x,y), 4)))

pred_y = clf_xgb.predict(x_test)
#print(pred_y)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
#print(sub_df.shape)
#datasets=pd.concat([sub_df['Id'],pred_y],axis=1)
#datasets.columns=['Id','SalePrice']


submission = pd.DataFrame({
        "Id": sub_df['Id'],
        "SalePrice": pred_y
    })

submission.to_csv('submission1.csv',index = False)
#datasets.to_csv('submission.csv',index=False)

