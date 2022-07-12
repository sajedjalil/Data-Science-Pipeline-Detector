
## With this simple archicture I could achieve a RMSE of 0.15 on leaderboard with only numerical data
## With a deeper autoe-encoder architecture  ,the result would probably be better. (DM me if you achieve better with a more complex architecture).



#### IMPORTANT ####
## I inspire the preprocessing from :
##https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard 
##as I wanted to focus on modeling mostly





## Please consider giving a star if  you like the code 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats as st
from scipy.stats import norm, skew
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics


## NN Libraries
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers

## Optimizer and sklearn libraries
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV


from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
import optuna





########################
## AUTOENCODER CLASS  ##
########################
class MyModel(tf.keras.Model):

	def __init__(self):
		super(MyModel, self).__init__()
		self.dense1 = tf.keras.layers.Dense(59, activation=tf.nn.tanh ,use_bias = True)
		self.dense2 = tf.keras.layers.Dense(50, activation=tf.nn.tanh,use_bias = True)
		self.dense3 = tf.keras.layers.Dense(40, activation=tf.nn.tanh,use_bias = True)

		self.dense4 = tf.keras.layers.Dense(30, activation=tf.nn.tanh,use_bias = True)
		self.dense5 = tf.keras.layers.Dense(20, activation=tf.nn.tanh,use_bias = True)
		self.dense6 = tf.keras.layers.Dense(15, activation=tf.nn.tanh,use_bias = True)
		self.dense7 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,use_bias = True)

		self.dense8 = tf.keras.layers.Dense(5, activation=tf.nn.tanh,use_bias = True)
		self.dense9 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,use_bias = True)


		self.dense10 = tf.keras.layers.Dense(15, activation=tf.nn.tanh,use_bias = True)

		self.dense11 = tf.keras.layers.Dense(20, activation=tf.nn.tanh,use_bias = True)

		self.dense12 = tf.keras.layers.Dense(40, activation=tf.nn.tanh,use_bias = True)

		self.dense13 = tf.keras.layers.Dense(50, activation=tf.nn.tanh,use_bias = True)

		self.dense14 = tf.keras.layers.Dense(59, activation=tf.nn.tanh,use_bias = True)


	def encode(self,X):
		
		x= self.dense1(X)
		x= self.dense2(x)
		x= self.dense3(x)
		x= self.dense4(x)
		x= self.dense5(x)
		x= self.dense6(x)
		x= self.dense7(x)
	 
		return self.dense8(x)

	def decode(self,H):
		return self.dense14(H)

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		x = self.dense4(x)
		x = self.dense5(x)
		x = self.dense6(x)
		x = self.dense7(x)
		x = self.dense8(x)
		x = self.dense9(x)
		x = self.dense10(x)
		x = self.dense11(x)
		x = self.dense12(x)
		x= self.dense13(x)
		
		

		return self.dense14(x)


## Please consider installing the package : pip install optuna
########################
##     Optimizer      ##
########################


def optimize(trial,x,y,regressor):

  if (regressor=="random_forest"):

    criterion = trial.suggest_categorical("criterion", ["mse","mae"])
    n_estimators = trial.suggest_int("n_estimators",100,1000)
    max_depth = trial.suggest_int("max_depth",3,30)
    max_features = trial.suggest_categorical("max_features",["sqrt","auto","log2"])

    model = ensemble.RandomForestRegressor(
      criterion=criterion,
      n_estimators=n_estimators,
      max_depth=max_depth,
      max_features=max_features
  )
  elif (regressor=="SVM"): ##SVM
    kernel = trial.suggest_categorical("kernel", ["rbf","linear","poly"])
    gamma = trial.suggest_categorical("gamma", ["scale","auto"])
    coef0 = trial.suggest_int("coef0",1,50)
    degree = trial.suggest_int("degree",1,4)


    model = SVR(
      kernel=kernel,
      gamma=gamma,
      coef0=coef0,
      degree=degree
  )
  else: ## XGboost
  # Int parameter
    max_depth = trial.suggest_int("max_depth",3,30)

    n_estimators = trial.suggest_int("n_estimators",100,3000)

    max_leaves= trial.suggest_int("max_leaves",1,10)
  # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.09)
  # Uniform parameter
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.0, 1.0) 

    gamma = trial.suggest_uniform('gamma', 0.0, 0.05)  

    min_child_weight = trial.suggest_uniform('min_child_weight',1,3)

    reg_lambda = trial.suggest_uniform('reg_lambda',0.5,1)
  

    model = xgb.XGBRegressor(
      objective ='reg:squarederror',
      
      n_estimators=n_estimators,
      max_depth=max_depth,
      learning_rate=learning_rate,
      colsample_bytree=colsample_bytree,
      gamma=gamma,
      min_child_weight=min_child_weight,
      reg_lambda=reg_lambda,
      max_leaves=max_leaves

  )

      

  kf=model_selection.KFold(n_splits=5)
  error=[]
  for idx in kf.split(X=x , y=y):
    train_idx , test_idx= idx[0],idx[1]
    xtrain=x[train_idx]
    ytrain=y[train_idx]
    xtest=x[test_idx]
    ytest=y[test_idx]   
    model.fit(x,y)
    y_pred = model.predict(xtest)
    fold_err = metrics.mean_squared_error(ytest,y_pred)
    error.append(fold_err)
  return 1.0 * np.mean(error)
    
    
    
    
    
    
    
    
    


def preprocessing(train ,test ):

	train_ID = train['Id']
	test_ID = test['Id']

	train.drop("Id", axis = 1, inplace = True)
	test.drop("Id", axis = 1, inplace = True)


	train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

	## Log transform the target ,because its distribution is skewed
	train["SalePrice"] = np.log1p(train["SalePrice"])

	ntrain = train.shape[0]
	ntest = test.shape[0]
	y_train = train.SalePrice.values
	all_data = pd.concat((train, test)).reset_index(drop=True)
	all_data.drop(['SalePrice'], axis=1, inplace=True)


	############################
	#Dealing with missing data#
	############################

	all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
	all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
	all_data["Alley"] = all_data["Alley"].fillna("None")
	all_data["Fence"] = all_data["Fence"].fillna("None")
	all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
	all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
		lambda x: x.fillna(x.median()))
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
			all_data[col] = all_data[col].fillna('None')
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
			all_data[col] = all_data[col].fillna(0)
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
			all_data[col] = all_data[col].fillna(0)
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
			all_data[col] = all_data[col].fillna('None')
	all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
	all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

	all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
	all_data = all_data.drop(['Utilities'], axis=1)
	all_data["Functional"] = all_data["Functional"].fillna("Typ")
	all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

	all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

	all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
	all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

	all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
	#MSSubClass=The building class
	all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
	all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
	all_data['YrSold'] = all_data['YrSold'].astype(str)
	all_data['MoSold'] = all_data['MoSold'].astype(str)


	cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
				'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
				'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
				'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
				'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
	for c in cols:
			lbl = LabelEncoder() 
			lbl.fit(list(all_data[c].values)) 
			all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
	print('Shape all_data: {}'.format(all_data.shape))

	all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

	all_data = pd.get_dummies(all_data)
	print(all_data.shape)

	train = all_data[:ntrain]
	test = all_data[ntrain:]

	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

	newdf = all_data.select_dtypes(include=numerics)

	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

	newdf = all_data.select_dtypes(include=numerics)
	sc = StandardScaler()
	all_data_encoded = sc.fit_transform(newdf)


	np.save("data.npy",all_data_encoded)
	np.save("y_train.npy",y_train)


    
    





# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/house-prices-advanced-regression-techniques'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
        test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
        train_ID = train['Id']
        test_ID = test['Id']
        print(train.head())
        
        
        #########################
        ##   Preprocess data   ##   I only considered numerics data
        #########################


        preprocessing(train,test)
        
        
        
        ############################
        ## Traning the autoencoder##
        ############################
        all_data_encoded = np.load("data.npy")


        model=MyModel()
        model.compile(
        optimizer="adam",
        loss="mean_squared_error"  
        )
        num_epochs = 10000
        batch_size = 64

        history = model.fit(x=all_data_encoded, y=all_data_encoded,
					epochs=num_epochs,
					batch_size=batch_size,
					shuffle=True,
					
					verbose=1)

        np.save("latent.npy",model.encode(all_data_encoded).numpy())
        
         ########################
         ##    Predictions     ##
         ########################
            
        latent_features = np.load("latent.npy")
        y_train = np.load("y_train.npy")
        train = latent_features[:1458] #n_train
        test = latent_features[1458:] 




        ###################
        ## Random Forest ##
        ###################

        optimization_function = partial(optimize , x=train,y=y_train,regressor="random_forest")
        study = optuna.create_study(direction="minimize")
        study.optimize(optimization_function,n_trials=100)

        ###################
        ##  	SVM 	 ##
        ###################

        optimization_function = partial(optimize , x=train,y=y_train,regressor="SVM")
        study = optuna.create_study(direction="minimize")
        study.optimize(optimization_function,n_trials=100)

        #{'criterion': 'mse', 'n_estimators': 522, 'max_depth': 26, 'max_features': 'log2'}

        model_rf = ensemble.RandomForestRegressor(
		criterion="mse",
		n_estimators=522,
		max_depth=26,
		max_features='log2')


        #{'kernel': 'rbf', 'gamma': 'auto', 'coef0': 24, 'degree': 1}
        model_svm = SVR(
		
		 coef0= 15,
		 degree= 3,
			gamma= 'scale',
			 kernel= 'poly')

        model_svm.fit(train,y_train)
        model_rf.fit(train,y_train)



        y_pred_svm = np.expm1(model_svm.predict(test))

        y_pred_rf = np.expm1(model_rf.predict(test))
        
        #################
        ## ensembling  ##
        #################

        ensemble = 0.7*y_pred_rf + 0.3*y_pred_svm

        sub = pd.DataFrame()
        sub['Id'] = test_ID
        sub['SalePrice'] = ensemble
        sub.to_csv('submission.csv',index=False)

        print("Finish.")
            
            
        
        
        