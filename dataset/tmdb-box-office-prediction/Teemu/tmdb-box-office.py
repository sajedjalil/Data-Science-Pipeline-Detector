#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import datetime
# Data:
train = pd.read_csv('../input/train.csv', parse_dates=["release_date"])
test = pd.read_csv('../input/test.csv')
#empty values in the dataframe per column
train.isnull().any()
#Great overview of cutting columns: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
#Let's remove unnecessary columsn not suitable for our model:
train = train[['id', 'budget', 'popularity', 'runtime',  'revenue']]
#convert categorical data into numbers
train_cat = pd.get_dummies(train)
variables = train_cat.drop(['revenue'], axis=1)
target = train_cat['revenue']
#Release date to year
train['release_date'] = pd.to_datetime(train['release_date'], format='%m/%d/%y')
train['release_year'] = train.release_date.dt.year

#sns.kdeplot(target)
#replace variables with empty values with mean:
variables = variables.fillna(variables.mean())
#skewed features
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.6]
skewed_feats 
#### XGBoost Model###
#X_train and y_train
X_train = variables
y_train = target
#X_test
X_test = test[['id', 'budget', 'popularity', 'runtime']]

# Separate targetvariable and the rest of the variables
import xgboost as xgb
df_train = xgb.DMatrix(X_train, y_train)
df_test = xgb.DMatrix(X_test)
params = {"max_depth":6, "eta":0.01}
model = xgb.cv(params, df_train,  num_boost_round=700, early_stopping_rounds=100)
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.01) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y_train)
xgb_preds = model_xgb.predict(X_test)

#Insert to CSV file the solution values with id and SalesPrice columns
solution = pd.DataFrame({"id":test.id, "revenue":xgb_preds})
solution.to_csv("revenue_estimates.csv", index = False)






#To follow up, here very interesting:
# https://cloud.google.com/blog/products/ai-machine-learning/how-20th-century-fox-uses-ml-to-predict-a-movie-audience
