# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

df = train
encoded_df = df[df.columns[(df.dtypes!='object') & (df.columns!=u'Hazard') & (df.columns!=u'Id') ]]
encoded_df.drop('T2_V10', axis=1, inplace=True)
encoded_df.drop('T2_V7', axis=1, inplace=True)
encoded_df.drop('T1_V13', axis=1, inplace=True)
encoded_df.drop('T1_V10', axis=1, inplace=True)

for name in df.columns[(df.dtypes=='object') & (df.columns!=u'Hazard') & (df.columns!=u'Id')]:
    
    encoded_column = pd.get_dummies(train[name],prefix=name)
    encoded_df = encoded_df.join(encoded_column.ix[:, :])
    


#encoding test data frame
df = test
encoded_test_df = df[df.columns[(df.dtypes!='object') & (df.columns!=u'Hazard') & (df.columns!=u'Id') ]]

encoded_test_df.drop('T2_V10', axis=1, inplace=True)
encoded_test_df.drop('T2_V7', axis=1, inplace=True)
encoded_test_df.drop('T1_V13', axis=1, inplace=True)
encoded_test_df.drop('T1_V10', axis=1, inplace=True)



for name in df.columns[(df.dtypes=='object') & (df.columns!=u'Hazard')]:
    
    encoded_column = pd.get_dummies(test[name],prefix=name)
    encoded_test_df = encoded_test_df.join(encoded_column.ix[:, :])



forest = RandomForestRegressor(n_estimators = 300, n_jobs=-1, max_depth=400, oob_score=False)
forest.fit(encoded_df,train.Hazard)

encoded_test_df['Hazard'] =  forest.predict(encoded_test_df)
encoded_test_df['Hazard'].to_csv('python_random_forest_sample.csv')
