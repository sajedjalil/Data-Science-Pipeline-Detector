#imports
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

#defs
def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == "object"]

def PerformOneHotEncoding(df,columnsToEncode):
    print('\tPerforming one-hot encoding on : ', columnsToEncode )
    return pd.get_dummies(df,columns = columnsToEncode)

def HandleMissingValues(df):
    # for Object columns fill using 'UNKOWN'
    # for Numeric columns fill using median
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    values = {}
    for a in cat_cols:
        values[a] = 'UNKOWN'

    for a in num_cols:
        values[a] = df[a].median()
        
    df.fillna(value=values,inplace=True)


#Set Configurations
TargetColumnName = 'SalePrice'        
TrainCSVPath = '../input/house-prices-advanced-regression-techniques/train.csv'
TestCSVPath = '../input/house-prices-advanced-regression-techniques/test.csv'
DropColumnsList = ['Id']   
key = 'Id'
cat_cols = []
    
#Read
train = pd.read_csv(TrainCSVPath) 
test  = pd.read_csv(TestCSVPath)
print('Data Reading Done...')
print('\tTrain Shape:{} \t \n\tTest Shape :{}'.format(train.shape,test.shape))

cat_cols = getObjectColumnsList(train) # Categorical columns : this will be used to perform one hot encoding.

# Dropping rows from Train Data where the target is missing
print('Dropping rows from Train Data where the target is missing')
train.dropna(axis=0, subset=[TargetColumnName], inplace=True)

trainLen = train.shape[0]

# Combine Test and Training sets to maintain consistancy. 
# Assumptions : Last column of the train df is the target column.
print('Combine Test and Training sets to maintain consistancy')
data=pd.concat([train.iloc[:,:-1],test],axis=0)
print('\tTrain Shape:{}\n\tTest Shape :{}\n\tComined Shape :{}'.format(train.shape,test.shape,data.shape))

# Dropping unwanted columns
print('Dropping Unwanted Columns : {}'.format(DropColumnsList))
data = data.drop(columns=DropColumnsList,axis=1)

# Missing Value Handling
print('Handling Missing Values')
HandleMissingValues(data)

# Check for any missing values
if (data.isnull().sum().sum() == 0):
    print('\tCheck Successful.. No missing Values in Data')
else:
    print('\tWarning ... Missing Values in Data')
    
#Categorical Feature Encoding
print('Categorical Feature Encoding')
data = PerformOneHotEncoding(data,cat_cols)

#spliting the data into train and test datasets
print('Spliting the data into train and test datasets')
train_data = data.iloc[:train.shape[0],:]
test_data  = data.iloc[train.shape[0]:,:]
print('\tTrain Shape:{}\n\tTest Shape :{}'.format(train_data.shape,test_data.shape))

# Get X,y for modelling
print('Get X,y for modelling')
X=train_data
y=train.loc[:,TargetColumnName]


print('train_test_split')
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
print('\t train_X Shape:{} \t train_y Shape :{}\n \t test_X Shape:{} \t test_y Shape :{}'.format(train_X.shape,train_y.shape,test_X.shape,test_y.shape))

'''
temp_X = pd.DataFrame(train_X['1stFlrSF'])
model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
model_xgb.fit(temp_X,train_y)
temp_predictions = model_xgb.predict(test_X['1stFlrSF'])
#print('\t\tRMSE:', np.sqrt(metrics.mean_squared_error(train_y, temp_predictions)))


print('Done \n\n')
'''
print('Predictive Modeling')

print('\tFit XGBRegressor')
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
model_xgb.fit(X, y)
predictions = model_xgb.predict(X)
print('\t\tMAE:', metrics.mean_absolute_error(y, predictions))
print('\t\tMSE:', metrics.mean_squared_error(y, predictions))
print('\t\tRMSE:', np.sqrt(metrics.mean_squared_error(y, predictions)))


print('Make the submission')
#make the submission data frame
Final_predictions = model_xgb.predict(test_data)
submission = {
    key: test[key].values,
    TargetColumnName: Final_predictions
}
solution = pd.DataFrame(submission)
solution.to_csv('submission.csv',index=False)

print('Done!..')