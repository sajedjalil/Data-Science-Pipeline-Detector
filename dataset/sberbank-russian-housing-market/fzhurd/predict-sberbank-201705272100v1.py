# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_raw = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test_raw = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
macro_raw = pd.read_csv('../input//macro.csv', parse_dates=['timestamp']) 

#Join macro-economic data
train_full = pd.merge(train_raw, macro_raw, how='left', on='timestamp')
test_full = pd.merge(test_raw, macro_raw, how='left', on='timestamp')

from sklearn.preprocessing import LabelEncoder

def encode_object_features(train, test):
    '''(DataFrame, DataFrame) -> DataFrame, DataFrame
    
    Will encode each non-numerical column.
    '''
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    cols_to_encode = train.select_dtypes(include=['object'], exclude=['int64', 'float64']).columns
    for col in cols_to_encode:
        le = LabelEncoder()
        #Fit encoder
        le.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        #Transform
        train[col] = le.transform(list(train[col].values.astype('str')))
        test[col] = le.transform(list(test[col].values.astype('str')))
    
    return train, test

train_df, test_df = encode_object_features(train_full, test_full)

def add_date_features(df):
    '''(DataFrame) -> DataFrame
    
    Will add some specific columns based on the date
    of the sale.
    '''
    #Convert to datetime to make extraction easier
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    #Extract features
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['year'] = df['timestamp'].dt.year
    
    #These features inspired by Bruno's Notebook at https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
    #Month-Year
    month_year = df['timestamp'].dt.month + df['timestamp'].dt.year * 100
    month_year_map = month_year.value_counts().to_dict()
    df['month_year'] = month_year.map(month_year_map)
    #Week-Year
    week_year = df['timestamp'].dt.weekofyear + df['timestamp'].dt.year * 100
    week_year_map = week_year.value_counts().to_dict()
    df['week_year'] = week_year.map(week_year_map)
    df.drop('timestamp', axis=1, inplace=True)
    return df
    
def add_state_features(df):
    '''(DataFrame) -> DataFrame
    
    Add's features, meant to be used for both train and test df's.
    Does some operations to the state grouping
    '''
    #Get median of full sq by state
    df['state_median_full_sq'] = df['full_sq'].groupby(df['state']).transform('median')
    #Build features from full sq median by state
    df['full_sq_state_median_diff'] = df['full_sq'] - df['state_median_full_sq']
    df['life_sq_state_median_full_diff'] = df['life_sq'] - df['state_median_full_sq']
    #Drop helper columns
    df.drop('state_median_full_sq', axis=1, inplace=True)
    
    return df
    
    
def add_features(df):
    '''(DataFrame) -> DataFrame
    
    Add's features, meant to be used for both train and test df's.
    '''
    #Floor
    df['floor_ratio'] = df['floor'] / df['max_floor'].astype(float)
    df['floor_from_top'] = df['max_floor'] - df['floor']
    #Sq areas
    df['kitch_sq_ratio'] = df['kitch_sq'] / df['full_sq'].astype(float)
    df['life_sq_ratio'] = df['life_sq'] / df['full_sq'].astype(float)
    df['full_sq_per_room'] = df['full_sq'] / df['num_room'].astype(float)
    df['life_sq_per_room'] = df['life_sq'] / df['num_room'].astype(float)
    df['full_living_sq_diff'] = df['full_sq'] - df['life_sq']
    #df['full_sq_per_floor'] = df['full_sq'] / df['max_floor'].astype(float) #No value added
    df = add_date_features(df)
    df = add_state_features(df)
    df['build_year_vs_year_diff'] = df['build_year'] - df['year']  #no change
    
    #Drop Id -> Made it worse
    #df.drop('id', axis=1, inplace=True)
    
    #School Variables -> Made it worse
    #df['preschool_quota_ratio'] = df["children_preschool"] / df["preschool_quota"].astype("float")
    #df['school_quota_ratio'] = df["children_school"] / df["school_quota"].astype("float")
    return df
    
train_df = add_features(train_df)
test_df = add_features(test_df)

train_df.shape

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

#Get Data
#Y_train = train_df['price_doc'].values
Y_train = np.log1p(train_df['price_doc'].values)
X_train = train_df.ix[:, train_df.columns != 'price_doc'].values
X_test = test_df.values

#Initialize Model
xgb = XGBRegressor()
#Create cross-validation
cv = TimeSeriesSplit(n_splits=5)
#Train & Test Model
cross_val_results = cross_val_score(xgb, X_train, Y_train, cv=cv, scoring='neg_mean_squared_error')
print(cross_val_results.mean())


model = xgb.fit(X_train, Y_train)
model.feature_importances_;

from xgboost import XGBRegressor

#Get Data
Y_train = train_df['price_doc'].values
X_train = train_df.ix[:, train_df.columns != 'price_doc'].values
X_test = test_df.values
#Init Model
xgb = XGBRegressor(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.7)
#Train Model
model = xgb.fit(X_train, Y_train)
#Make Predictions
predictions = xgb.predict(X_test)


#Make Submission File
submission_df = pd.DataFrame({'id':test_full['id'], 'price_doc':predictions})
submission_df.to_csv('xgb-added_features.csv', index=False)

