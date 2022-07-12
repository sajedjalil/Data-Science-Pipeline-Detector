## HELP FUNCTIONS (from https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt)

## Function to reduce the DF size 
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def CalcOutliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return


#pip install -U pandas-profiling
#pip install 'git+https://github.com/joeddav/get_smarties.git#egg=get_smarties'



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
#import pandas_profiling
#from get_smarties import Smarties
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import  metrics


# read and merge 2 files with training data
df = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
df = reduce_mem_usage(df)
#df = df.sample(n=100000, random_state=1)
df_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
df_identity = reduce_mem_usage(df_identity)
df_keys = pd.merge(df, df_identity, how='left', on='TransactionID')
df_keys.describe()

print(f'Train dataset has {df_keys.shape[0]} rows and {df_keys.shape[1]} columns.')


# drop rows with NA target label
df_keys.dropna(axis=0, subset=['isFraud'], inplace=True)

#separate target and features
y = df_keys.isFraud
df_keys.drop(['isFraud'], axis=1, inplace=True)

# check if classes are imbalaned
sns.barplot([0,1], y.value_counts().values)
plt.title('Target variable count')

#drop unnecessary columns with ids
id_columns = ['TransactionID']
df_keys.drop(id_columns, axis=1, inplace=True)

# split into train and test datasets
X_train, X_valid, y_train, y_valid = train_test_split(df_keys, y, train_size=0.8, test_size=0.2, random_state=1)

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# calculate the percent of values in a column with missing values and  drop features with more than 80% NULL values 
percentage_null = X_train.isnull().sum() / len(X_train)
missing_features = percentage_null[percentage_null > 0.80].index
X_train.drop(missing_features, axis=1, inplace=True)


numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object'] #X_train[cname].nunique() < 10 and


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')  # 'mean', 'constant', 'most_frequent'

# encoding for binary columns 
#data_describe = data.describe(include=[object])
#nonbinary_columns = [c for c in non_numeric_columns if data_describe[c]['unique'] > 2]
#binary_columns    = [c for c in non_numeric_columns if data_describe[c]['unique'] ==2]
#for c in binary_columns[0:]:
#    top = data_describe[c]['top']
#    top_items = df_cat[c] == top
#    df_cat.loc[top_items, c] = 0
#    df_cat.loc[np.logical_not(top_items), c] = 1


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
	#('dummies', Smarties())
    
])


# OneHotEncoder VS get_dummies 
# OneHotEncoder cannot process string values directly. (if nominal features are strings, then they should be first mapped into integers)
# pandas.get_dummies only converts string columns into one-hot representation, unless columns are specified


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ])
	
# Define model
model = RandomForestClassifier(n_estimators=100, random_state=1)	


# Bundle preprocessing and modeling code in a pipeline
fraud_detection_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('model', model)
                                         ])

# Preprocessing of training data, fit model 
fraud_detection_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
predictions = fraud_detection_pipeline.predict(X_valid)

# Evaluate the model
roc1=metrics.roc_auc_score(y_train, fraud_detection_pipeline.predict(X_train))
roc2=metrics.roc_auc_score(y_valid, predictions)
print(roc1, roc2)

# running the model with the test data
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_data = pd.merge(test_transaction, test_identity, how='left', on='TransactionID')

final_predictions =  fraud_detection_pipeline.predict_proba(test_data)[:, 1]  

#Create a  DataFrame with predictions
submission = pd.DataFrame({'TransactionID': test_data['TransactionID'].tolist(), 'isFraud':final_predictions.tolist()})
filename = 'My submission Fraud Prediction.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

