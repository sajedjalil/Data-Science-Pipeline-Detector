# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
#%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import xgboost as xgb

from sklearn import cross_validation, decomposition, grid_search
from sklearn.preprocessing import LabelEncoder

####################################################
# Functions                                        #
####################################################
# Remove outliers
def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values<=min_val, col_values>=max_val), np.NaN, col_values)
    
    return df 

##AGNE Example: df['age'].values is an array with all the values inside the column 'age'
##AGNE The method  numpy.where(condition[, x, y]) return elements, either from x or y, depending on condition.



# Home made One Hot Encoder
def convert_to_binary(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates()) 
    ##AGNE: Here I construct a list of all different values appearing in a given column to convert
    ##AGNE:  ex: for the column 'gender'  categories=['-unknown-', 'MALE', 'FEMALE', 'OTHER']
    
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert[:12] + '_' + cat_name[:10]
        df[col_name] = 0  ##AGNE: I initialize the new column to zero
        df.loc[(df[column_to_convert] == category), col_name] = 1 ##AGNE: df.loc[row_indexer,column_indexer]
    
    return df

# Count occurrences of value in a column
def convert_to_counts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates() ##AGNE: I create a list of all IDs appearing (they appear many times, but I want only one entry per ID)
    
    df_counts = df.loc[:,[id_col, column_to_convert]]
    df_counts['count'] = 1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=False, sort=False).sum()
    #grouped_df_counts = df_counts.groupby([id_col, column_to_convert], as_index=False, sort=False).sum() ##AGNE: my version!
 
    new_df = df_counts.pivot(index=id_col, columns=column_to_convert, values='count')
    #new_df = grouped_df_counts.pivot(index=id_col, columns=column_to_convert, values='count') ##AGNE: my version!
    new_df = new_df.fillna(0)
    
    # Rename Columns
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns = {category:col_name}, inplace=True)
        
    return new_df





####################################################
# Cleaning                                         #
####################################################

# Import data
print("Reading in data...")
tr_filepath = "../input/train_users_2.csv"
df_train = pd.read_csv(tr_filepath, header=0, index_col=None) ##AGNE: index_col=None means I have no label/names for the rows
te_filepath = "../input/test_users.csv"
df_test = pd.read_csv(te_filepath, header=0, index_col=None)



# Combine into one dataset

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)  
##AGNE: axis=0 means i concatenate along rows, 
##AGNE: ignore_index=True means I do not use the index values(=row names) along the concatenation axis

print(df_all.axes)
features=list(df_all.axes[1])
#print(features[0])
for feature in features:
    print(feature, df_all[feature].nunique())
for feature in features:
    if feature != 'id' and feature !='age' and feature!='date_account_created' and feature !='date_first_booking' and feature!='timestamp_first_active':  
        print(feature, df_all[feature].unique())


# Change Dates to consistent format

print("Fixing timestamps...")
df_all['date_account_created'] =  pd.to_datetime(df_all['date_account_created'], format='%Y-%m-%d')
##AGNE: format is what it is expected to find in the data 
df_all['timestamp_first_active'] =  pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all['date_account_created'].fillna(df_all.timestamp_first_active, inplace=True) ##AGNE: I do not see any empty row, bo??

##AGNE: NB: You may be wondering why this is necessary , after all cannot we all see what the dates are supposed to represent
## when we look at the data? The reason we need to convert the values in the date columns is that, if we want to do anything 
## with those dates (e.g. subtract one date from another, extract the month of the year from each date etc.),
## it will be far easier if Python recognizes the values as dates. This will become much clearer when we start adding various
## new features to the training data based on this date information.



# Remove date_first_booking column
df_all.drop('date_first_booking', axis=1, inplace=True)


# Fixing age column
print("Fixing age column...")
df_all = remove_outliers(df=df_all, column='age', min_val=15, max_val=90)
df_all['age'].fillna(-1, inplace=True)

# Fill first_affiliate_tracked column
print("Filling first_affiliate_tracked column...")
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)

#Fixing gender column
df_all['gender'].replace('-unknown-', np.nan, inplace=True)

#print(df_all['affiliate_channel'])

categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    df_all[categorical_feature] = df_all[categorical_feature].astype('category')
    

#print(df_all['affiliate_channel'].type)
#sns.despine()
df_all['gender'].value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Gender')
plt.savefig('gender_distribution.png')

