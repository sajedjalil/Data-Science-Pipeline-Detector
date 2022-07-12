# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
        
class DataFrameImputer2(TransformerMixin):

    def __init__(self):
        """Impute missing values.
    
        """

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)        

# The competition datafiles are in the directory ../input
# List the files we have available to work with
#print("> ls ../input")
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Read train data file:
train = pd.read_csv("../input/train.csv") #.fillna(method='ffill')  #.fillna(-999)
#test = pd.read_csv("../input/test.csv") #.fillna(-999)



# Write summaries of the train and test sets to the log
#print('\nSummary of train dataset:\n')
#print(train.describe())

#print(train.loc[1])

const_cols = [col for col in train.columns if len(train[col].unique()) == 1]
#print(const_cols) # ['VAR_0207', 'VAR_0213', 'VAR_0840', 'VAR_0847', 'VAR_1428']


#bool_cols = [col for col in train.columns if len(train[col].unique()) == 2]
#print(len(bool_cols)) # ['VAR_0207', 'VAR_0213', 'VAR_0840', 'VAR_0847', 'VAR_1428']
['VAR_0008', 'VAR_0009', 'VAR_0010', 'VAR_0011', 'VAR_0012', 'VAR_0018', 'VAR_0019', 'VAR_0020', 'VAR_0021', 'VAR_0022', 'VAR_0023', 'VAR_0024', 'VAR_0025', 'VAR_0026', 'VAR_0027', 'VAR_0028', 'VAR_0029', 'VAR_0030', 'VAR_0031', 'VAR_0032', 'VAR_0038', 'VAR_0039', 'VAR_0040', 'VAR_0041', 'VAR_0042', 'VAR_0043', 'VAR_0044', 'VAR_0188', 'VAR_0189', 'VAR_0190', 'VAR_0196', 'VAR_0197', 'VAR_0199', 'VAR_0202', 'VAR_0203', 'VAR_0215', 'VAR_0216', 'VAR_0221', 'VAR_0222', 'VAR_0223', 'VAR_0229', 'VAR_0239', 'VAR_0246', 'VAR_0394', 'VAR_0438', 'VAR_0446', 'VAR_0527', 'VAR_0528', 'VAR_0530', 'VAR_0563', 'VAR_0566', 'VAR_0567', 'VAR_0732', 'VAR_0733', 'VAR_0736', 'VAR_0737', 'VAR_0739', 'VAR_0740', 'VAR_0741', 'VAR_0924', 'VAR_1162', 'VAR_1163', 'VAR_1164', 'VAR_1165', 'VAR_1427', 'target']

#var_cols = [col for col in train.columns if len(train[col].unique()) > 5]
#print(len(var_cols)) # 1704

train = train.drop(const_cols, axis=1)
#test = test.drop(const_cols, axis=1)

cols_types = set()
for col in train.columns:
    cols_types.add(train[col].dtype)

print(cols_types)

train = DataFrameImputer2().fit_transform(train)

# encoding not digital data
lbl = preprocessing.LabelEncoder()
for col in train.columns:
    if  train[col].dtype == 'object':
    #if type(train[col][0]) is str:
        #lbl.fit(list(train[col]) + list(test[col]))
        lbl.fit(train[col].values)
        train[col] = lbl.transform(train[col])
#        test[col] = lbl.transform(test[col])


print(train.describe())

train.to_csv('train1.cssv', index = False)
#test.to_csv('test1.csv', index = False)
