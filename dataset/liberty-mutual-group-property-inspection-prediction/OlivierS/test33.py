# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
import sqlite3

from math import sqrt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

numeric_cols= ['T1_V1','T1_V2','T1_V3','T1_V10','T1_V13','T1_V14','T2_V1','T2_V2','T2_V4','T2_V6','T2_V7','T2_V8','T2_V9','T2_V10','T2_V14','T2_V15']

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>

max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )		# not really needed

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train		# scale test by max_train

print(x_num_train)

# y

y_train = train.Hazard
#y_test = test.Hazard

# categorical

cat_train = train.drop( numeric_cols + [ 'Id', 'Hazard'], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'Id'], axis = 1 )

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))


print(x_train.shape)
print(x_test.shape)
