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
# Author - Stanislav Pogodin
# Email - stas.pogodin@gmail.com

# This script was created for participation in Kaggle Competition


import time

from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from sklearn.ensemble import GradientBoostingClassifier

init_time = time.clock()

# Let's get some data we need
df_train = pd.read_csv('../input/train.csv', header=0, delimiter=',')
df_test = pd.read_csv('../input/test.csv', header=0, delimiter=',')

print ("Data loaded.") 

# simple function to extract column names with text attributes
def catColumns(DataFrame):
    # function takes a DataFrame as argument
    # and returns a list of columns with data in category format

    cat_col = []

    for col in DataFrame.columns.values:

        if type(df_train[col][0]) == str or type(df_train[col][1]) == str:
            cat_col.append(col)

    return cat_col

y_train = df_train['target'].values
train_ID = df_train['ID'].values
test_ID = df_test['ID'].values

# function to transform raw data frame with text attributes
# to sparse matrix with numeric attributes only
def dataTransformation(DF_train, DF_test):
    """
    function takes train and test data frames with text arguments as an input
    and returns sparse matrices prepared to train and test
    """

    # check is it a train or test DF to drop correct columns
    # and assign appropriate value to the variable
    # try:
    DF_train = DF_train.drop(['ID', 'target'], 1)
    DF_test = DF_test.drop('ID', 1)

    df_cat_train = DF_train[catColumns(DF_train)]  # data frame with categorized attributes
    df_cat_test = DF_test[catColumns(DF_test)]
    df_numeric_train = DF_train.drop(catColumns(DF_train), 1)  # data frame with numeric attributes
    df_numeric_test = DF_test.drop(catColumns(DF_test), 1)

    na_numeric_train = df_numeric_train.fillna(0).values  # replacing empty numeric attributes with zeros
    na_numeric_test = df_numeric_test.fillna(0).values
    df_cat_train = df_cat_train.fillna('nan')  # replacing empty category attributes with 'nan'
    df_cat_test = df_cat_test.fillna('nan')

    # we will transform text attributes to numeric format using DictVectorizer class
    # this approach is known as "one-hot coding"
    DV = DictVectorizer()
    s_cat_num_train = DV.fit_transform(df_cat_train.to_dict('records'))
    s_cat_num_test = DV.transform(df_cat_test.to_dict('records'))

    s_numeric_train = sparse.csr_matrix(na_numeric_train)  # creating sparse matrix - takes less memory
    s_numeric_test = sparse.csr_matrix(na_numeric_test)

    X_train = sparse.hstack([s_numeric_train, s_cat_num_train])
    X_test = sparse.hstack([s_numeric_test, s_cat_num_test])

    print ("data shape train:", X_train._shape)
    print ("data shape test:", X_test._shape)
    return X_train, X_test

print ("Transforming train and test datasets...", time.clock() - init_time, 's')
X_train, X_test = dataTransformation(df_train, df_test)
X_test = X_test.tocsr()  # transform coo_matrix to csr_

train_X_train, train_X_test, train_y_train, train_y_test = \
    train_test_split(X_train, y_train, test_size=0.2, random_state=241)  # splitting train data to assess models

# Gradient boosting model
# loops to get best parameters
"""
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    for n_est in [10, 15, 20, 25, 30, 50, 100, 150, 200, 250]:
        for rms in [57, 112, 182, 241]:

            GBC = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=rms)

            print()
            print ('Fitting GBC model for lr=' + str(lr) + ' and n_est=' + str(
                n_est)+ ', rms= '+str(rms) + '...', time.clock() - init_time, 's')

            GBC.fit(train_X_train, train_y_train)
            print ('Predicting values, GBC model ...', time.clock() - init_time, 's')
            y_test_prob = []

            for row in range(0, train_X_test._shape[0]):  # memory error when trying to convert full matrix - that's why loop
                y_test_prob.append(GBC.predict_proba(train_X_test[row, :].toarray())[:, 1])

            y_test_prob = np.array(y_test_prob)

            print ('Predicting finished.', time.clock() - init_time, 's')
            print ("LogLoss =", log_loss(train_y_test, y_test_prob))
"""

# getting results for submission
# best log_loss for learning rate = 0.3 and n_estimators = 150
GBC = GradientBoostingClassifier(n_estimators=150, learning_rate=0.3, random_state=241)

print ()
print ('Fitting GBC model for lr=0.3' + ' and n_est=150' + '...', time.clock() - init_time, 's')

GBC.fit(X_train, y_train)
print ('Predicting values, GBC model ...', time.clock() - init_time, 's')
y_test_prob = []

for row in range(0, X_test._shape[0]):
    y_test_prob.append(GBC.predict_proba(X_test[row, :].toarray())[:, 1])  # memory error when trying to convert full matrix - that's why loop

print ('Predicting finished.', time.clock() - init_time, 's')

# writing output for submission
result = open('result.csv', 'w')
result.write('ID,PredictedProb' + '\n')
for i in range(0, len(test_ID)):
    result.write(str(test_ID[i]) + ',' + str(y_test_prob[i][0])+'\n')

result.close()
