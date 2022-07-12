# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def splitFeaturesFromLabels(df):
    labels = df['y']
    del df['y']
    return df, labels

def prepColumn(df, columnName):
    uniqueList = pd.unique(df[columnName])
    for uniqueId in uniqueList:
        newColumnName = columnName + "_" + uniqueId
        df[newColumnName] = (df[columnName] == uniqueId) * 1

    del df[columnName]

    
def cleanupDataSet(df):
    #prepColumn(df, 'X0')
    prepColumn(df, 'X1')
    #prepColumn(df, 'X2')
    prepColumn(df, 'X3')
    prepColumn(df, 'X4')
    #prepColumn(df, 'X5')
    prepColumn(df, 'X6')
    prepColumn(df, 'X8')
    del df['X0']
    del df['X2']
    del df['X5']
    del df['X11']
    del df['X23']
    del df['X50']
    del df['X56']
    del df['X61']
    del df['X62']
    return df
    
mercedes_train = pd.read_csv('../input/train.csv')
mercedes_test = pd.read_csv('../input/test.csv')


mercedes_features_train, mercedes_labels_train = splitFeaturesFromLabels(mercedes_train)
mercedes_features_test = mercedes_test
mercedes_features_train = cleanupDataSet(mercedes_features_train)
mercedes_features_test = cleanupDataSet(mercedes_features_test)
trainNames = pd.unique(mercedes_features_train.columns.values)
testNames = pd.unique(mercedes_features_test.columns.values)
print("Items in train and not in test:")
for element in trainNames:
    if element not in testNames:
        print(element)
print("Items in test and not in train:")
for element in testNames:
    if element not in trainNames:
        print(element)
mercedes_features_train.to_csv("Show.csv", header=True, index=False)

# from sklearn import ensemble
# clf = ensemble.AdaBoostClassifier()
# clf.fit(mercedes_features_train, mercedes_labels_train)
# pred = clf.predict(mercedes_features_test)

from sklearn import linear_model
clf = linear_model.LinearRegression(normalize=False).fit(mercedes_features_train, mercedes_labels_train)
pred = clf.predict(mercedes_features_test)

# # import matplotlib.pyplot as plt
# # plt.plot(mercedes_features_train['y'], mercedes_features_train['Complexity'])

mercedes_results = pd.DataFrame()
mercedes_results['ID'] = mercedes_test['ID']
#mercedes_results['y'] = mercedes_train['y'].mean()
mercedes_results['y'] = pred
mercedes_results.to_csv("AdaBoost.csv", header=True, index=False)

import statsmodels.api as sm

X = mercedes_features_train
y = mercedes_labels_train

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())





