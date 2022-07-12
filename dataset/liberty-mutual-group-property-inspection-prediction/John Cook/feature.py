# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
from sklearn import svm
from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import preprocessing

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write summaries of the train and test sets to the log
print('\nSummary of train dataset:\n')
print(train.describe())
print('\nSummary of test dataset:\n')
print(test.describe())

for j in range(len(train.columns)):
    if train[train.columns[j]].dtype == 'object':
        # Create groups
        by_group = train.groupby(train.columns[j])
        # find the mean of Hazard for each group
        by = by_group['Hazard'].mean()
        # Sort ascending so the smallest number has the smallest mean
        by.sort(ascending=True)
        rank = []
        for i in range(len(by)):
            rank.append(i)
        ind = []
        for i in range(len(by)):
            ind.append(by.index[i])
        rank_df = pd.DataFrame({train.columns[j] + '_rank' : rank,
#                                train.columns[j] + '_weight' : by,
                                train.columns[j] : ind
                                })
        
        train = pd.merge(train, rank_df, on = train.columns[j], how='left')
        test = pd.merge(test, rank_df, on = train.columns[j], how ='left')

# Keep only numeric variables
train_2_Num = train.loc[:, train.dtypes != 'object']
test_2_Num = test.loc[:, test.dtypes != 'object']

# Add hazard column so the two can be joined
test_2_Num['Hazard'] = -1

full = pd.concat([train_2_Num, test_2_Num])

full_trans = preprocessing.PolynomialFeatures(degree = 2).fit_transform(full.drop(['Id', 'Hazard'], axis=1))

train_trans = full_trans[:50999, ]
test_trans  = full_trans[51000:, ]

# perform ridge regression
ridgeR = linear_model.Ridge(alpha=0.3)
#score_rr = cv.cross_val_score(ridge, train_trans, train_2_Num['Hazard'], cv = 5, scoring='mean_squared_error')
ridgeR = ridgeR.fit(train_trans, train_2_Num['Hazard'])

ridgeR