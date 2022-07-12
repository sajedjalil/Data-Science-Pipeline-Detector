# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import os
from collections import Counter
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve,ShuffleSplit
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import lightgbm as lgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


def teamWorkhandle(DF):
    DF["teamWork"] = DF['assists'] + DF['revives']
    DF.drop(['assists', 'revives'], axis=1, inplace=True)
    return DF


def itemsHandle(DF):
    DF["items"] = DF['boosts'] + DF['heals']
    DF.drop(['boosts', 'heals'], axis=1, inplace=True)
    return DF


def boostHealHandle(DF):
    DF["toolDistance"] = DF['rideDistance'] + DF['swimDistance']
    DF.drop(['rideDistance', 'swimDistance'], axis=1, inplace=True)
    return DF


def groupbyMatchAndGroup(orignal_DF,targets,aggtype):
    return orignal_DF.groupby(['matchId','groupId'])[targets].agg(aggtype)

def rankbyMatch(orignal_DF,targets):
    return orignal_DF.groupby('matchId')[targets].rank(pct=True).reset_index()

def Preprocessing(DF,is_train=True):
    DF = DF.dropna()
    y=None
    test_idx = None




    DF=teamWorkhandle(DF)
    DF=boostHealHandle(DF)
    DF=itemsHandle(DF)
    target = 'winPlacePerc'

    list_of_features=["teamWork", "items", "toolDistance",'damageDealt', 'DBNOs',
           'headshotKills', 'killPlace', 'killPoints', 'kills',
           'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
           'numGroups', 'rankPoints',  'roadKills',
           'teamKills', 'vehicleDestroys', 'walkDistance',
           'weaponsAcquired', 'winPoints']
    if is_train:
        print("Read Labels")
        groupedLabel=groupbyMatchAndGroup(DF,target,'first')
        y = np.array(groupedLabel, dtype=np.float64)

        finalOut = groupedLabel.reset_index()[['matchId','groupId']]
    else:
        finalOut = DF[['matchId','groupId']]
        test_idx = DF.Id
    #  mean and it' rankByMatch
    meanAccordingGroupAndMatch = groupbyMatchAndGroup(DF, list_of_features, 'mean')

    meanRankingByMatch = rankbyMatch(meanAccordingGroupAndMatch, list_of_features)
    finalOut = finalOut.merge(meanAccordingGroupAndMatch.reset_index(), how='left', on=['matchId', 'groupId'])
    finalOut = finalOut.merge(meanRankingByMatch, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    del meanAccordingGroupAndMatch,meanRankingByMatch

    #  max and it' rankByMatch
    maxAccordingGroupAndMatch = groupbyMatchAndGroup(DF, list_of_features, 'max')

    maxRankingByMatch = rankbyMatch(maxAccordingGroupAndMatch, list_of_features)

    finalOut = finalOut.merge(maxAccordingGroupAndMatch.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    finalOut = finalOut.merge(maxRankingByMatch, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    del maxAccordingGroupAndMatch,maxRankingByMatch

    minAccordingGroupAndMatch = groupbyMatchAndGroup(DF, list_of_features, 'min')

    minRankingByMatch = rankbyMatch(minAccordingGroupAndMatch, list_of_features)
    finalOut = finalOut.merge(minAccordingGroupAndMatch.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    finalOut = finalOut.merge(minRankingByMatch, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    del minAccordingGroupAndMatch,minRankingByMatch

    sizeAccordingGroupAndMatch = DF.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    finalOut = finalOut.merge(sizeAccordingGroupAndMatch, how='left', on=['matchId', 'groupId'])

    del sizeAccordingGroupAndMatch
    matchMean = DF.groupby(['matchId'])[list_of_features].agg('mean').reset_index()
    finalOut = finalOut.merge(matchMean, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    del matchMean
    matchSize = DF.groupby(['matchId']).size().reset_index(name='match_size')
    finalOut = finalOut.merge(matchSize, how='left', on=['matchId'])

    del matchSize

    X= finalOut.drop(columns=['groupId','matchId'])
    return X, y, test_idx


file_name_orignal = ["/kaggle/input/train_V2.csv", "/kaggle/input/test_V2.csv"]
Data_train = pd.read_csv(file_name_orignal[0])
# Data_train = reduce_mem_usage(Data_train)
Data_test = pd.read_csv(file_name_orignal[1])


Data_train.head()

# y = np.array(Data_train.groupby(['matchId','groupId'])['winPlacePerc'].agg('first'), dtype=np.float64)

# tag=['winPlacePerc']

# a=np.array(Data_train.head().groupby(['matchId','groupId'])[tag].agg('first'))
# iiii=Data_train.head().groupby(['matchId','groupId'])[tag].agg('first')

# iiii.reset_index()[['matchId','groupId']]
# tag2='winPlacePerc'
# np.array(Data_train.head().groupby(['matchId','groupId'])[tag2].agg('first'))

# y
Train_X, Train_y,_ = Preprocessing(Data_train)
Test_X, _, testId = Preprocessing(Data_test,False)
# Train_X1, Train_y1,_ = Preprocessing(Data_train.head())
# Test_X1, _, testId1 = Preprocessing(Data_test.head(),False)

# Train_X2, Train_y2,_ = Preprocessing2(Data_train.head())
# Test_X2, _, testId2 = Preprocessing2(Data_test.head(),False)

# Train_X1
# Train_X2

# Test_X.shape
# Train_X.shape
# Train_y.shape
# len(Train_X.columns)
# len(Train_y)

# len(Test_X.columns)
# Train_X, Train_y = Preprocessing(Data_train)

# Test_X=Preprocessing(Data_test,False)
# from sklearn.tree import DecisionTreeRegressor


''' DecisionTree Regression =========================================================== '''

# clf=DecisionTreeRegressor()
# clf.fit(Train_X, Train_y)

# predict_y=clf.predict(Test_X)

# test_Survived = pd.Series(predict_y, name="winPlacePerc")
# # test_Survived = pd.Series(omg, name="stroke_in_2018")
# results = pd.concat([testId,test_Survived],axis=1)


# results.to_csv("./oneclfsubmission.csv",index=False)

''' Linear Regression =========================================================== '''

# # 
# clf=LinearRegression()
# clf.fit(Train_X, Train_y)

# predict_y=clf.predict(Test_X)

# test_Survived = pd.Series(predict_y, name="winPlacePerc")
# # test_Survived = pd.Series(omg, name="stroke_in_2018")
# results = pd.concat([testId,test_Survived],axis=1)


# results.to_csv("./lineragresionSubmission.csv",index=False)


''' Lgb Regression =========================================================== '''


params={'learning_rate': 0.1,
        'objective':'mae',
        'metric':'mae',
        'num_leaves': 31,
        'verbose': 1,
        'random_state':42,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
       }

reg = lgb.LGBMRegressor(**params, n_estimators=10000)
reg.fit(Train_X, Train_y)
predict_y = reg.predict(Test_X, num_iteration=reg.best_iteration_)

test_predic = pd.Series(predict_y, name="winPlacePerc")

results = pd.concat([testId,test_predic],axis=1)


results.to_csv("./lgbdBsubmission.csv",index=False)
# # # RB

''' Random forest Regression (grid_search) =========================================================== '''

# random_state=0

# cv = ShuffleSplit(n_splits=5,test_size=0.25)


# rf_param_grid = {"max_depth": [None],
#               "max_features": [50,70],
#               "min_samples_split": [100,200],
#               "min_samples_leaf": [20],
#               "bootstrap": [False],
#               "n_estimators" :[20]
#                 }


# RFR = RandomForestRegressor()

# gsRFC = GridSearchCV(RFR,param_grid = rf_param_grid, cv=cv, scoring="neg_mean_absolute_error", verbose = 1)

# gsRFC.fit(Train_X,Train_y)

# RFC_best = gsRFC.best_estimator_

# # Best score
# print(gsRFC.best_score_)


# # RF.fit(Train_X, Train_y)
# predict_y=RFC_best.predict(Test_X)

# test_predic = pd.Series(predict_y, name="winPlacePerc")

# results = pd.concat([testId,test_predic],axis=1)


# results.to_csv("./RFgridSearchsubmission.csv",index=False)








