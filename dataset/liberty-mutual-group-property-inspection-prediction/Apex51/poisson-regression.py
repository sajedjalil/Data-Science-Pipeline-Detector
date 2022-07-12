# # Note: Kaggle only runs Python 3, not Python 2

# # We'll use the pandas library to read CSV files into dataframes
# import pandas as pd

# # The competition datafiles are in the directory ../input
# # Read competition data files:
# train = pd.read_csv("../input/train.csv")
# test  = pd.read_csv("../input/test.csv")

# # Write summaries of the train and test sets to the log
# print('\nSummary of train dataset:\n')
# print(train.describe())
# print('\nSummary of test dataset:\n')
# print(test.describe())

import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import PCA

##################################################################################
# cal metric

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

# gini wrapper for xgboost
# return 0 - gini score for decrease purpose
def gini_xgb(pred, dtrain):
    target = dtrain.get_label()
    err = 0. - normalized_gini(target, pred)
    return 'gini', err

##################################################################################
#load train and test 

train  = pd.read_csv("../input/train.csv", index_col=0)
test  = pd.read_csv("../input/test.csv", index_col=0)
train_y = np.array(train.Hazard) - 1

# drop train_y -> train_y
train.drop('Hazard', axis=1, inplace=True)
# # drop noisy features
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)
# train.drop('T1_V6', axis=1, inplace=True)

# train.drop('T2_V11', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)
# test.drop('T1_V6', axis=1, inplace=True)

# test.drop('T2_V11', axis=1, inplace=True)

# columns and index for later use
columns = train.columns
test_ind = test.index

columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

train = train.astype(np.int64)
test = test.astype(np.int64)


train_x_sp, test_x_sp, train_y_sp, test_y_sp = train_test_split(train, train_y, train_size=0.8, random_state=50)

params = {}
params["objective"] = "count:poisson"
params["eta"] = 0.01
params["max_depth"] = 7
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["min_child_weight"] = 5
params["silent"] = 1

plst = list(params.items())

num_rounds = 100000

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train_x_sp, label=train_y_sp)
xgval = xgb.DMatrix(test_x_sp, label=test_y_sp)

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
rgrs = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

print('gini score is {}'.format(normalized_gini(test_y_sp, rgrs.predict(xgval))))


