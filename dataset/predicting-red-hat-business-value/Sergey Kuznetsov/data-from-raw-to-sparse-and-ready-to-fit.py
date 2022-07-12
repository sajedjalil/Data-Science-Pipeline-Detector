# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(' RAW DATA: ')
#print(check_output(["ls", "../input"]).decode("utf8"))
#print('')

# --- ADDITIONAL MODULES ---
# for data transformations:
import datetime
import scipy as sp
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.cross_validation import train_test_split

# --- INITIALIZING ---
np.random.seed(1337)

# --- READING RAW DATA ---

acttrain = pd.read_csv('../input/act_train.csv')
acttest = pd.read_csv('../input/act_test.csv')

acttest['outcome'] = -1

rawdata = pd.concat([acttrain, acttest], axis=0)
print(rawdata.head(2), '\n\n')

people = pd.read_csv('../input/people.csv')

# both DATA and PEOPLE contain char_<...> columns, so they are t be renamed
renamer = {}
for i in range(1, 39):
    renamer['char_{0}'.format(i)]='aiur_{0}'.format(i)
    
people.rename(columns=renamer, inplace=True)

data = pd.merge(rawdata, people, on='people_id')

categorical = []
for i in range(1,11):
    categorical.append('char_{0}'.format(i))
for i in range(1,10):
    categorical.append('aiur_{0}'.format(i))
categorical += ['activity_category', 'group_1']
print('CATEGORICAL VARIABLES:')
print(categorical, '\n')

logical = []
for i in range(10,38):
    logical.append('aiur_{0}'.format(i))
print('LOGICAL VARIABLES:')
print(logical, '\n')

chronological = ['date_x', 'date_y']
print('CHRONOLOGICAL VARIABLES:')
print(chronological, '\n')

# Changing data like "type 3"<str> ->  3<int>, "None" -> -1<int> 
def decategorize(x):
    if isinstance(x, str):
        return int(x.split(' ')[1])
    else:
        return -1

data[categorical]=data[categorical].applymap(decategorize)
print('Decategorized!')

# Changing (True, False) -> (1, 0)
data[logical]=data[logical].applymap(lambda x: int(x))
print('Bool->Int!')

# Changing data to number of days since the
#    beginning of the epoch (which is 1970-Jan-1)
def daysofepoch(x):
    dateattrs = list(map(lambda y: int(y), x.split('-')))
    d_o_e = (datetime.date(*dateattrs) - datetime.date(1970, 1, 1)).days
    return int(d_o_e)

data[chronological]=data[chronological].applymap(daysofepoch)
print('Date->Int!\n')

data.drop('people_id', axis='columns', inplace=True) #not needed anymore

print(data.iloc[9])
print(" It's now all numerical! (except for the activity identifier)\n")

# --- WORKING WITH NUMERIZED DATA ---

overdata = data[['activity_id', 'outcome']]
# these are to be separated
data.drop(['activity_id', 'outcome'], axis=1, inplace=True)

# introducing weekdays
data['date_x_7'] = data['date_x'] % 7
data['date_y_7'] = data['date_y'] % 7

# introducing delta_date
data['d_date'] = data['date_x'] - data['date_y']

# --- REMOVING MINOR (LEAST FREQUENT) CATEGORIES

# let's see how many we have
minority_threshold = 15
for c in categorical:
    cnts = data[c].value_counts()
    cntsmaj = cnts[cnts >= minority_threshold]
    print(c)
    print('Categories =', len(cnts))
    print('Major categories =', len(cntsmaj))
    print(' ')

# let's remove minor categories, replacing them with "-1" category
for c in categorical:
    cnts = data[c].value_counts()
    cntsmin = cnts[cnts < minority_threshold]
    data.loc[ [(d in cntsmin) for d in data[c]] , c] = -1

# let's create a mask that masks categorical features
ncols = data.shape[1]
print('Total columns now:', ncols, '\n')

mask = []
for i in range(ncols):
    if data.columns[i] in categorical:
        mask.append(True)
    else:
        mask.append(False)

inversemask=list(map(lambda x: not x, mask))

# --- RESCALING NON-CATEGORICAL FEATURES ---
# like dates or aiur_38
data = data.astype(np.float32)
scaler = MinMaxScaler()
data[data.columns[inversemask]] = data[
        data.columns[inversemask]].apply(lambda x: scaler.fit_transform(x))
        
print('Scaled!\n')

dataM = data.as_matrix()

# --- CATEGORICAL ONE HOT ENCODING -> SPARSE ---
#    (the most important part of this kernel!)

stacks = []
for i in range(dataM.shape[1]):
    colM = dataM[:,i]
    if mask[i]:     # binarizing categorical features and making them sparse
        binz = LabelBinarizer(sparse_output=True)
        binz.fit(colM)
        X = binz.transform(colM)
    else:       # making non-categorical features sparse while leaving values intact
        colM=colM.reshape((2695978, 1))
        X = sp.sparse.coo_matrix(colM)
    stacks.append(X)
    print('Column #{}'.format(i), X.shape)

print(' ')

bigX=sp.sparse.hstack(stacks, dtype=np.float32)

# --- TRAIN - TEST - VALUATION SPLIT ---

# Redefining data
data = bigX.tocsr().astype(np.float32)
# It's all sparse CSR now!
outcome = overdata['outcome'].astype(np.int16)

Ytrainval = outcome.loc[overdata['outcome'] != -1]
Ytest     = outcome.loc[overdata['outcome'] == -1]

overtrainval = Ytrainval.index
overtest     = Ytest.index

# Now we have our train+val and test indices. 
# Let's split train+val into train and val

overtrain, overval, Ytrain, Yval = train_test_split(overtrainval, Ytrainval, 
                    test_size=0.008, stratify=Ytrainval)

Ids = overdata[overdata['outcome'] == -1]['activity_id']

Xtrain = data[list(overtrain)]
Xval   = data[list(overval)]
Xtest  = data[list(overtest)]

for X in (Xtrain, Xval, Xtest):
    print(X.shape)
    
print('Done!')

# --- AND THERE IT IS, FRIENDS - A DATA TO FIT ---
# do something like
# model.fit(Xtrain, Ytrain)
# print(accuracy_score(Yval, model.predict(Xval)
# preds = model.predict(Xtest)
# result = pd.DataFrame({'activity_id' : list(Ids), 'outcome':  preds})
# submit(result)

# --- HERE'S AN EXAMPLE ---
#from sklearn.linear_model import SGDClassifier
#from sklearn.metrics import roc_auc_score

#clf = SGDClassifier(loss='log', alpha=0.0005)

# --- AND ANOTHER ONE ---
import xgboost

clf=xgboost.XGBClassifier(n_estimators=50, colsample_bytree=0.3)
clf.fit(Xtrain.tocsc(), Ytrain)
print('AUC on VALIDATION:', roc_auc_score(Yval, clf.predict(Xval.todense())))
preds = clf.predict(Xtest.tocsc())
result = pd.DataFrame({'activity_id' : list(Ids), 'outcome':  preds})
result.to_csv('output.csv', index=False)

# Good luck!

# Any results you write to the current directory are saved as output.

