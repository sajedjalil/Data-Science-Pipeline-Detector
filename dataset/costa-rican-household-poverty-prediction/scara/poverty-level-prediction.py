# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pylab as pl
import math
import random
from sklearn import linear_model
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import tree
from sklearn import svm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def read():
    # clean data
    mapping = {"yes": 1, "no": 0}
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    train.head()
    # Apply same operation to both train and test
    for df in [train, test]:
        clean_data(df)
        add_features(df)
        drop_columns(df)

    for df in [train, test]:
        # Fill in the values with the correct mapping
        df['dependency'] = np.sqrt(df['SQBdependency'])
        df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
        df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)
    
    # Groupby the household and figure out the number of unique values
    all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    households_leader = train.groupby('idhogar')['parentesco1'].sum()
    # Find households without a head
    households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]
     
    # Iterate through each household
    for household in not_equal.index:
        # Find the correct label (for the head of household)
        true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

        # Set the correct label for all members in the household
        train.loc[train['idhogar'] == household, 'Target'] = true_target
        
    train = train.fillna(0);
    test = test.fillna(0);
    train = train.drop(columns = ["Id", "idhogar"])
    test = test.drop(columns = ["Id", "idhogar"])
    return train, test

def drop_columns(df):
    df = df.drop(columns = ['tamhog', 'tamviv', 'r4t3', 'hogar_total'])
    df = df.drop(columns = ['male'])
    df = df.drop(columns = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq'])
    
def clean_data(train):
    train['walls'] = np.argmax(np.array(train[['epared1', 'epared2', 'epared3']]),
                               axis = 1)
    train = train.drop(columns = ['epared1', 'epared2', 'epared3'])
    train['roof'] = np.argmax(np.array(train[['etecho1', 'etecho2', 'etecho3']]),
                               axis = 1)
    train = train.drop(columns = ['etecho1', 'etecho2', 'etecho3'])
    train['floor'] = np.argmax(np.array(train[['eviv1', 'eviv2', 'eviv3']]),
                               axis = 1)
    train = train.drop(columns = ['eviv1', 'eviv2', 'eviv3'])
    train['education'] = np.argmax(np.array(train[[c for c in train if c.startswith('instl')]]), axis = 1)
    train = train.drop(columns = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'])
    water = []
    for i, row in train.iterrows():
        if row['abastaguano'] == 1:
            water.append(0)
        elif row['abastaguadentro'] == 1 or row['abastaguafuera'] == 1:
            water.append(1)
        else:
            water.append(0)
            
    train['water'] = water
    train = train.drop(columns = ['abastaguano', 'abastaguadentro', 'abastaguafuera'])
    
    cooking = []
    for i, row in train.iterrows():
        if row['energcocinar1'] == 1:
            cooking.append(0)
        elif row['energcocinar2'] == 1 or row['energcocinar3'] == 1 or row['energcocinar4'] == 1:
            cooking.append(1)
        else:
            cooking.append(0)
            
    train['cooking'] = cooking
    train = train.drop(columns = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4'])
    
    toilet = []
    for i, row in train.iterrows():
        if row['sanitario1'] == 1:
            toilet.append(0)
        elif row['sanitario2'] == 1 or row['sanitario3'] == 1 or row['sanitario5'] == 1 or row['sanitario6'] == 1:
            toilet.append(1)
        else:
            toilet.append(0)
            
    train['toilet'] = toilet
    train = train.drop(columns = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6'])
    
    electricity = []
    for i, row in train.iterrows():
        if row['noelec'] == 1:
            electricity.append(0)
        elif row['public'] == 1 or row['planpri'] == 1 or row['coopele'] == 1:
            electricity.append(1)
        else:
            electricity.append(0)
            
    train['electricity'] = electricity
    train = train.drop(columns = ['noelec', 'public', 'planpri', 'coopele'])

def add_features(df):
    aggregationType = {
        'escolari': 'mean', #maybe average/min? years of schooling
        'rez_esc': 'mean' #maybe average/min?, #Years behind in school
    }
    aggregated_df = df.groupby(['idhogar']).agg(aggregationType)
    def aggregateFn(row, key): 
        return aggregated_df.loc[row['idhogar']][key]
        
    for key in aggregationType.keys():
        df[key + "_agg"] = df.apply(lambda row: aggregateFn(row, key), axis=1)
    
    #print(df)
def household_features(df):
    # Groupby the household and figure out the number of unique values
    all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    households_leader = train.groupby('idhogar')['parentesco1'].sum()
    # Find households without a head
    households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]
     
    # Iterate through each household
    for household in not_equal.index:
        # Find the correct label (for the head of household)
        true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

        # Set the correct label for all members in the household
        train.loc[train['idhogar'] == household, 'Target'] = true_target
        
###======================================reading================================================
train, test = read()
#X = train

X = train.drop(columns = ["Target"])
y = train["Target"]


# print(X)
# print(y)
#X_train, X_test, y_train, y_test = train_test_split(X, y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = train.drop(columns = ["Target"])
y_train = train["Target"]
X_test = test
###======================================normalize================================================
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

test = sc.transform(test)

###======================================feature selection================================================
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

##lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
##model = SelectFromModel(lsvc, prefit=True)
##X_train = model.transform(X_train)
##X_test = model.transform(X_test)

##sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
##X_train = sel.fit_transform(X_train)
##X_test = sel.transform(X_test)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

###======================================feature selection================================================

clf = ExtraTreesClassifier(n_estimators=200, max_depth=35, min_samples_leaf=2, criterion ="gini", random_state=0)
clf = clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

test = model.transform(test)

###======================================classify================================================
classifier = RandomForestClassifier(max_depth=25, random_state=0, criterion="entropy", n_estimators=200)
classifier.fit(X_train, y_train)  
#y_pred = classifier.predict(X_test) 
y_pred = classifier.predict(test) 

file = open('submission.csv','w')
df=pd.read_csv('../input/test.csv')
IDs = df.values[:,0]

print("Id" + ',' + "Target", file=file)
for i  in range(len(y_pred)):
    print(str(IDs[i]) + ',' + str(y_pred[i]), file=file)
file.close()
###======================================report================================================
##cm = confusion_matrix(y_test, y_pred)  
##print(cm)  
#print('Accuracy = ' + str(accuracy_score(y_test, y_pred)))
#print('f1_score = ' + str(f1_score(y_test, y_pred, average='macro')))