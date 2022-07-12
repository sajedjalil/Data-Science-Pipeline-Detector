import warnings
import numpy as np
import pandas as pd

# read data
basepath = '../input/'
Train = pd.read_csv(basepath+'train.csv', na_values="-1")
Test = pd.read_csv(basepath+'test.csv', na_values="-1")

save_RAM = True
if save_RAM:
    for c in Train.select_dtypes(include=['float64']).columns:
        Train[c]=Train[c].astype(np.float32)
        Test[c]=Test[c].astype(np.float32)
    for c in Train.select_dtypes(include=['int64']).columns[2:]:
        Train[c]=Train[c].astype(np.int8)
        Test[c]=Test[c].astype(np.int8)

IGNORE_WORNNING = True
if IGNORE_WORNNING:
    warnings.filterwarnings("ignore")

# compute statistical quantity
def p_gu(p1, p2, n1, n2):
    return (p1*n1+p2*n2)/(n1+n2)
def z(p1,p2,n1,n2):
    return (p1-p2)/np.sqrt(p_gu(p1, p2, n1, n2)*(1-p_gu(p1, p2, n1, n2))*(1/n1 + 1/n2))
    
def dummies_cat(train, test, target, threshold=1, show=False):
    col_cat = train.columns[train.columns.str.endswith('_cat')]
    train = pd.get_dummies(train, columns=col_cat, drop_first=False)
    test = pd.get_dummies(test, columns=col_cat, drop_first=False)

    nucorr_drop=[]
    col = [x for x in train.columns if x.count('_cat_') == 1]
    for feature in col:
        p1 = target[target == 1][train[feature] == 1].count() / target[train[feature] == 1].count()
        p2 = target[target == 1][train[feature] == 0].count() / target[train[feature] == 0].count()
        n1 = target[train[feature] == 1].count()
        n2 = target[train[feature] == 0].count()
        p = z(p1,p2,n1,n2)
        if np.abs(p) < threshold:
            nucorr_drop.append(feature)
    train = train.drop(nucorr_drop, axis=1)
    test = test.drop(nucorr_drop, axis=1)
    if show:
        print('The useless categories include:')
        print(nucorr_drop)    
    return train, test
    
 
target = Train.target
id_train = Train.id
id_test = Test.id

# data copy to avoid read data from disk repeatedly
train = Train.copy()
test = Test.copy()

# the "threshold" id the threshold of statistical quantity
train, test = dummies_cat(train, test, target, threshold=1.96, show=True)


