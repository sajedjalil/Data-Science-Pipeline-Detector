import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# clean and split data

# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

features = train.columns[1:-1]
pca = PCA(n_components=2)
x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
x_test_projected = pca.transform(normalize(test[features], axis=0))
train.insert(1, 'PCAOne', x_train_projected[:, 0])
train.insert(1, 'PCATwo', x_train_projected[:, 1])
test.insert(1, 'PCAOne', x_test_projected[:, 0])
test.insert(1, 'PCATwo', x_test_projected[:, 1])

# split data into train and test
test_id = test.ID
test = test.drop(["ID"],axis=1)

train = train.replace(-999999,2)
test = test.replace(-999999,2)

#X = train.iloc[:,:-1]
#y = train.TARGET

#X['n0'] = (X==0).sum(axis=1)
#train['n0'] = X['n0']

#X = test.iloc[:,:-1]

#X['n0'] = (X==0).sum(axis=1)
#test['n0'] = X['n0']


#X = train.iloc[:,:-1]
#y = train.TARGET

#X['b0'] = (X<0).sum(axis=1)
#train['b0'] = X['b0']

#X = test.iloc[:,:-1]

#X['b0'] = (X<0).sum(axis=1)
#test['b0'] = X['b0']

#X = train.iloc[:,:-1]
#y = train.TARGET

#X['g0'] = (X>0).sum(axis=1)
#train['g0'] = X['g0']

#X = test.iloc[:,:-1]

#X['g0'] = (X>0).sum(axis=1)
#test['g0'] = X['g0']

#train['var38mc'] = np.isclose(train.var38, 117310.979016)
#train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
#train.loc[train['var38mc'], 'logvar38'] = 0

#print(test.var38.value_counts())
#test['var38mc'] = np.isclose(test.var38, 117310.979016)
#test['logvar38'] = test.loc[~test['var38mc'], 'var38'].map(np.log)
#test.loc[test['var38mc'], 'logvar38'] = 0



from sklearn import ensemble, metrics, linear_model
import random

#Some parameters to play with
rnd=12
random.seed(rnd)
n_ft=20 #Number of features to add
max_elts=7 #Maximum size of a group of linear features

class addNearestNeighbourLinearFeatures:
    
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd=random_state
        self.n=n_neighbours
        self.max_elts=max_elts
        self.verbose=verbose
        self.neighbours=[]
        self.clfs=[]
        
    def fit(self,train,y):
        if self.rnd!=None:
            random.seed(rnd)
        if self.max_elts==None:
            self.max_elts=len(train.columns)
        list_vars=list(train.columns)
        random.shuffle(list_vars)
        
        lastscores=np.zeros(self.n)+1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars=list_vars[self.n:]
        
        for elt in list_vars:
            indice=0
            scores=[]
            for elt2 in self.neighbours:
                if len(elt2)<self.max_elts:
                    clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1) 
                    clf.fit(train[elt2+[elt]], y)
                    scores.append(metrics.log_loss(y,clf.predict(train[elt2 + [elt]])))
                    indice=indice+1
                else:
                    scores.append(lastscores[indice])
                    indice=indice+1
            gains=lastscores-scores
            if gains.max()>0:
                temp=gains.argmax()
                lastscores[temp]=scores[temp]
                self.neighbours[temp].append(elt)

        indice=0
        for elt in self.neighbours:
            clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1) 
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice=indice+1
                    
    def transform(self, train):
        indice=0
        for elt in self.neighbours:
            train['_'.join(pd.Series(elt).sort_values().values)]=self.clfs[indice].predict(train[elt])
            indice=indice+1
        return train
    
    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)

target = train['TARGET'].values
train1 = train.drop(["TARGET","ID"],axis=1)

a=addNearestNeighbourLinearFeatures(n_neighbours=n_ft, max_elts=max_elts, verbose=True, random_state=rnd)
a.fit(train1, target)

train1 = a.transform(train1)
test = a.transform(test)


train2 = train1[1:55000]
train21 = train[1:55000]

X = train2
y = train21.TARGET.values

train3 = train1[55001:75000]
train31 = train[55001:75000]

X1 = train3
y1 = train31.TARGET.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, test.shape)

## # Feature selection
#clf = ExtraTreesClassifier(random_state=1729)
#selector = clf.fit(X_train, y_train)
#clf.feature_importances_ 
#fs = SelectFromModel(selector, prefit=True)

#X_train = fs.transform(X_train)
#X_test = fs.transform(X_test)
#test = fs.transform(test)
#X1 = fs.transform(X1)

print(X_train.shape, X_test.shape, test.shape)

## # Train Model
# classifier from xgboost
m2_xgb = xgb.XGBClassifier(n_estimators=700, nthread=-1, seed=1729)
m2_xgb.fit(X_train, y_train, eval_metric="auc", verbose = False,
           eval_set=[(X_test, y_test)])

# calculate the auc score
print("Roc AUC: ", roc_auc_score(y_test, m2_xgb.predict_proba(X_test)[:,1],
              average='macro'))

print("Roc AUC: ", roc_auc_score(y1, m2_xgb.predict_proba(X1)[:,1],
              average='macro'))

              
## # Submission
probs = m2_xgb.predict_proba(test)

submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)



