
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_train = pd.read_csv('../input/train.csv')
id_train = data_train.ID


y_train = data_train['target'].values
data_train = data_train.drop(['ID', 'target', 'v22'], axis=1)

data_train.shape
for i in data_train.columns:
    if data_train[i].dtype == float:
        data_train[i] = data_train[i].fillna(data_train[i].mean())
categ_col = [c for c in data_train.columns if data_train[c].dtype.name == 'object']
num_col = [c for c in data_train.columns if data_train[c].dtype.name != 'object']

import scipy
from sklearn import preprocessing
data_des = data_train.describe(include=[object])
for i in categ_col:
    data_train[i] = data_train[i].fillna(data_des[i]['top'])
for i in data_train.columns:
    if data_train[i].dtype == float:
        data_train[i] = data_train[i].fillna(data_train[i].mean())
data_cat = pd.get_dummies(data_train[categ_col], columns=categ_col)
data_num = data_train[num_col]
data_num = pd.DataFrame(preprocessing.scale(data_num),)
data = np.hstack((data_num, data_cat))
data.shape

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
random_state=1
from sklearn.cross_validation import train_test_split
X, X_test, y, y_test = train_test_split(data, y_train, test_size = 0.3, random_state = random_state)
#Spliting train data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.35, 
                                                      random_state=random_state)

print('Data shape:')
print('X_train: %s, X_valid: %s, X_test: %s \n' %(X_train.shape, X_valid.shape, 
                                                  X_test.shape))
    
from sklearn import cross_validation
def avg(x):
    s = 0.0
    for t in x:
        s += t
    return (s/len(x))*100.0
glm = LogisticRegression(penalty='l1', tol=1)
scores = cross_validation.cross_val_score(glm, X, y, cv = 10)
print("Logistic Regression with L1 metric - " + ' avg = ' + ('%2.1f'%avg(scores)))
clfs = {'LR'  : LogisticRegression(penalty='l1', random_state=random_state), 
        'SVM' : SVC(kernel='linear', C=1, probability=True, random_state=random_state), 
        'RF'  : RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                       random_state=random_state), 
        'GBM' : GradientBoostingClassifier(n_estimators=50, 
                                           random_state=random_state), 
        'ETC' : ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
                                     random_state=random_state),
        'KNN' : KNeighborsClassifier(n_neighbors=10)}
    
#predictions on the validation and test sets
p_valid = []
p_test = []
print('Performance of individual classifiers (1st layer) on X_test')   
print('------------------------------------------------------------')
   
for nm, clf in clfs.items():
    #First run. Training on (X_train, y_train) and predicting on X_valid.
    clf.fit(X_train, y_train)
    yv = clf.predict_proba(X_valid)
    p_valid.append(yv)
        
    #Second run. Training on (X, y) and predicting on X_test.
    clf.fit(X, y)
    yt = clf.predict_proba(X_test)
    p_test.append(yt)