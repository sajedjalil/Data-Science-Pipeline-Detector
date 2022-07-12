'''
In this scipt, SVM's SVR model is used, which can be tuned with different parameters

@Author : Nikesh Bajaj

'''
# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

labels = np.log1p(train.Hazard)
test_ind = test.Id
train.drop('Hazard', axis=1, inplace=True)
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

columns = train.columns

train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

print(train.shape)
print(test.shape)

mn = np.mean(train,axis=0)
st = np.std(train,axis=0)
train =(train-mn)/(st)
test =(test-mn)/(st)

Xtr, Xts, ytr, yts = train_test_split(train, labels, test_size=.2)

clf = svm.SVR(C=100, kernel='rbf', degree =3, gamma=0.0,  
            max_iter=100, shrinking=True, tol=0.001, verbose=False, 
               coef0=0.0, epsilon=0.5)

# Try different parameters when you get better train modal with complete data set and
# then predict test data for submission


clf.fit(Xtr, ytr)
ytp = clf.predict(Xtr)
ysp = clf.predict(Xts)
print('Tr Score: ', Gini(ytr, ytp))
print('Va Score: ', Gini(yts, ysp))

clf.fit(train,labels)

yp = clf.predict(test)

preds = pd.DataFrame({"Id": test_ind, "Hazard": yp})
preds = preds.set_index('Id')
preds.to_csv('Benchmark_SVM.csv')



