# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler #StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB

def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    #print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, test):

    features = train.columns[2:]
    print(type(features))
    for col in features:
        if((train[col].dtype == 'object') and (col!="v22")):
            #print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)
            train[col] = train[col].astype(float)
            test[col] = test[col].astype(float)
    return train, test


print('Load data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y_train = train['target'].values
id_train = train['ID'].values
id_test = test['ID']

train, test = MungeData(train, test)

target = train['target'].values
id_train = train['ID'].values
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = train[train_name].mean() 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = test[test_name].mean()

#scaler = StandardScaler()
scaler =MinMaxScaler()
scaler.fit(train)
X_train = scaler.transform(train) #HDKIM numpy array
X_test = scaler.transform(test) #HDKIM numpy array

clf = MLPClassifier(hidden_layer_sizes=(64, 32, 64, 8), activation='relu', 
     beta_1=0.95, beta_2=0.999, alpha = 0.01, early_stopping = True, validation_fraction = 0.1,
     learning_rate_init=0.0009, max_iter = 12000, random_state = 9996, #63458, #8888, #1235, 
     learning_rate='adaptive') 

#
# CV
#

nCV = 4
rng = np.random.RandomState(31337)
kf = StratifiedKFold(y_train, n_folds=nCV, shuffle=True, random_state=rng) 
cv_preds = np.array([0.0] * X_train.shape[0])
i = 0
for train_index, test_index in kf: 
   i = i+1
   print("CV iteration:",i)
   clf.fit(X_train[train_index,:],y_train[train_index]) 
   pred = clf.predict_proba(X_train[test_index,:])[:,1]
   print("k-fold score:",log_loss(y_train[test_index],pred))
   print(pred[0:5]) 
   cv_preds[test_index] = pred

print("cv score: ",log_loss(y_train,cv_preds)) 
preds_out = pd.DataFrame({"ID": id_train, "PredictedProb": cv_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('NN_CV4val.csv')

#
# Submission
#

nBagging = 1
bagging_preds = np.array([0.0] * X_test.shape[0])
for i in range(nBagging):
   clf.fit(X_train, y_train)
   y_pred= clf.predict_proba(X_test)[:,1]
   y_pred_train = clf.predict_proba(X_train)[:,1]
   print('Overall AUC:', log_loss(y_train, y_pred_train)) 
   bagging_preds = bagging_preds + y_pred

submission = pd.DataFrame({"ID":id_test, "TARGET":bagging_preds})
submission.to_csv("NN_CV4.csv", index=False)