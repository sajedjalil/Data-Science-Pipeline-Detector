#Upvote --> https://www.kaggle.com/indranilbhattacharya/bojan-chris-cv
import numpy as np
import pandas as pd 
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
oof_svnu = np.zeros(len(train)) 
pred_te_svnu = np.zeros(len(test))

oof_svc = np.zeros(len(train)) 
pred_te_svc = np.zeros(len(test))

oof_knn = np.zeros(len(train)) 
pred_te_knn = np.zeros(len(test))

oof_lr = np.zeros(len(train)) 
pred_te_lr = np.zeros(len(test))

cols = [c for c in train.columns if c not in ["id" , "target" , "wheezy-copper-turtle-magic"]] 

for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic'] == i]
    test2 = test[test['wheezy-copper-turtle-magic'] == i]
    
    id_tr = train2.index ## row indexes
    id_te = test2.index
    
    train2.reset_index(drop=True,inplace=True) # check
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols]) #### need to check whats going on here
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    ### stratified k fold ###
    skf = StratifiedKFold(n_splits=5, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        knn = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)
        knn.fit(train3[train_index,:],train2.loc[train_index]['target'])
        
        svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42, nu=0.6, coef0=0.6) 
        svnu.fit(train3[train_index,:],train2.loc[train_index]['target'])
        
        #lr = linear_model.LogisticRegression(solver='liblinear',penalty='l1',C=0.1)
        lr = neural_network.MLPClassifier(random_state=3,  activation='relu', solver='lbfgs', tol=1e-06, hidden_layer_sizes=(250, ))
        lr.fit(train3[train_index,:],train2.loc[train_index]['target'])
        
        svc = svm.SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42)
        svc.fit(train3[train_index,:],train2.loc[train_index]['target'])
    
        
        oof_svnu[id_tr[test_index]] = svnu.predict_proba(train3[test_index,:])[:,1]
        pred_te_svnu[id_te] += svnu.predict_proba(test3)[:,1] / skf.n_splits
        
        oof_svc[id_tr[test_index]] = svc.predict_proba(train3[test_index,:])[:,1]
        pred_te_svc[id_te] += svc.predict_proba(test3)[:,1] / skf.n_splits
        
        oof_knn[id_tr[test_index]] = knn.predict_proba(train3[test_index,:])[:,1]
        pred_te_knn[id_te] += knn.predict_proba(test3)[:,1] / skf.n_splits
        
        oof_lr[id_tr[test_index]] = lr.predict_proba(train3[test_index,:])[:,1]
        pred_te_lr[id_te] += lr.predict_proba(test3)[:,1] / skf.n_splits
        
print('CV score svnu=',round(roc_auc_score(train['target'],oof_svnu),5))
print('CV score svc=',round(roc_auc_score(train['target'],oof_svc),5))
print('CV score knn=',round(roc_auc_score(train['target'],oof_knn),5))
print('CV score lr=',round(roc_auc_score(train['target'],oof_lr),5))
print('CV score ensemble=',round(roc_auc_score(train['target'],oof_svnu*0.50 + oof_svc*0.15 + oof_knn*0.25 + oof_lr*0.1),5))

sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pred_te_svnu*0.50 + pred_te_svc*0.25 + pred_te_knn*0.2 + pred_te_lr*0.05
sub.to_csv('submission.csv',index=False)