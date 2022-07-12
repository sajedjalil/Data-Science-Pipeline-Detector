#based on the following kernel: https://www.kaggle.com/prashantkikani/ig-pca-nusvc-knn-lr-stack
#which was based on: https://www.kaggle.com/hyeonho/pca-nusvc-0-95985
#and on: https://www.kaggle.com/graf10a/single-qda-lb-0-96610-time-1-min

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold

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

oof_mlp = np.zeros(len(train)) 
pred_te_mlp = np.zeros(len(test))

oof_qda = np.zeros(len(train)) 
pred_te_qda = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))
    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]
    
    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=1.5).fit_transform(data[cols]))
    train4 = data2[:train2.shape[0]]; test4 = data2[train2.shape[0]:]
    
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=5, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):

        clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_svnu[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_te_svnu[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_knn[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_te_knn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = linear_model.LogisticRegression(solver='saga',penalty='l1',C=0.1)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_lr[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_te_lr[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = neural_network.MLPClassifier(random_state=3,  activation='relu', solver='lbfgs', tol=1e-06, hidden_layer_sizes=(250, ))
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_mlp[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_te_mlp[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = svm.SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_svc[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_te_svc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = QuadraticDiscriminantAnalysis(reg_param=0.111)
        clf.fit(train4[train_index,:],train2.loc[train_index]['target'])
        oof_qda[idx1[test_index]] = clf.predict_proba(train4[test_index,:])[:,1]
        pred_te_qda[idx2] += clf.predict_proba(test4)[:,1] / skf.n_splits
        
        
print('lr', roc_auc_score(train['target'], oof_lr))
print('knn', roc_auc_score(train['target'], oof_knn))
print('svc', roc_auc_score(train['target'], oof_svc))
print('svcnu', roc_auc_score(train['target'], oof_svnu))
print('mlp', roc_auc_score(train['target'], oof_mlp))
print('qda', roc_auc_score(train['target'], oof_qda))
print('blend 1', roc_auc_score(train['target'], oof_svnu*0.7 + oof_svc*0.05 + oof_knn*0.2 + oof_mlp*0.05))
print('blend 2', roc_auc_score(train['target'], oof_qda*0.5+oof_svnu*0.35 + oof_svc*0.025 + oof_knn*0.1 + oof_mlp*0.025))

oof_svnu = oof_svnu.reshape(-1, 1)
pred_te_svnu = pred_te_svnu.reshape(-1, 1)
oof_svc = oof_svc.reshape(-1, 1)
pred_te_svc = pred_te_svc.reshape(-1, 1)
oof_knn = oof_knn.reshape(-1, 1)
pred_te_knn = pred_te_knn.reshape(-1, 1)
oof_mlp = oof_mlp.reshape(-1, 1)
pred_te_mlp = pred_te_mlp.reshape(-1, 1)
oof_lr = oof_lr.reshape(-1, 1)
pred_te_lr = pred_te_lr.reshape(-1, 1)
oof_qda = oof_qda.reshape(-1, 1)
pred_te_qda = pred_te_qda.reshape(-1, 1)

tr = np.concatenate((oof_svnu, oof_svc, oof_knn, oof_mlp, oof_lr, oof_qda), axis=1)
te = np.concatenate((pred_te_svnu, pred_te_svc, pred_te_knn, pred_te_mlp, pred_te_lr, pred_te_qda), axis=1)
print(tr.shape, te.shape)

oof_lrr = np.zeros(len(train)) 
pred_te_lrr = np.zeros(len(test))
skf = StratifiedKFold(n_splits=5, random_state=42)

for train_index, test_index in skf.split(tr, train['target']):
    # lrr = svm.NuSVC(probability=True, kernel='poly', degree=2, gamma='auto', random_state=42, nu=0.6, coef0=0.6) 
    lrr = linear_model.LogisticRegression() # solver='liblinear',penalty='l1',C=0.1
    lrr.fit(tr[train_index], train['target'][train_index])
    oof_lrr[test_index] = lrr.predict_proba(tr[test_index,:])[:,1]
    pred_te_lrr += lrr.predict_proba(te)[:,1] / skf.n_splits
    
print('stack CV score =',round(roc_auc_score(train['target'],oof_lrr),6))

print('stack + blend CV score =',roc_auc_score(train['target'],0.4*(oof_qda.flatten()*0.5+oof_svnu.flatten()*0.35 + 
                                                                    oof_svc.flatten()*0.025 + oof_knn.flatten()*0.1 + oof_mlp.flatten()*0.025)+0.6*oof_lrr))

sub = pd.read_csv('../input/sample_submission.csv')

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred_te_svnu*0.7 + pred_te_svc*0.05 + pred_te_knn*0.2 + pred_te_lr*0.05
sub.to_csv('submission_blend_1.csv', index=False)

sub['target'] = pred_te_qda*0.5+pred_te_svnu*0.35 + pred_te_svc*0.025 + pred_te_knn*0.1 + pred_te_lr*0.025
sub.to_csv('submission_blend_2.csv', index=False)

sub['target'] = pred_te_lrr
sub.to_csv('submission_stack.csv', index=False)

sub['target'] = 0.6*pred_te_lrr +0.4*(pred_te_qda.flatten()*0.5+pred_te_svnu.flatten()*0.35 + pred_te_svc.flatten()*0.025 + pred_te_knn.flatten()*0.1 + pred_te_lr.flatten()*0.025)
sub.to_csv('submission_blend_3.csv', index=False)