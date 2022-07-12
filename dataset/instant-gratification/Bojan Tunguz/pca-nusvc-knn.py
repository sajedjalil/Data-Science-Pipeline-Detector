#based on the following kernel: https://www.kaggle.com/hyeonho/pca-nusvc-0-95985

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

oof = np.zeros(len(train))
preds = np.zeros(len(test))
oof_2 = np.zeros(len(train))
preds_2 = np.zeros(len(test))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))
    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]
    
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=5, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):

        clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_2[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds_2[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
    #if i%15==0: print(i)
        
print(roc_auc_score(train['target'], oof))
print(roc_auc_score(train['target'], oof_2))
print(roc_auc_score(train['target'], 0.8*oof+0.2*oof_2))
print(roc_auc_score(train['target'], 0.95*oof+0.05*oof_2))
print(roc_auc_score(train['target'], 1.05*oof-0.05*oof_2))

sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv', index=False)

sub['target'] = 0.8*preds+0.2*preds_2
sub.to_csv('submission_2.csv', index=False)

sub['target'] = 0.95*preds+0.05*preds_2
sub.to_csv('submission_3.csv', index=False)

sub['target'] = 1.05*preds-0.05*preds_2
sub.to_csv('submission_3.csv', index=False)