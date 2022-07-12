import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
oof = np.zeros(len(train))
preds = np.zeros(len(test))

for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index
    idx2 = test2.index
    train2.reset_index(drop=True, inplace=True)
    
    # feature selection (use approx 40 of 255 features)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
       
auc = roc_auc_score(train['target'],oof)
print('QDA scores CV =',round(auc,5))

test['target'] = preds
oof = np.zeros(len(train))
preds = np.zeros(len(test))

for k in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==k] 
    train2p = train2.copy()
    idx1 = train2.index 
    test2 = test[test['wheezy-copper-turtle-magic']==k]
    
    # add pseudo label
    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()
    test2p.loc[test2p['target']>=0.5, 'target'] = 1
    test2p.loc[test2p['target']<0.5, 'target'] = 0 
    train2p = pd.concat([train2p,test2p],axis=0)
    train2p.reset_index(drop=True,inplace=True)

    poly = PolynomialFeatures(degree=2)
    sc = StandardScaler()
    train3p = poly.fit_transform(sc.fit_transform(VarianceThreshold(threshold=1.5).fit_transform(train2p[cols])))
    train3 = poly.fit_transform(sc.fit_transform(VarianceThreshold(threshold=1.5).fit_transform(train2[cols])))
    test3 = poly.fit_transform(sc.fit_transform(VarianceThreshold(threshold=1.5).fit_transform(test2[cols])))
        
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3p, train2p['target']):
        test_index3 = test_index[test_index<len(train3)]
        
        clf = LogisticRegression(solver='saga', penalty='l2', C=0.01, tol=0.001)
        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])
        oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3,:])[:,1]
        preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits
    
auc = roc_auc_score(train['target'], oof)
print('Pseudo Labeled LR scores CV =',round(auc, 5))

sub['target'] = preds
sub.to_csv('submission.csv', index=False)
