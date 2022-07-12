import pandas as pd
from sklearn import svm, neighbors
train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')
col = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42, nu=0.8, coef0=0.09)
for m in sorted(train['wheezy-copper-turtle-magic'].unique()):
    idx_tr, idx_te  = (train['wheezy-copper-turtle-magic']==m), (test['wheezy-copper-turtle-magic']==m)
    knn.fit(train[idx_tr][col], train[idx_tr]['target'])
    svnu.fit(train[idx_tr][col], train[idx_tr]['target'])
    test.loc[idx_te,'target'] = 0.9*svnu.predict_proba(test[idx_te][col])[:,1] + 0.1*knn.predict_proba(test[idx_te][col])[:,1]
test[['id','target']].to_csv("submission.csv", index=False)
