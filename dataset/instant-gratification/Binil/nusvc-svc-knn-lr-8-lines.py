import pandas as pd,sklearn.svm as svm,sklearn.neighbors as neighbors,sklearn.linear_model as skl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv');col = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
knn = neighbors.KNeighborsClassifier(n_neighbors=10); svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42, nu=0.6, coef0=0.08); svc = svm.SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42);lr = skl.LogisticRegression(solver='liblinear',penalty='l1',C=0.05,n_jobs=-1)
for m in sorted(train['wheezy-copper-turtle-magic'].unique()):
    idx_tr, idx_te  = (train['wheezy-copper-turtle-magic']==m), (test['wheezy-copper-turtle-magic']==m)
    
    data = pd.concat([pd.DataFrame(train[idx_tr][col]), pd.DataFrame(test[idx_tr][col])])
    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[col]))
    train3 = data2[:train[idx_tr][col].shape[0]]; test3 = data2[train.shape[0]:]
    
    svc.fit(train3, train[idx_tr]['target']); knn.fit(train3, train[idx_tr]['target']); svnu.fit(train3, train[idx_tr]['target']);lr.fit(train3, train[idx_tr]['target'])
    test.loc[idx_te,'target'] = 0.75*svnu.predict_proba(test3[col])[:,1]#+ 0.08*svc.predict_proba(test3[col])[:,1] + 0.12*knn.predict_proba(test3[col])[:,1] + 0.05*lr.predict_proba(test3[col])[:,1]
test[['id','target']].to_csv("submission.csv", index=False)