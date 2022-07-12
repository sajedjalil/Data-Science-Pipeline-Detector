# I'm still poking around the impressive Kaggle interface. I thought I would try changing the SVM parameter gamma from 'auto' to 'scale' because I could (the cloud computation is so much better than my old laptop :) 
# 'scale' is recommended in the scikit docs, and is scheduled to become the new default
import pandas as pd; from sklearn import svm, neighbors, linear_model; train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv'); col = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
knn = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9); svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='scale', random_state=42, nu=0.6, coef0=0.08); lr = linear_model.LogisticRegression(solver='liblinear',penalty='l1',C=0.05,n_jobs=-1); svc = svm.SVC(probability=True, kernel='poly', degree=4, gamma='scale', random_state=42); lr = linear_model.LogisticRegression(solver='liblinear',penalty='l1',C=0.05,n_jobs=-1)
for m in sorted(train['wheezy-copper-turtle-magic'].unique()):
    idx_tr, idx_te  = (train['wheezy-copper-turtle-magic']==m), (test['wheezy-copper-turtle-magic']==m)
    svc.fit(train[idx_tr][col], train[idx_tr]['target']); knn.fit(train[idx_tr][col], train[idx_tr]['target']); svnu.fit(train[idx_tr][col], train[idx_tr]['target']); lr.fit(train[idx_tr][col], train[idx_tr]['target'])
    test.loc[idx_te,'target'] = 0.75*svnu.predict_proba(test[idx_te][col])[:,1]+ 0.06*svc.predict_proba(test[idx_te][col])[:,1] + 0.14*knn.predict_proba(test[idx_te][col])[:,1]+ 0.05*lr.predict_proba(test[idx_te][col])[:,1]
test[['id','target']].to_csv("submission.csv", index=False)


### Scikit docs (https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC)[https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC]
# gamma : float, optional (default=’auto’)
#     Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#     Current default is ‘auto’ which uses 1 / n_features, if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma. 
#     The current default of gamma, ‘auto’, will change to ‘scale’ in version 0.22. 
#     ‘auto_deprecated’, a deprecated version of ‘auto’ is used as a default indicating that no explicit value of gamma was passed.
