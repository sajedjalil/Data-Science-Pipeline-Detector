import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import roc_auc_score

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

cols = [c for c in data_train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

pred_train = pd.DataFrame()
pred_test = pd.DataFrame()
for i in range(512):
    
    X = data_train[data_train['wheezy-copper-turtle-magic']==i]
    y = X['target']
    train_size = X.shape[0]
    X = X.append(data_test[data_test['wheezy-copper-turtle-magic']==i], sort=False)
    ids = X['id']
    X = VarianceThreshold(threshold=2).fit_transform(X[cols])
    test_size = X.shape[0]-train_size
    
    target_weight = 10
    Xy = np.hstack([X, np.reshape(np.append(y, np.repeat(0.5, test_size)), (X.shape[0], 1))*target_weight])
    
    weights_init = np.ones(6)/6
    mean_0 = np.append(np.mean(X[:train_size,:][y==0,:], axis=0), 0)
    mean_1 = np.append(np.mean(X[:train_size,:][y==1,:], axis=0), target_weight)
    means_init = np.vstack([np.tile(mean_0, (3,1)), np.tile(mean_1, (3,1))])
    precisions_init = np.tile(np.diag(np.repeat(1/14, X.shape[1]+1)), (6,1,1))
    
    gmms = []
    scores = []
    for j in range(5):
        np.random.seed(j)
        noise = np.random.randn(6, X.shape[1]+1)/100
        gmms = gmms + [GMM(n_components=6, covariance_type='full', tol=1e-10, max_iter=1000,
                           weights_init=weights_init, means_init=means_init+noise, precisions_init=precisions_init)]
        gmms[j].fit(Xy)
        scores = scores + [gmms[j].lower_bound_]
    
    gmm_best = gmms[scores.index(max(scores))]
    
    pred = gmm_best.predict(Xy)
    clusters = pd.crosstab(pred[:train_size], y)
    positive_clusters = clusters.index[clusters[1]>clusters[0]].values
    pred = [1 if p in positive_clusters else 0 for p in pred]
    
    pred_train = pred_train.append(pd.DataFrame({'id': ids[:train_size], 'magic': i, 'pred': pred[:train_size], 'target': y}))
    pred_test = pred_test.append(pd.DataFrame({'id': ids[train_size:], 'magic': i, 'pred': pred[train_size:]}))

print("AUC: " + str(roc_auc_score(pred_train['target'], pred_train['pred'])))

# guarantee of perfect prediction
if (pred_train.append(pred_test, sort=False).groupby('magic').sum()['pred'] == 512).all():
    pred_test['target'] = pred_test['pred'] + np.random.randn(pred_test.shape[0])/100
else:
    pred_test['target'] = 0.5

pred_test[['id','target']].to_csv('submission.csv', index=False)
