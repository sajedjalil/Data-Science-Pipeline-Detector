import xgboost as xgb
from sklearn import *
import sklearn
import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
y = train["y"]; y_mean = np.mean(y)
train = train.drop('y', axis=1)
test = pd.read_csv('../input/test.csv')
pid = test['ID'].values

#dcol =[]
#c = train.columns
#for i in range(len(c)-1):
#    v = train[c[i]].values +  test[c[i]].values
#    for j in range(i+1,len(c)):
#        if np.array_equal(v,train[c[j]].values + test[c[j]].values):
#            dcol.append(c[i])
#            print("dropping column: ", c[i], " duplicate of: ", c[j])
#            break
#train = train.drop(dcol,axis=1)
#test = test.drop(dcol,axis=1)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder(); print(c)
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        train[c+'len'] = train[c].map(lambda x: len(str(x)))
        test[c] = lbl.transform(list(test[c].values))
        test[c+'len'] = test[c].map(lambda x: len(str(x)))

def fall(X): return X
def fsq(X): return X ** 2
def fps(X): return preprocessing.maxabs_scale(X)

fp = pipeline.Pipeline([('pi', preprocessing.Imputer(strategy='mean')),
        ('union', pipeline.FeatureUnion(n_jobs = -1, 
                    transformer_weights={'standard': 1.0, 'ps': 1.0, 'tsvd': .9, 'kmeans': .7, 'pca': .8, 'fica': .8},
                    transformer_list = [
                        ('standard', preprocessing.FunctionTransformer(fall)),
                        #('fs', feature_selection.SelectKBest(k=80)),
                        #('sq', preprocessing.FunctionTransformer(fsq)),
                        ('ps', preprocessing.FunctionTransformer(fps)),
                        ('tsvd', decomposition.TruncatedSVD(n_components=30, n_iter=25, random_state=300)),
                        ('kmeans', cluster.KMeans(n_clusters=5, random_state=320)),
                        ('pca',  decomposition.PCA(n_components=10, random_state=500)),
                        ('fica', decomposition.FastICA(n_components=10, random_state=600))],
                        ))])
train = fp.fit_transform(train, y); print(train.shape)
test = fp.transform(test); print(test.shape)

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', metrics.r2_score(labels, preds)

params = {
    'eta': 0.03,
    'max_depth': 4,
    'objective': 'reg:linear',
    'base_score': y_mean,
    'seed': 320,
    'silent': True
}

x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=320)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, feval=xgb_r2_score, maximize=True, verbose_eval=50, early_stopping_rounds=50)
pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
submission = pd.DataFrame({'id': pid, 'y': pred})
submission.to_csv('submission.csv', index=False)

# This is just a test for fun like everything else right...
watchlist = [(xgb.DMatrix(test, pred), 'train'), (xgb.DMatrix(train, y), 'valid')]
model = xgb.train(params, xgb.DMatrix(test, pred), 1000,  watchlist, feval=xgb_r2_score, maximize=True, verbose_eval=50, early_stopping_rounds=50)
pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit) #overfit validation on train
submission = pd.DataFrame({'id': pid, 'y': pred})
submission.to_csv('submission_ov.csv', index=False)

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15.0, 15.0)
xgb.plot_importance(model)
plt.savefig('feature_importance.png') #will be complex w/ all transformations and data obfuscation


