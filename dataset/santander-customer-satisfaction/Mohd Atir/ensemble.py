
import time; start_time = time.time()
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.metrics import log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import random; random.seed(2016)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)
        
train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
features = train.columns[1:-1]
pca = PCA(n_components=2)
x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
x_test_projected = pca.transform(normalize(test[features], axis=0))
train.insert(1, 'PCAOne', x_train_projected[:, 0])
train.insert(1, 'PCATwo', x_train_projected[:, 1])
test.insert(1, 'PCAOne', x_test_projected[:, 0])
test.insert(1, 'PCATwo', x_test_projected[:, 1])

num_train = train.shape[0]

y_train = train['TARGET'].values
train = train.drop(["TARGET"],axis=1)
id_test = test['ID']


def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_
LL  = make_scorer(flog_loss, greater_is_better=False)

g={'ne':150,'md':6,'mf':80,'rs':2016} #change to g={'ne':500,'md':40,'mf':60,'rs':2016}
etc = ensemble.ExtraTreesClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], criterion='entropy', min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)      
etr = ensemble.ExtraTreesRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)      
rfc = ensemble.RandomForestClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], criterion='entropy', min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)
rfr = ensemble.RandomForestRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)
xgr = xgb.XGBRegressor(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='reg:linear')
xgc = xgb.XGBClassifier(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='binary:logistic') #try 'binary:logitraw'
clf = {'etc':etc, 'etr':etr, 'rfc':rfc, 'rfr':rfr, 'xgr':xgr, 'xgc':xgc} # use this line instead
#clf = {'etr':etr, 'rfr':rfr, 'xgr':xgr} # removed due to kaggle performance, would prefer less time and more cores than more time and less cores :)

y_pred=[]
best_score = 0.0
id_results = id_test[:]
for c in clf:
    if c[:1] != "x": #not xgb
        model = GridSearchCV(estimator=clf[c], param_grid={}, n_jobs =-1, cv=2, verbose=0, scoring=LL)
        model.fit(train, y_train)
        if c[-1:] != "c": #not classifier
            y_pred = model.predict(test)
            print("Ensemble Model: ", c, " Best CV score: ", model.best_score_, " Time: ", round(((time.time() - start_time)/60),2))
        else: #classifier
            best_score = (log_loss(y_train, model.predict_proba(train)))*-1
            y_pred = model.predict_proba(test)[:,1]
            print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))
    else: #xgb
        X_fit, X_eval, y_fit, y_eval= train_test_split(train, y_train, test_size=0.35, train_size=0.65, random_state=g['rs'])
        model = clf[c]
        model.fit(X_fit, y_fit, early_stopping_rounds=20, eval_metric="logloss", eval_set=[(X_eval, y_eval)], verbose=0)
        if c == "xgr": #xgb regressor
            best_score = (log_loss(y_train, model.predict(train)))*-1
            y_pred = model.predict(test)
        else: #xgb classifier
            best_score = (log_loss(y_train, model.predict_proba(train)))*-1
            y_pred = model.predict_proba(test)[:,1]
        print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))

    for i in range(len(y_pred)):
        if y_pred[i]<0.0:
            y_pred[i] = 0.0
        if y_pred[i]>1.0:
            y_pred[i] = 1.0
    df_in = pd.DataFrame({"ID": id_test, c: y_pred})
    id_results = pd.concat([id_results, df_in[c]], axis = 1)
    

id_results['avg'] = id_results.drop('ID' , axis = 1).apply(np.average, axis = 1)
id_results['min'] = id_results.drop('ID', axis = 1).apply(min, axis = 1)
id_results['max'] = id_results.drop('ID', axis = 1).apply(max, axis = 1)
id_results['diff'] = id_results['max'] - id_results['min']
for i in range(10):
    print(i, len(id_results[id_results['diff']>(i/10)]))
id_results.to_csv("results_analysis.csv", index=False)
ds = id_results[['ID','max']]
ds.columns = ['ID','TARGET']
ds.to_csv('submission.csv',index=False)
