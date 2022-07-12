#scikit learn ensembe workflow for binary probability
import time; start_time = time.time()
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.metrics import log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import random; random.seed(2016)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
num_train = train.shape[0]

y_train = train['TARGET']
train = train.drop(['TARGET'],axis=1)
id_test = test['ID']

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['null_count'] = df_all.isnull().sum(axis=1).tolist()
df_all = df_all.fillna(-1)
df_all_temp = df_all['ID']
df_all = df_all.drop(['ID'],axis=1)
df_data_types = df_all.dtypes[:] #{'object':0,'int64':0,'float64':0,'datetime64':0}
d_col_drops = []

for i in range(len(df_data_types)):
    if str(df_data_types[i])=='object':
        df_u = pd.unique(df_all[str(df_data_types.index[i])].ravel())
        print("Column: ", str(df_data_types.index[i]), " Length: ", len(df_u))
        d={}
        j = 1000
        for s in df_u:
            d[str(s)]=j
            j+=5
        df_all[str(df_data_types.index[i])+'_vect_'] = df_all[str(df_data_types.index[i])].map(lambda x:d[str(x)])
        d_col_drops.append(str(df_data_types.index[i]))
        if len(df_u)<150:
            dummies = pd.get_dummies(df_all[str(df_data_types.index[i])]).rename(columns=lambda x: str(df_data_types.index[i]) + '_' + str(x))
            df_all_temp = pd.concat([df_all_temp, dummies], axis=1)

#f_all_temp = df_all_temp.drop(['ID'],axis=1)
df_all = pd.concat([df_all, df_all_temp], axis=1)
print(len(df_all), len(df_all.columns))
train = df_all.iloc[:num_train]
test = df_all.iloc[num_train:]
train = train.drop(d_col_drops,axis=1)
test = test.drop(d_col_drops,axis=1)

def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_
LL  = make_scorer(flog_loss, greater_is_better=False)

g={'ne':30,'md':10,'mf':300,'rs':2016}
etc = ensemble.ExtraTreesClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], criterion='entropy', min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)      
etr = ensemble.ExtraTreesRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)      
rfc = ensemble.RandomForestClassifier(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], criterion='entropy', min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)
rfr = ensemble.RandomForestRegressor(n_estimators=g['ne'], max_depth=g['md'], max_features=g['mf'], random_state=g['rs'], min_samples_split= 4, min_samples_leaf= 2, verbose = 0, n_jobs =-1)
xgr = xgb.XGBRegressor(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85)
xgc = xgb.XGBClassifier(n_estimators=g['ne'], max_depth=g['md'], seed=g['rs'], missing=np.nan, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85)
clf = {'etc':etc, 'etr':etr, 'rfc':rfc, 'rfr':rfr, 'xgr':xgr, 'xgc':xgc}

y_pred=[]
best_score = 0.0
for c in clf:
    if c[:1] != "x": #not xgb
        model = GridSearchCV(estimator=clf[c], param_grid={}, n_jobs =-1, cv=2, verbose=0, scoring=LL)
        model.fit(train, y_train.values)
        if c[-1:] != "c": #not classifier
            y_pred = model.predict(test)
            print("Ensemble Model: ", c, " Best CV score: ", model.best_score_, " Time: ", round(((time.time() - start_time)/60),2))
        else: #classifier
            best_score = (log_loss(y_train.values, model.predict_proba(train)[:,1]))*-1
            y_pred = model.predict_proba(test)[:,1]
            print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))
    else: #xgb
        X_fit, X_eval, y_fit, y_eval= train_test_split(train, y_train, test_size=0.35, train_size=0.65, random_state=g['rs'])
        model = clf[c]
        model.fit(X_fit, y_fit.values, early_stopping_rounds=20, eval_metric="logloss", eval_set=[(X_eval, y_eval)], verbose=0)
        if c == "xgr": #xgb regressor
            best_score = (log_loss(y_train.values, model.predict(train)))*-1
            y_pred = model.predict(test)
        else: #xgb classifier
            best_score = (log_loss(y_train.values, model.predict_proba(train)[:,1]))*-1
            y_pred = model.predict_proba(test)[:,1]
        print("Ensemble Model: ", c, " Best CV score: ", best_score, " Time: ", round(((time.time() - start_time)/60),2))

    for i in range(len(y_pred)):
        if y_pred[i]<0.0:
            y_pred[i] = 0.0
        if y_pred[i]>1.0:
            y_pred[i] = 1.0
    pd.DataFrame({"ID": id_test, "TARGET": y_pred}).to_csv(c + '.csv',index=False)

#Combine the results
id_results = id_test[:]
col1, col2 = ['ID','TARGET']
for c in clf:
    df_in = pd.read_csv(c + '.csv')
    df_in.columns = ['ID', c]
    id_results = pd.concat([id_results, df_in[c]], axis=1)
id_results['avg'] = id_results.drop('ID', axis=1).apply(np.average, axis=1)
id_results['min'] = id_results.drop('ID', axis=1).apply(min, axis=1)
id_results['max'] = id_results.drop('ID', axis=1).apply(max, axis=1)
id_results['diff'] = id_results['max'] - id_results['min']
for i in range(10):
    print(i, len(id_results[id_results['diff']>(i/10)]))
id_results.to_csv("results_analysis.csv", index=False)