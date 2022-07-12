import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

seed = 42

#train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
train = pd.read_csv('../input/categoricalfeatureencodingiifolddata/train_kfolds.csv')



test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

y = train.target.values
train = train.drop(['id', 'target', 'kfolds'], axis=1)
test = test.drop('id', axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
        
"""
model = xgb.XGBClassifier(n_estimators=700,
                        nthread=-1,
                        max_depth=9,
                        learning_rate=0.026,
                        silent=False,
                        subsample=0.8,
                        colsample_bytree=0.75)

"""

model = xgb.XGBClassifier(n_estimators=650)
#n_estimators = range(50,700,50)
max_depth = range(1,12,1)
#learning_rate = np.arange(0.001,0.31 ,0.001)#)[0.0001, 0.001, 0.01, 0.1]
param_grid = dict(max_depth=max_depth)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
gridsearch = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
gridresult = gridsearch.fit(train, y)
print("Best: %f using %s" % (gridresult.best_score_, gridresult.best_params_))
            

"""


# train test

#X_train, X_test, y_train, y_test = train_test_split(train, y, sitest_ze=0.30, random_state=seed)
#eval_set = [(X_train, y_train), (X_test, y_test)]
#xgb_model = model.fit(X_train, y_train, eval_metric=["auc"],eval_set=eval_set,early_stopping_rounds=30, verbose=True)

kfold = StratifiedKFold(n_splits=10,shuffle=True, random_state=seed)
results = cross_val_score(model, train, y, cv=kfold, verbose=True)

#xgb_model = model.fit(train, y, eval_metric="auc", early_stopping_rounds=30)
print("Accuracy: %.2f%%" % (results.mean()*100))
#preds = xgb_model.predict_proba(test)[:,1]
preds = model.predict_proba(test)[:,1]
sample = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

sample.target = preds
sample.to_csv('submission.csv', index=False)

"""
print("Done")