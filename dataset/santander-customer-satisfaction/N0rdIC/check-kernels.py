import pandas
import numpy as np
import xgboost as xgb


from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.cross_validation import train_test_split

print('hello')
df_train = pandas.read_csv('../input/train.csv')
df_test  = pandas.read_csv('../input/test.csv')   

#%% avoid useless variables

remove = []

for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)
        
df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)


id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values
X_test = df_test.drop(['ID'], axis=1).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%% do classification

#train_set=5000;
#clf = SVC(kernel='rbf', gamma=7e-9, C=3000,probability=True)
#scores = cross_validation.cross_val_score(clf, X_train[1:train_set], y_train[1:train_set], scoring='roc_auc', cv=2) 
#print(scores.mean())

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.3)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

             
#clf.fit(X_train[1:train_set], y_train[1:train_set])
y_pred = clf.predict_proba(X_test)

submission = pandas.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)