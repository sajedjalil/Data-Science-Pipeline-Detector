import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns

for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

len_train = len(X_train)
len_test  = len(X_test)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 

clf = MLPClassifier(#hidden_layer_sizes=(,128,64,32), 
					activation='relu', 
     				beta_1=0.6, 
     				beta_2=0.9,
                    alpha = 0.001,
                    early_stopping = True,
                    shuffle = True,
                    warm_start = True,
                    validation_fraction = 0.3,
     				learning_rate_init=0.01, 
     				max_iter = 14000, 
     				random_state = 1235, 
     				learning_rate='adaptive')


clf.fit(X_train, y_train)
y_pred= clf.predict_proba(X_test)[:,1]
print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

print('Completed!')