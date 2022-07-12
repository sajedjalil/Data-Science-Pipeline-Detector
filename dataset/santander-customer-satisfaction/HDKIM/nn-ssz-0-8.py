
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler #StandardScaler
from sklearn.cross_validation import StratifiedKFold


# load data
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
id_train = df_train['ID'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values


id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

scaler = StandardScaler()
#scaler =MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

#clf = MLPClassifier(hidden_layer_sizes=(64, 32, 64, 8), activation='relu', 
#     beta_1=0.96, beta_2=0.999, alpha = 0.01, early_stopping = True, validation_fraction = 0.1,
#     learning_rate_init=0.001, max_iter = 12000, random_state = 123289, #63458, #8888, #1235, 
#     learning_rate='adaptive')
clf = MLPClassifier(hidden_layer_sizes=(96, 36, 72, 6), activation='relu', 
     beta_1=0.85, beta_2=0.999, alpha = 0.008, 
     learning_rate_init=0.0012, max_iter =12000, random_state = 8123, 
     learning_rate='adaptive')
#
# CV
#
'''
nCV = 4
rng = np.random.RandomState(31337)
kf = StratifiedKFold(y_train, n_folds=nCV, shuffle=True, random_state=rng) 
cv_preds = np.array([0.0] * X_train.shape[0])
i = 0

for train_index, test_index in kf: 
   i = i+1
   print("CV iteration:",i)
   clf.fit(X_train[train_index,:],y_train[train_index]) 
   pred = clf.predict_proba(X_train[test_index,:])[:,1]
   print(pred[0:5]) 
   print("k-fold score:",roc_auc_score(y_train[test_index],pred))
   cv_preds[test_index] = pred

print("cv score: ",roc_auc_score(y_train,cv_preds))
preds_out = pd.DataFrame({"ID": id_train, "TARGET": cv_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('SSZ_NN_v16_CV4val.csv')
'''
#
# Submission
#

nBagging = 1
bagging_preds = np.array([0.0] * X_test.shape[0])
for i in range(nBagging):
   print(i)
   clf.fit(X_train, y_train)
   y_pred= clf.predict_proba(X_test)[:,1]
   y_pred_train = clf.predict_proba(X_train)[:,1]
   print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
   bagging_preds = bagging_preds + y_pred

submission = pd.DataFrame({"ID":id_test, "TARGET":bagging_preds})
submission.to_csv("SSZ_NN_v16.csv", index=False)

print('Completed!')