import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(8)


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

print('Original feature size:')

print(X_train.shape, X_test.shape)

pca = PCA(n_components=0.25)
#X_train = pca.fit_transform(X_train)
#X_test = pca.fit_transform(X_test)
#print(X_train.shape, X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 




clf = MLPClassifier(hidden_layer_sizes=(64, 32, 64, 8), activation='logistic', 
     beta_1=0.9, beta_2=0.999, alpha = 0.01, early_stopping = True, validation_fraction = 0.25,
     learning_rate_init=0.001, max_iter = 8000, random_state = 1235, 
     learning_rate='adaptive')

clf.fit(X_train, y_train)
y_pred= clf.predict_proba(X_test)[:,1]
y_pred_train = clf.predict_proba(X_train)[:,1]
print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
res_test = pd.DataFrame({"ID":id_test, "RES":y_pred})
res_train = pd.DataFrame({"ID":id_train, "RES":y_pred_train})

submission.to_csv("nn_logistic.csv", index=False)
#res_test.to_csv("y_test.csv", index=False)
#res_train.to_csv("y_train.csv", index=False)


print('Completed!')