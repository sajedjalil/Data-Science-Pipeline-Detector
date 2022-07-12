import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

df_train = pandas.read_csv("../input/train.csv")
df_test  = pandas.read_csv("../input/test.csv")   

id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1) #.values
X_test = df_test.drop(['ID'], axis=1) #.values

nineteenfeatures = ['imp_ent_var16_ult1',
                    'var38',
                    'ind_var30',
                    'delta_imp_aport_var13_1y3',
                    'saldo_medio_var13_corto_hace2',
                    'num_op_var39_hace3',
                    'imp_var43_emit_ult1',
                    'num_meses_var5_ult3',
                    'delta_num_aport_var13_1y3',
                    'num_var42_0',
                    'imp_op_var40_ult1',
                    'num_var22_ult1',
                    'saldo_var5',
                    'num_op_var40_ult1',
                    'imp_aport_var13_ult1',
                    'saldo_var42', 'ind_var39_0',
                    'num_aport_var13_ult1',
                    'var15']
X_train=X_train[nineteenfeatures]
X_test=X_test[nineteenfeatures]

X_train=X_train.values
X_test=X_test.values

clf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=8)

scores = cross_validation.cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=5) 
print(scores.mean())


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

submission = pandas.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission_BTB_RF_v2.csv", index=False)

