import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

# Select the features calculated in https://www.kaggle.com/cast42/santander-customer-satisfaction/select-features-rfecv/code
# 
# features = \
# ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 
# 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 
# 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var8_0', 'ind_var30', 'num_var4',
# 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_ult1', 
# 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var37', 
# 'num_var39_0', 'num_var42', 'saldo_var5', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 
# 'imp_trans_var37_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 
# 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 
# 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1',
# 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult3', 
# 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',
# 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'var38', 'n0']

# features = ['var15', 'saldo_var30', 'saldo_var42', 'num_var22_hace2', 'num_var22_ult3',
# 'num_meses_var5_ult3', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',\
# 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 
# 'var38', 'n0']

features = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult3', 
'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 
'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_ult1', 'ind_var8_0', 
'ind_var30_0', 'ind_var30', 'num_op_var41_hace2', 'num_op_var41_ult3', 
'num_var37_med_ult2', 'saldo_var5', 'saldo_var8', 'saldo_var26', 'saldo_var30', 
'saldo_var37', 'saldo_var42', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 
'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3',
'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3',
'num_op_var39_comer_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1',
'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1',
'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 
'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2',
'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult3',
'saldo_medio_var13_corto_hace2', 'var38', 'n0']

X_sel = X[features]

nrfold = 20

skf = cross_validation.StratifiedKFold(y, n_folds=nrfold, shuffle=True, random_state=1301)

scores = []
X = X.as_matrix()
y = y.as_matrix()
X_sel = X_sel.as_matrix()
test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]

y_pred_list = []
for k, (train,valid) in enumerate(skf):
    print ('Train on fold {}'.format(k))
    ratio = float(np.sum(y == 1)) / np.sum(y==0)
    clf = xgb.XGBClassifier(missing=9999999999,
                    max_depth = 6,
                    n_estimators=500,
                    learning_rate=0.1, 
                    nthread=4,
                    subsample=1.0,
                    colsample_bytree=0.5,
                    min_child_weight = 3,
                    scale_pos_weight = ratio,
                    seed=1301,
                    reg_alpha=0.03,
                    silent=True)
    clf.fit(X_sel[train], y[train], early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_sel[train], y[train]), (X_sel[valid], y[valid])])
    score = roc_auc_score( y[valid], clf.predict_proba(X_sel[valid], ntree_limit=clf.best_iteration)[:,1])
    scores.append((score, clf.best_iteration))
    print('Fold: %d, Dist. %s, #Rounds: %d AUC: %.3f' % (k+1, np.bincount(y[train]), clf.best_iteration, score))
    print('Predicting test values')
    if score > 0.83:
        print('Add fold {} to the ensemble'.format(k))
        y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)
        y_pred_list.append(y_pred)
    else:
        print('Removed fold {} from the ensemble because auc = {}'.format(k,score))

auc_scores =  pd.Series([score[0] for score in scores])
print ('Avg valid AUC sore %.4f +/- %.4f' % (auc_scores.mean(), auc_scores.std()))

pred = np.mean(np.array(y_pred_list), axis = 0)[:,1]
pred[np.where(sel_test['var15'] < 23)] = 0
submission = pd.DataFrame({"ID":test.index, "TARGET":pred})
submission.to_csv("submission.csv", index=False)

plt.figure()
plt.xlabel("Fold")
plt.ylabel("Cross validation score (roc auc)")
plt.plot(range(1, nrfold+1), [score[0] for score in scores])
plt.savefig('fold_auc.png', bbox_inches='tight', pad_inches=1)

plt.figure()
plt.xlabel("Fold")
plt.ylabel("N estimators")
plt.plot(range(1, nrfold+1), [score[1] for score in scores])
plt.savefig('fold_nrest.png', bbox_inches='tight', pad_inches=1)