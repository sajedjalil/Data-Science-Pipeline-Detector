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
features = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 
'imp_op_var39_comer_ult3', 'imp_op_var40_efect_ult3', 'imp_op_var41_comer_ult1', 
'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var39_efect_ult1', 
'imp_op_var39_ult1', 'ind_var41_0', 'num_var1_0', 'num_var4', 'num_var5_0', 
'num_var8_0', 'num_var25_0', 'num_op_var41_hace2', 'num_op_var41_ult3', 
'num_var30_0', 'num_var37_med_ult2', 'num_var42_0', 'saldo_var1', 'saldo_var5',
'saldo_var8', 'saldo_var14', 'saldo_var26', 'saldo_var25', 'saldo_var30',
'saldo_var37', 'saldo_var40', 'saldo_var42', 'var36', 'imp_var7_recib_ult1',
'imp_reemb_var17_ult1', 'imp_var43_emit_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 
'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 
'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1',
'num_op_var39_efect_ult1', 'num_reemb_var17_ult1', 'num_var43_emit_ult1',
'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1',
'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3',
'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2',
'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3',
'saldo_medio_var12_hace2', 'var38', 'n0']
X_sel = X[features]

nrfold = 4

# skf = cross_validation.StratifiedKFold(y, n_folds=nrfold, shuffle=True, random_state=1301)

scores = []
# X = X.as_matrix()
# y = y.as_matrix()
# X_sel = X_sel.as_matrix()
test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]
X_train, X_valid, y_train, y_valid = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.5)
y_pred_list = []
for k in range(nrfold):
    print ('Train on fold {}'.format(k))
    X_train, X_valid, y_train, y_valid = \
      cross_validation.train_test_split(X_sel, y, stratify=y, test_size=0.5)
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
                    reg_alpha=0.01,
                    seed=1301)
    clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_valid, y_valid)])
    score = roc_auc_score( y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1])
    scores.append((score, clf.best_iteration))
    print('Fold: %d, Dist. %s, #Rounds: %d AUC: %.3f' % (k+1, np.bincount(y_train), clf.best_iteration, score))
    y_train_pred = clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)
    print('Predicting test values')
    y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)
    y_pred_list.append(y_pred)
    if k==0:
        y_pred_geom = y_pred[:,1]
        y_train_pred_geom = y_train_pred[:,1]
    else:
        y_pred_geom *= y_pred[:,1]
        y_train_pred_geom *= y_train_pred[:,1]

auc_scores =  pd.Series([score[0] for score in scores])
print ('Avg valid AUC sore %.4f +/- %.4f' % (auc_scores.mean(), auc_scores.std()))

print ('AUC geom mean train predictions: {:0.4f}'.format(roc_auc_score(y, y_train_pred_geom)))

y_train_pred_geom = np.power(y_train_pred_geom, 1.0/nrfold)
submission = pd.DataFrame({"ID":X_sel.index, "TARGET":y_train_pred_geom})
submission.to_csv("submission_train_geom.csv", index=False)

pred = np.mean(np.array(y_pred_list), axis = 0)
submission = pd.DataFrame({"ID":test.index, "TARGET":pred[:,1]})
submission.to_csv("submission.csv", index=False)

y_pred_geom = np.power(y_pred_geom, 1.0/nrfold)
submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred_geom})
submission.to_csv("submission_geom.csv", index=False)

plt.figure()
plt.title('{} cross folds AUC: {:0.4f} +/- {:0.4f}'.format(nrfold, auc_scores.mean(), auc_scores.std()))
plt.xlabel("Fold")
plt.ylabel("Cross validation score (roc auc)")
plt.plot(range(1, nrfold+1), [score[0] for score in scores])
plt.savefig('fold_auc.png', bbox_inches='tight', pad_inches=1)

plt.figure()
plt.xlabel("Fold")
plt.ylabel("N estimators")
plt.plot(range(1, nrfold+1), [score[1] for score in scores])
plt.savefig('fold_nrest.png', bbox_inches='tight', pad_inches=1)