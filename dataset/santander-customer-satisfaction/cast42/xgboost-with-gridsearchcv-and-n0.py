import numpy as np
import pandas as pd

print('loading train data')
train_data = pd.read_csv('../input/train.csv')

print('loading test data')
test_data = pd.read_csv('../input/test.csv')

# training data
X_train = train_data.drop(['ID','TARGET'], axis=1)
y_train = train_data['TARGET']

# test data
X_test = test_data.drop(['ID'], axis=1)

# Add zeros per row as extra feature
X_train['n0'] = (X_train == 0).sum(axis=1)
X_test['n0'] = (X_test == 0).sum(axis=1) 

# print('select the same columns for test data')
# X_test = X_test[X_train.columns]

print('feature selection')
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

seed = 1301
tree_clf = ExtraTreesClassifier(n_estimators=400, random_state=seed, 
   class_weight='balanced_subsample', bootstrap=True, oob_score=True,
   min_samples_leaf=3, max_features='log2', max_depth=5)
selector = tree_clf.fit(X_train, y_train)
# print(tree_clf.feature_importances_)

fs = SelectFromModel(selector, prefit=True)

X_train_trans = fs.transform(X_train)
X_test_trans = fs.transform(X_test)

features = [ f for f,s in zip(X_train.columns, fs.get_support()) if s]

# features = ['var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult3', 
# 'imp_op_var41_comer_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 
# 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var5', 'ind_var30',
# 'num_op_var41_hace2', 'num_op_var39_ult3', 'saldo_var5', 'saldo_var30', 'saldo_var42', 
# 'var36', 'imp_trans_var37_ult1', 'num_ent_var16_ult1', 'num_var22_hace2',
# 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3',
# 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3',
# 'num_op_var39_comer_ult3', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 
# 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 
# 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'var38', 'n0']
# X_train_trans = X_train[features]
# X_test_trans = X_test[features]

print('Feature selections with ExtraTreees selected {} features.'.format(len(features)))
print (features)


from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn import cross_validation

# I commented these as they take too long to run under kaggle environment

#print('XGBoost with grid search')
# play with these params
#params={
#    'max_depth': [5], #[3,4,5,6,7,8,9],
#    'subsample': [0.6], #[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#    'colsample_bytree': [0.5], #[0.5,0.6,0.7,0.8],
#    'n_estimators': [2000],
#    'reg_alpha': [0.03] #[0.01, 0.02, 0.03, 0.04]
#}

#xgb_clf = xgb.XGBClassifier()
#rs = GridSearchCV(xgb_clf,
#                  params,
#                  cv=5,
#                  scoring="log_loss",
#                  n_jobs=1,
#                  verbose=2)
#rs.fit(X_train_trans, y_train)
#best_est = rs.best_estimator_
#print(best_est)

#XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
#       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
#       min_child_weight=1, missing=None, n_estimators=2000, nthread=-1,
#       objective='binary:logistic', reg_alpha=0.03, reg_lambda=1,
#       scale_pos_weight=1, seed=0, silent=True, subsample=0.6)
#('Roc AUC: ', 0.97747361197765803)

#
# Using missing param has no impact on the performance
#
#XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
#       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
#       min_child_weight=1, missing=9999999999, n_estimators=2000,
#       nthread=-1, objective='binary:logistic', reg_alpha=0.03,
#       reg_lambda=1, scale_pos_weight=1, seed=0, silent=True,
#       subsample=0.6)
#('Roc AUC: ', 0.97747361197765803)


X_t, X_valid, y_t, y_valid = \
  cross_validation.train_test_split(X_train_trans, y_train, random_state=1301,
  stratify=y_train, test_size=0.3)
print('XGBoost')
best_est = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,
       min_child_weight=1, missing=None, n_estimators=2000, nthread=-1,
       objective='binary:logistic', reg_alpha=0.03, reg_lambda=1,
       scale_pos_weight=1, seed=1301, silent=True, subsample=0.6)
best_est.fit(X_t, y_t, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_t, y_t), (X_valid, y_valid)])

# Roc AUC with all train data
validation = best_est.predict_proba(X_train_trans, ntree_limit=best_est.best_iteration)
print("Roc AUC: ", roc_auc_score(y_train, validation[:,1], average='macro'))

print('prepare for submisssion')
probs = best_est.predict_proba(X_test_trans, ntree_limit=best_est.best_iteration)


# ids for test data
ids = test_data.ix[X_test.index, 'ID']
submission = pd.DataFrame({"ID":ids, "TARGET": probs[:,1]})
submission.to_csv("submission.csv", index=False)

print('done')