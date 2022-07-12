import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

print("my 1st OOF...,,,,,,,,,,,,,never give up!!!!")
SEED = 7    # CR7 always..

application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
POS_CASH = pd.read_csv('../input/POS_CASH_balance.csv')
credit_card = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
previous_app = pd.read_csv('../input/previous_application.csv')
subm = pd.read_csv("../input/sample_submission.csv")


print("Converting...")
le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

bureau_cat_features = [f for f in bureau.columns if bureau[f].dtype == 'object']
for f in bureau_cat_features:
    bureau[f] = le.fit_transform(bureau[f].astype(str))
    nunique = bureau[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()
    bureau['NUNIQUE_'+f] = nunique[f]
    bureau.drop([f], axis=1, inplace=True)
bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

previous_app_cat_features = [f for f in previous_app.columns if previous_app[f].dtype == 'object']
for f in previous_app_cat_features:
    previous_app[f] = le.fit_transform(previous_app[f].astype(str))
    nunique = previous_app[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()
    previous_app['NUNIQUE_'+f] = nunique[f]
    previous_app.drop([f], axis=1, inplace=True)
previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

print("Merging...")
data_train = application_train.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(),
                                                             how='left', on='SK_ID_CURR')
data_test = application_test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(),
                                                           how='left', on='SK_ID_CURR')

data_train = data_train.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(),
                                                         how='left', on='SK_ID_CURR')
data_test = data_test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(),
                                                       how='left', on='SK_ID_CURR')
                                                       
data_train = data_train.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(),
                                                    how='left', on='SK_ID_CURR')
data_test = data_test.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(),
                                                  how='left', on='SK_ID_CURR')
                                                  
data_train = data_train.merge(previous_app.groupby('SK_ID_CURR').mean().reset_index(),
                                                          how='left', on='SK_ID_CURR')
data_test = data_test.merge(previous_app.groupby('SK_ID_CURR').mean().reset_index(),
                                                        how='left', on='SK_ID_CURR')
   
target_train = data_train['TARGET']
data_train.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
data_test.drop(['SK_ID_CURR'], axis=1, inplace=True)

cat_features = [f for f in data_train.columns if data_train[f].dtype == 'object']
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
cat_features_inds = column_index(data_train, cat_features)    
print("Cat features are: %s" % [f for f in cat_features])
print(cat_features_inds)

for col in cat_features:
    data_train[col] = le.fit_transform(data_train[col].astype(str))
    data_test[col] = le.fit_transform(data_test[col].astype(str))
    
data_train.fillna(-1, inplace=True)
data_test.fillna(-1, inplace=True)
cols = data_train.columns

ntrain = data_train.shape[0]
ntest = data_test.shape[0]

print(data_train.shape)

kf = KFold(data_train.shape[0], n_folds=5, shuffle=True, random_state=7)
NFOLDS = 5
x_train = np.array(data_train)
x_test = np.array(data_test)
y_train = target_train.ravel()

# from https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867/code
class SklearnWrapper(object):
    def __init__(self, clf, seed=7, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        print("Training..")
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        print("Predicting..")
        return self.clf.predict_proba(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        print("Training..")
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        print("Predicting..")
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)[:,1]  # or [:,0]
        oof_test_skf[i, :] = clf.predict(x_test)[:,1]  # or [:,0]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def get_oof_xgb(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)  # or [:,0]
        oof_test_skf[i, :] = clf.predict(x_test)  # or [:,0]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
    'nrounds': 350
}

cb_params = {
'iterations':1000,
'learning_rate':0.1,
'depth':6,
'l2_leaf_reg':40,
'bootstrap_type':'Bernoulli',
'subsample':0.7,
'scale_pos_weight':5,
'eval_metric':'AUC',
'metric_period':50,
'od_type':'Iter',
'od_wait':45,
'allow_writing_files':False    
}

xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
# cb = SklearnWrapper(clf=CatBoostClassifier, seed=SEED, params=cb_params)

print("xg..")
xg_oof_train, xg_oof_test = get_oof_xgb(xg)
print("et..")
et_oof_train, et_oof_test = get_oof(et)
print("rf..")
rf_oof_train, rf_oof_test = get_oof(rf)
# print("cb..")
# cb_oof_train, cb_oof_test = get_oof(cb)

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test), axis=1)

np.save('x_train', x_train)
np.save('x_test', x_test)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}

print("xgb cv..")
res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)
best_nrounds = res.shape[0] - 1

"""
print("\nCatBoost...")                                     
cb_model = CatBoostClassifier(iterations=1000,
                              learning_rate=0.1,
                              depth=6,
                              l2_leaf_reg=40,
                              bootstrap_type='Bernoulli',
                              subsample=0.7,
                              scale_pos_weight=5,
                              eval_metric='AUC',
                              metric_period=50,
                              od_type='Iter',
                              od_wait=45,
                              random_seed=17,
                              allow_writing_files=False)

cb_model.fit(X_train, y_train,
             eval_set=(X_valid, y_valid),
             cat_features=cat_features_inds,
             use_best_model=True,
             verbose=True)
             
print('AUC:', roc_auc_score(y_valid, cb_model.predict_proba(X_valid)[:,1]))
y_preds = cb_model.predict_proba(data_test)[:,1]
subm['TARGET'] = y_preds
subm.to_csv('submission.csv', index=False)
"""
print("meta xgb train..")
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
fi = gbdt.predict(dtest)
fi = np.array(fi)
np.save('fi', fi)

subm['TARGET'] = fi
subm.to_csv('stack1.csv', index=False)
