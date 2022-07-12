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
import gc

print("my 3rd OOF...!!")
SEED = 7    # CR7 always..
subm = pd.read_csv("../input/sample_submission.csv")
print("Loading data...\n")
lb=LabelEncoder()
def LabelEncoding_Cat(df):
    df=df.copy()
    Cat_Var=df.select_dtypes('object').columns.tolist()
    for col in Cat_Var:
        df[col]=lb.fit_transform(df[col].astype('str'))
    return df    

def Fill_NA(df):
    df=df.copy()
    Num_Features=df.select_dtypes(['float64','int64']).columns.tolist()
    df[Num_Features]= df[Num_Features].fillna(-999)
    return df

bureau= (pd.read_csv("../input/bureau.csv")
         .pipe(LabelEncoding_Cat))

cred_card_bal=(pd.read_csv("../input/credit_card_balance.csv")
               .pipe(LabelEncoding_Cat))

pos_cash_bal=(pd.read_csv("../input/POS_CASH_balance.csv")
               .pipe(LabelEncoding_Cat))
               
prev =(pd.read_csv("../input/previous_application.csv") 
               .pipe(LabelEncoding_Cat))

print("Preprocessing...\n")
Label_1=[s+'_'+l for s in bureau.columns.tolist() if s!='SK_ID_CURR' for l in ['mean','count','median','max']]
avg_bureau=bureau.groupby('SK_ID_CURR').agg(['mean','count','median','max']).reset_index()
avg_bureau.columns=['SK_ID_CURR']+Label_1

Label_2=[s+'_'+l for s in cred_card_bal.columns.tolist() if s!='SK_ID_CURR' for l in ['mean','count','median','max']]
avg_cred_card_bal=cred_card_bal.groupby('SK_ID_CURR').agg(['mean','count','median','max']).reset_index()
avg_cred_card_bal.columns=['SK_ID_CURR']+Label_2

Label_3=[s+'_'+l for s in pos_cash_bal.columns.tolist() if s not  in ['SK_ID_PREV','SK_ID_CURR'] for l in ['mean','count','median','max']]
avg_pos_cash_bal=pos_cash_bal.groupby(['SK_ID_PREV','SK_ID_CURR'])\
            .agg(['mean','count','median','max']).groupby(level='SK_ID_CURR')\
            .agg('mean').reset_index()
avg_pos_cash_bal.columns=['SK_ID_CURR']+Label_3

Label_4=[s+'_'+l for s in prev.columns.tolist() if s!='SK_ID_CURR' for l in ['mean','count','median','max']]
avg_prev=prev.groupby('SK_ID_CURR').agg(['mean','count','median','max']).reset_index()
avg_prev.columns=['SK_ID_CURR']+Label_4

del(Label_1,Label_2,Label_3,Label_4)
tr = pd.read_csv("../input/application_train.csv") 
te =pd.read_csv("../input/application_test.csv")

tri=tr.shape[0]
y = tr.TARGET.copy()

tr_te=(tr.drop(labels=['TARGET'],axis=1).append(te)
         .pipe(LabelEncoding_Cat)
         .pipe(Fill_NA)
         .merge(avg_bureau,on='SK_ID_CURR',how='left')
         .merge(avg_cred_card_bal,on='SK_ID_CURR',how='left')
         .merge(avg_pos_cash_bal,on='SK_ID_CURR',how='left')
         .merge(avg_prev,on='SK_ID_CURR',how='left'))

del(tr,te,bureau,cred_card_bal,pos_cash_bal,prev, avg_prev,avg_bureau,avg_cred_card_bal,avg_pos_cash_bal)
gc.collect()

print("Preparing data...\n")
tr_te.drop(labels=['SK_ID_CURR'],axis=1,inplace=True)
tr=tr_te.iloc[:tri,:].copy()
te=tr_te.iloc[tri:,:].copy()
data_train = tr
data_test = te

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
y_train = y.ravel()

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
    'eval_metric': 'auc',
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
cb = SklearnWrapper(clf=CatBoostClassifier, seed=SEED, params=cb_params)

print("xg..")
xg_oof_train, xg_oof_test = get_oof_xgb(xg)
print("et..")
et_oof_train, et_oof_test = get_oof(et)
print("rf..")
rf_oof_train, rf_oof_test = get_oof(rf)
print("cb..")
cb_oof_train, cb_oof_test = get_oof(cb)

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, cb_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, cb_oof_test), axis=1)

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
    'eval_metric': 'auc',
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
subm.to_csv('stack3.csv', index=False)