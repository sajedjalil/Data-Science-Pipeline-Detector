# Basic Kernel or reference: https://www.kaggle.com/kailex/tidy-xgb-0-778/code

import numpy as np
import pandas as pd
import gc
import lightgbm as gbm
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

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

del(tr_te)

Dparam = {'objective' : 'binary',
          'boosting_type': 'gbdt',
          'metric' : 'auc',
          'nthread' : 4,
          'shrinkage_rate':0.025,
          'max_depth':8,
          'min_data_in_leaf':100,
          'min_child_weight': 2,
          'bagging_fraction':0.75,
          'feature_fraction':0.75,
          'min_split_gain':.01,
          'lambda_l1':1,
          'lambda_l2':1,
          'num_leaves':36}     

print("Training model...\n")

folds = KFold(n_splits=5, shuffle=True, random_state=123456)

oof_preds = np.zeros(tr.shape[0])
sub_preds = np.zeros(te.shape[0])
feature_importance_df = pd.DataFrame()
feats = [f for f in tr.columns if f not in ['SK_ID_CURR']]
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(tr)):
    dtrain =gbm.Dataset(tr.iloc[trn_idx], y.iloc[trn_idx])
    dval =gbm.Dataset(tr.iloc[val_idx], y.iloc[val_idx])
    m_gbm=gbm.train(params=Dparam,train_set=dtrain,num_boost_round=3000,verbose_eval=1000,valid_sets=[dtrain,dval],valid_names=['train','valid'])
    oof_preds[val_idx] = m_gbm.predict(tr.iloc[val_idx])
    sub_preds += m_gbm.predict(te) / folds.n_splits
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = m_gbm.feature_importance()
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y.iloc[val_idx],oof_preds[val_idx])))
    del dtrain,dval
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))   

def display_importances(feature_importance_df_):
    # Plot feature importances/ Oliver's function 
    #Kernel: https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

display_importances(feature_importance_df)

print("Output Model")
tr_oof = pd.read_csv("../input/application_train.csv",usecols=['SK_ID_CURR','TARGET'])
tr_oof['TARGET_oof']=oof_preds.copy()
tr_oof.to_csv("Target_Simple_2_Model_GBM_oof.csv", index=False)


Submission=pd.read_csv("../input/sample_submission.csv")
Submission['TARGET']=sub_preds.copy()
Submission.to_csv("Lightgbm_Simple_2_Model.csv", index=False)   

