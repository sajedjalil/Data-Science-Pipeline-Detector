import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns


test_2_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
test_1_df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
train_df= pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
train_df=train_df.dropna(axis=0,subset=['CNT_PAYMENT'])
train_df['CNT_PAYMENT'] = train_df['CNT_PAYMENT'].astype('int')

for i in [test_1_df, test_2_df, train_df]:
    i['screwratio1']=(i.AMT_CREDIT-i.AMT_GOODS_PRICE)/i.AMT_GOODS_PRICE
    i['screwratio2']=(i.AMT_CREDIT-i.AMT_GOODS_PRICE)/i.AMT_CREDIT
    i['saint_CNT']=i.AMT_CREDIT/i.AMT_ANNUITY
    i['angel_CNT']=i.AMT_GOODS_PRICE/i.AMT_ANNUITY
    i['simple_diff']=i.AMT_CREDIT-i.AMT_GOODS_PRICE

feats=['saint_CNT', 'AMT_ANNUITY', 'angel_CNT', 'AMT_GOODS_PRICE', 'screwratio2', 'screwratio1', 'AMT_CREDIT','simple_diff']
train_df=train_df.fillna(-1)
clf = LGBMClassifier(
            nthread=4,
            objective='multiclass',
            n_estimators=1000,
            learning_rate=0.02, #originally 0.02
            num_leaves=50,
            max_depth=11,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=100, )
print('fitting')

clf.fit(train_df[feats], train_df['CNT_PAYMENT'], verbose= 500)
print('training on previous apps done')
for frame in [[test_1_df,'train'],[test_2_df,'test']]:
    test_df=frame[0]
    tag=frame[1]
    j=clf.predict_proba(test_df[feats], verbose= 500)
    test_df=test_df.fillna(-1)
    gc.collect()
    feature_importance_df = pd.DataFrame()
    sqsum=[]
    test_df['certainty']=0
    print(np.arange(0,j.shape[1]-1))
    print(j.shape)
    for k in np.arange(0,j.shape[1]-1):
            test_df['CNT_prob_'+str(k)]=j[:,k]
            test_df['CNT_prob_sq_'+str(k)]=test_df['CNT_prob_'+str(k)]*test_df['CNT_prob_'+str(k)]
            test_df['certainty']+=test_df['CNT_prob_sq_'+str(k)]

    predictions=pd.DataFrame()
    for k in np.arange(0,j.shape[1]-1):
            predictions[str(clf.classes_[k])]=j[:,k]
    predictions['best_guess']=predictions.idxmax(axis=1)
    predictions['best_guess']=predictions.best_guess.astype('int')
    test_df['lgbm_CNT']=predictions['best_guess']
    print('starting the long, arduous task of computing interest rates')
    x=[]
    #must loop here - np.rate has a bug
    for i in range(0,len(test_df.index)):    
            x.append(np.rate(test_df['lgbm_CNT'][i],test_df['AMT_ANNUITY'][i],-test_df['AMT_CREDIT'][i],0.0))
    test_df['rate_credit']=x
    del x
    x=[]
    #must loop here - np.rate has a bug
    for i in range(0,len(test_df.index)):    
            x.append(np.rate(test_df['lgbm_CNT'][i],test_df['AMT_ANNUITY'][i],-test_df['AMT_GOODS_PRICE'][i],0.0))
    test_df['rate_goods']=x
    del x
    test_df[['rate_goods','SK_ID_CURR','lgbm_CNT','rate_credit','certainty']].to_csv('lgbm_CNT_'+tag+'.csv', index= False)
feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = feats
feature_importance_df["importance"] = clf.feature_importances_
feature_importance_df[['feature', 'importance']].to_csv('importances.csv', index= False)

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    
display_importances(feature_importance_df)








        

