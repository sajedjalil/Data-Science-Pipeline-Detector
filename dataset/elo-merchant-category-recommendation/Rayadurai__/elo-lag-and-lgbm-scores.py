

import pandas as pd
import numpy as np
import os,gc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
tqdm.pandas()
pd.options.display.max_columns =100
gc.enable()
#import swifter 


path = '../input/'
#merchant = pd.read_csv('merchants.csv')
new_trans = pd.read_csv(path+'new_merchant_transactions.csv',parse_dates=['purchase_date'])
hist_trans = pd.read_csv(path+'historical_transactions.csv',parse_dates=['purchase_date'])


def load_data(dataset):
    path_df = path + dataset +'.csv'
    month_parser = lambda x : [ pd.datetime.strptime(i,'%Y-%m') if i is not np.NaN   else datetime.date(2017, 3, 9) for i in x  ]
    df = pd.read_csv(path_df,parse_dates=['first_active_month'],
                        date_parser=month_parser)
    df['elapsed_days'] = (df.first_active_month.dt.date.max() - df['first_active_month'].dt.date).dt.days
    for i in ['feature_1','feature_2','feature_3']:
        cat_cols = pd.get_dummies(df[i],prefix=i)
        df = pd.concat([df.drop(i,axis=1),cat_cols],axis=1)
    
    return df

def auth_agg(hist_trans):
    agg_fun = {'authorized_flag': ['sum', 'mean']}
    hist_trans['authorized_flag'] = hist_trans.authorized_flag.map({'Y':1,'N':0}).astype(int)
    auth_agg = hist_trans.groupby(['card_id']).agg(agg_fun)
    auth_agg.columns = ['hist_'.join(col).strip() for col in auth_agg.columns.values]
    auth_agg.reset_index(inplace=True)
    return auth_agg  
    
hist_auth  = auth_agg(hist_trans)
hist_trans = hist_trans[hist_trans.authorized_flag == 1]


train = load_data('train')
test = load_data('test')
train_length = len(train)
df = pd.concat([train,test],axis=0)
print('dataset loaded')

del hist_trans['authorized_flag'],new_trans['authorized_flag']
del train,test

def process_transactions(trans,agg_name=''):
    
    time_of_the_day = pd.get_dummies(pd.cut(trans.purchase_date.dt.hour,
                                            bins=5),prefix='time_of_the_day')
    trans = pd.concat([trans,time_of_the_day],axis=1)
    
    for i in ['category_1','category_2','category_3']:
        cat_cols = pd.get_dummies(trans[i],prefix=i)
        trans = pd.concat([trans.drop(i,axis=1),cat_cols],axis=1)
     
    trans['weekend'] = (trans.purchase_date.dt.weekday >=5).astype(int)
    
    trans_group = trans.groupby('card_id')    
    # lag variables
    trans_positive_month = trans[trans.month_lag >=0 ]
    trans_neg_month = trans[trans.month_lag <0 ]
    
    pos_month_lag = (trans_positive_month.groupby('card_id')['month_lag']).sum()
    neg_month_lag = (trans_neg_month.groupby('card_id')['month_lag']).sum()
    pos_month_lag_m = (trans_positive_month.groupby('card_id')['month_lag']).mean()
    neg_month_lag_m = (trans_neg_month.groupby('card_id')['month_lag']).mean()
    
    del trans_positive_month,trans_neg_month
    #ratio_of_month_lag = (trans_group['month_lag']).apply( lambda x : sum(x[x<0]) / (sum(x[x>=0])+1))
    
    trans_positive_purchase = trans[trans.purchase_amount >=0 ]
    trans_neg_purchase = trans[trans.purchase_amount <0 ]
    
    pos_purchase = (trans_positive_purchase.groupby('card_id')['purchase_amount']).sum()
    neg_purchase = (trans_neg_purchase.groupby('card_id')['purchase_amount']).sum()
    pos_purchase_m = (trans_positive_purchase.groupby('card_id')['purchase_amount']).mean()
    neg_purchase_m = (trans_neg_purchase.groupby('card_id')['purchase_amount']).mean()
    
    for i in ['pos_month_lag','neg_month_lag','pos_month_lag_m',
              'neg_month_lag_m','pos_purchase','neg_purchase',
              'pos_purchase_m','neg_purchase_m']:
        df[agg_name + i] = df.card_id.map(eval(i).to_dict())
        print('processed',i)
    return trans


def aggregate_transactions(trans,agg_name = ''):
    
    agg_func = {
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'median','mean', 'max'],
        'installments': ['sum', 'mean', 'max'],
        'purchase_date': [np.ptp],
        'weekend' : ['sum']
        }
    category =  {i:['mean', 'sum'] for i in trans.columns.tolist() if 'category_' in i}
    time_day =  {i:['sum', 'mean'] for i in trans.columns.tolist() if 'time_day' in i}
    
    agg_func.update(category)
    agg_func.update(time_day)
    
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [agg_name + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df_ = (trans.groupby('card_id')
          .size()
          .reset_index(name=agg_name + 'transactions_count'))
    
    agg_trans = pd.merge(df_, agg_trans, on='card_id', how='left')
    
    for j in agg_trans.columns.tolist():
        if 'purchase_date_ptp' in j:
            agg_trans[j] = agg_trans[j].dt.days

    return agg_trans


print('processing_all_trans')

hist_trans = process_transactions(hist_trans,agg_name='hist_')
hist_trans = aggregate_transactions(hist_trans,'hist_')

new_trans = process_transactions(new_trans,agg_name='new_')
new_trans = aggregate_transactions(new_trans,'new_')

df = pd.merge(df, hist_trans, on='card_id',how='left')
df = pd.merge(df, new_trans, on='card_id',how='left')
df = pd.merge(df, hist_auth, on='card_id',how='left')


features  = [c for c in df.columns if c not in ['card_id',
                                                   'merchant_category_id',
                                                   'first_active_month',
                                                   'merchant_id',
                                                   'purchase_amount',
                                                   'purchase_date',
                                                   'target']]



train = df[:train_length]
test  = df[train_length:]
target = train['target']

train = train[features].fillna(0.0011)
test = test[features].fillna(0.0011)

train.to_csv('train_before_lgb.csv')
target.to_csv('train_target_before_lgb.csv')


del df


# some random parameters
lgb_params = {
        "objective" : "regression",
        "metric" : "rmse",
        "boosting_type" : "gbdt",
        'bagging_fraction': 0.3, 
        'bagging_frequency': 11, 
        'colsample_bytree': 0.68, 
        'feature_fraction': 0.5, 
        'learning_rate': 0.05, 
        'max_depth': 16, 
        'min_child_samples': 370,
        'min_split_gain': 21, 
        'n_estimators': 301, 
        'num_leaves': 81,
        'reg_alpha': 0.5, 
        'reg_lambda': 0.1}



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

FOLDs = KFold(n_splits=8, shuffle=True, random_state=6547)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_))
    num_round = 2000
    model = lgb.train(lgb_params, trn_data,
                num_round, valid_sets = [trn_data, val_data],
                verbose_eval=100, 
                early_stopping_rounds =200)
    oof_lgb[val_idx] = model.predict(train.iloc[val_idx], num_iteration=model.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = model.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += model.predict(test, num_iteration=model.best_iteration) / FOLDs.n_splits
    

print(np.sqrt(mean_squared_error(oof_lgb, target)))



plt.figure(figsize=(14,18))
sns.barplot(x="importance",
            y="feature",
            data=feature_importance_df_lgb.sort_values(by="importance",
                                           ascending=False))
plt.title('lgb features avg')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] =  predictions_lgb 
sub_df.to_csv("submission.csv", index=False)