# %% [code]
# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold,GroupKFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import math
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code]
########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# %% [code]
########################### Vars
#################################################################################
SEED = 7
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

# %% [code]
########################### DATA LOAD
#################################################################################
print('Loading Data')
root_dir = '../input/data-pre-massaging'
train_df = pd.read_pickle(os.path.join(root_dir,'train_transaction.pkl'))
print('\tLoaded train_transaction.pkl')
test_df = pd.read_pickle(os.path.join(root_dir,'test_transaction.pkl'))
print('\tLoaded test_transaction.pkl')
train_identity = pd.read_pickle(os.path.join(root_dir,'train_identity.pkl'))
print('\tLoaded train_identity.pkl')
test_identity = pd.read_pickle(os.path.join(root_dir,'test_identity.pkl'))
print('\tLoaded test_identity.pkl')
    
base_columns = list(train_df) + list(train_identity)
print('Shape control:', train_df.shape, test_df.shape)

# %% [code]
########################### D9 and TransactionDT
# Let's add temporary "time variables" for aggregations
# and add normal "time variables"

# Also, seems that D9 column is an hour
# and it is the same as df['DT'].dt.hour
for df in [train_df, test_df]:
    # Temporary
    # TransactionDT is the number of seconds from a date point (START_DATE)
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    # Here we get the month/week/day, but not within the year, since the start point. 
    # That's why it's added the extra years since start point (multiplied by a quantity accordingly to if it's month/week/day)
    # DataFrame/Series .dt allows us to access values as datetime.
    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month
    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear
    
    df['DT_hour'] = df['DT'].dt.hour
    df['DT_day_week'] = df['DT'].dt.dayofweek
    df['DT_day'] = df['DT'].dt.day
    
    # D9 column
    # zeroing every value, or 1 if it was null. why?
    df['D9'] = np.where(df['D9'].isna(),0,1)

# %% [code]
########################### Reset values for "noise" card1
i_cols = ['card1']

for col in i_cols: 
    # gets all possible card values
    valid_card = pd.concat([train_df[[col]], test_df[[col]]])
    # does a value count
    valid_card = valid_card[col].value_counts()
    # discards cards that have only appeared 1 or 2 times
    valid_card = valid_card[valid_card>2]
    # since the variable now holds a value count, the index holds the field's values that we want
    valid_card = list(valid_card.index)

    # these nans every value in the column that does not exist in the other dataset.
    # i.e if card nr 1234 exists in train but not in test, it gets a nan. And viceversa
    train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
    test_df[col]  = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

    # this makes sure that all values not "naned" above are within the list of cards we considered 'valid' in the above step
    # so this only applies MORE nans in case they are not on the list
    train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
    test_df[col]  = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)

# %% [code]
########################### M columns (except M4)
# All these columns are binary encoded 1/0 (This was done in the pre-massaging, these used to be categorical values)
# We can have some features from it
i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

for df in [train_df, test_df]:
    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)

# %% [code]
########################### ProductCD and M4 Target mean
for col in ['ProductCD','M4']:
    # gets a mean by value of the column. Reset_index makes it so the grouped by column values are not the index, and a column instead.
    # it then renames the mean column
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                        columns={'mean': col+'_target_mean'})
    # once again sets the index as the column values, but maintains the col column (so we basically have it twice, as column and as index)
    temp_dict.index = temp_dict[col].values
    # we needed the values as index so to_dict creates a dictionary with them
    temp_dict = temp_dict[col+'_target_mean'].to_dict()
    
    # now we create a new column that finds the 'col' value on the dict key, and maps the dict value as the new column's value
    # ie dict is {23046: 0.05, 62397: 0.047} then if the 'col' value is 23046 then the new column value is .05
    # if the 'col' value is 62397 the new column's value is .047 and so on
    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)

# %% [code]
########################### TransactionAmt

# Let's add some kind of client uID based on cardID ad addr columns
# The value will be very specific for each client so we need to remove it
# from final feature. But we can use it for aggregations.
train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)
test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)

train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

train_df['uid3'] = train_df['uid2'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)
test_df['uid3'] = test_df['uid2'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)

# Check if the Transaction Amount is common or not (we can use freq encoding here)
# In our dialog with a model we are telling to trust or not to these values   
train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

# For our model current TransactionAmt is a noise
# https://www.kaggle.com/kyakovlev/ieee-check-noise
# (even if features importances are telling contrariwise)
# There are many unique values and model doesn't generalize well
# Lets do some aggregations
i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']

for col in i_cols:
    for agg_type in ['mean','std']:
        new_col_name = col+'_TransactionAmt_'+agg_type
        temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col,'TransactionAmt']]])
        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()   
    
        train_df[new_col_name] = train_df[col].map(temp_df)
        test_df[new_col_name]  = test_df[col].map(temp_df)
           
# Small "hack" to transform distribution 
# (doesn't affect auc much, but I like it more)
# please see how distribution transformation can boost your score 
# (not our case but related)
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])

# %% [code]
########################### 'P_emaildomain' - 'R_emaildomain'
p = 'P_emaildomain'
r = 'R_emaildomain'
uknown = 'email_not_provided'

for df in [train_df, test_df]:
    df[p] = df[p].fillna(uknown)
    df[r] = df[r].fillna(uknown)
    
    # Check if P_emaildomain matches R_emaildomain
    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)

    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])

## Local test doesn't show any boost here, 
## but I think it's a good option for model stability 

## Also, we will do frequency encoding later

# %% [code]
########################### Device info
for df in [train_identity, test_identity]:
    ########################### Device info
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
    
    ########################### Device info 2
    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
    
    ########################### Browser
    df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
    df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

# %% [code]
########################### Merge Identity columns
temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df,temp_df], axis=1)
    
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df,temp_df], axis=1)

# %% [code]
########################### Freq encoding
i_cols = ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain',
          'DeviceInfo','DeviceInfo_device','DeviceInfo_version',
          'id_30','id_30_device','id_30_version',
          'id_31_device',
          'id_33',
          'uid','uid2','uid3',
         ]

for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


for col in ['DT_M','DT_W','DT_D']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train_df[col+'_total'] = train_df[col].map(fq_encode)
    test_df[col+'_total']  = test_df[col].map(fq_encode)
        

periods = ['DT_M','DT_W','DT_D']
i_cols = ['uid']
for period in periods:
    for col in i_cols:
        new_column = col + '_' + period
            
        temp_df = pd.concat([train_df[[col,period]], test_df[[col,period]]])
        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
        fq_encode = temp_df[new_column].value_counts().to_dict()
            
        train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
        test_df[new_column]  = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)
        
        train_df[new_column] /= train_df[period+'_total']
        test_df[new_column]  /= test_df[period+'_total']

# %% [code]
########################### Encode Str columns
# For all such columns (probably not)
# we already did frequency encoding (numeric feature)
# so we will use astype('category') here
for col in list(train_df):
    if train_df[col].dtype=='O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col]  = test_df[col].fillna('unseen_before_label')
        
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train_df[col])+list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

# %% [code]
########################### Model Features 
## We can use set().difference() but the order matters
## Matters only for deterministic results
## In case of remove() we will not change order
## even if variable will be renamed
## please see this link to see how set is ordered
## https://stackoverflow.com/questions/12165200/order-of-unordered-python-sets
rm_cols = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'uid','uid2','uid3',             # Our new client uID -> very noisy data
    'bank_type',                     # Victims bank could differ by time
    'DT','DT_M','DT_W','DT_D',       # Temporary Variables
    'DT_hour','DT_day_week','DT_day',
    'DT_D_total','DT_W_total','DT_M_total',
    'id_30','id_31','id_33',
]

# %% [code]
########################### Features elimination 
from scipy.stats import ks_2samp
features_check = []
# base_columns were the columns from the dataset as it was loaded
# meaning without extra columns created by feature engineering.
columns_to_check = set(list(train_df)).difference(base_columns+rm_cols)
for i in columns_to_check:
    # scipy.stats.ks_2samp computes the Kolmogorov-Smirnov statistic on 2 samples
    # This tests whether 2 samples are drawn from the same distribution.
    # returns 2 values, [0] is KS statistic and [1] is p-value 
    # If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.
    features_check.append(ks_2samp(test_df[i], train_df[i])[1])

features_check = pd.Series(features_check, index=columns_to_check).sort_values() 
# we are discarding features for which the p-value is 0 between the test and train distributions. Meaning even though they are the same feature
# the samples were extracted from different distributions??? i'm assuming this hurts the model by making it harder for it to generalize
# since it learned something from a feature that later proves to behave itself differently. 
# Samples were not representative? Thats why when analyzed they seem to come from different distributions perhaps?
features_discard = list(features_check[features_check==0].index)
print(features_discard)

# We will reset this list for now (use local test drop),
# Good droping will be in other kernels
# with better checking
features_discard = [] 

# Final features list
features_columns = [col for col in list(train_df) if col not in rm_cols + features_discard]

# %% [code]
########################### Model params
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':.007,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
#                     'colsample_bytree': 0.7,
                    'feature_fraction': 0.4505111428218119,
#                     'subsample_freq':1,
#                     'subsample':0.7,
                    'bagging_fraction': 0.7948351068174488,
                    'n_estimators':10000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                    'min_child_samples': 100,
                    'reg_alpha': 0.03642787958374098,
                    'reg_lambda': 0.11826155670871391,
                }

# %% [code]
########################### Model
import lightgbm as lgb

def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    
#     folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    folds = GroupKFold(n_splits=NFOLDS)
#     folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]
    split_groups = tr_df['DT_M']

    tt_df = tt_df[['TransactionID',target]]    
    predictions = np.zeros(len(tt_df))
    oof = np.zeros(len(tr_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y) 
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS
        
        oof_preds = estimator.predict(vl_x)
        oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
        
    tt_df['prediction'] = predictions
    print('OOF AUC:', metrics.roc_auc_score(y, oof))
    
    return tt_df
## -------------------

# %% [code]
test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=6)

# %% [code]
test_predictions['isFraud'] = test_predictions['prediction']
test_predictions[['TransactionID','isFraud']].to_csv('pondered_prediction.csv', index=False)