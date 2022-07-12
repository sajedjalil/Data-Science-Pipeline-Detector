import gc
import numpy as np 
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
class BetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta =  beta_prior + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)

        else:
            num = self.prior_mean
            dem = np.ones_like(N_prior)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
        


print('load data')
cat_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'param_1', 'param_2', 'param_3', 'image_top_1']
train = pd.read_csv('../input/train.csv', usecols=cat_cols+['deal_probability'])
test = pd.read_csv('../input/test.csv', usecols=cat_cols)
subm = pd.read_csv('../input/sample_submission.csv')
subm['deal_probability'] = 0

print('fill in missing')
train['image_top_1'] = train['image_top_1'].astype(str)
test['image_top_1'] = test['image_top_1'].astype(str)
train.fillna('', inplace=True)
test.fillna('', inplace=True)

lgb_params = {
    'learning_rate': 0.05,
    'application': 'regression',
    'max_depth': 9,
    'num_leaves': 300,
    'verbosity': -1,
    'metric': 'rmse',
    'data_random_seed': 3,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.4,
    'nthread': 32,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'min_data_in_leaf': 40,
}

n_folds = 5
n_rounds = 10 # increase to 2000

for N_min in [10, 100, 1000, 10000, -1]: 

    print('label encoding')
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(np.concatenate([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        
            
    scores = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for i, (dev_index, val_index) in enumerate(kf.split(train.index.values)):
        
        print(f'Fold {i}:')
        
        # split data
        dev = train.loc[dev_index].reset_index(drop=True)
        val = train.loc[val_index].reset_index(drop=True)
    
        # built-in categorical encoding
        if N_min==-1: 
            
            feature_cols = cat_cols
            
            # setup lightgbm data
            d_dev = lgb.Dataset(dev[cat_cols],
                                label=dev.deal_probability,
                                feature_name=cat_cols,
                                categorical_feature=cat_cols)
            d_val = lgb.Dataset(val[cat_cols],
                                label=val.deal_probability,
                                feature_name=cat_cols,
                                categorical_feature=cat_cols)
    
        # target encoding
        else: 
            
            # encode variables
            feature_cols = []
            for var_name in cat_cols:
        
                # fit encoder
                be = BetaEncoder(var_name)
                be.fit(dev, 'deal_probability')
        
                # mean
                feature_name = f'{var_name}_mean'
                dev[feature_name]  = be.transform(dev,  'mean', N_min)
                val[feature_name]  = be.transform(val,  'mean', N_min)
                test[feature_name] = be.transform(test, 'mean', N_min)        
                feature_cols.append(feature_name)
                
            # setup lightgbm data
            d_dev = lgb.Dataset(dev[feature_cols], label=dev.deal_probability)
            d_val = lgb.Dataset(val[feature_cols], label=val.deal_probability)
        
        # fit model
        mdl = lgb.train(lgb_params,
                          train_set = d_dev,
                          num_boost_round = n_rounds,
                          valid_sets = [d_dev, d_val],
                          verbose_eval = n_rounds//5)
        scores.append(mdl.best_score['valid_1']['rmse'])
        
        # make predictions on test set
        subm['deal_probability'] += mdl.predict(test[feature_cols])/n_folds    
        
    # clean up
    del d_dev, d_val, mdl
    gc.collect()
    
    # print results
    if N_min==-1:
        print(f'baseline: {np.mean(scores):0.2f}')
    else: 
        print(f'N_min={N_min}: {np.mean(scores):0.2f}')
        
    # save data
    subm['deal_probability'] = np.clip(subm['deal_probability'], 0, 1)
    if N_min==-1:
        subm.to_csv(f'submission-baseline.csv', index=False)
    else: 
        subm.to_csv(f'submission-{N_min}.csv', index=False)