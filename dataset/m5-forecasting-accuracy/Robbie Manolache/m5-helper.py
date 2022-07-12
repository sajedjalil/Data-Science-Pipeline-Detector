# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Helper Script for M5 feature generation \--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\-
# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-


import os
import string
import numpy as np 
import pandas as pd 


# IMPORT AND PRE-PROC -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def load_data(file_codes=['sale', 'cale', 'sell', 'samp'], stage="evaluation"):
    """
    Loads the all the data files from the /kaggle/input directory
    """

    files = {}
    if stage == "evaluation":
        ignore = "sales_train_validation.csv"
    else:
        ignore = "sales_train_evaluation.csv"
    
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename != ignore:
                files[filename[:4]] = os.path.join(dirname, filename)   
 
    return [pd.read_csv(files[f]) for f in file_codes]


def create_id_col(prc_df, stage="evaluation"):
    """
    Creates `id` column in the price data using item_id and store_id.
    This is so that it can be merged with training data.
    
    (NOTE: '_validation' suffix may have to be changed at evaluation stage)
    """   

    prc_df.loc[:,'id'] = prc_df['item_id'] + '_' + prc_df['store_id'] + '_' + stage
    return prc_df.drop(['store_id', 'item_id'],1).sort_values(by=['id', 'wm_yr_wk'])


# STANDARD GROUP ID GENERATION -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def add_grp_id(df, grp_names):
    """
    Uses the product 'id' to generate group-level IDs to aggregate over. 
    """
    
    if type(grp_names) != list:
        grp_names = [grp_names]
    else:
        pass
    
    grp_dict = {
        'item': [0, 1, 2],
        'dept': [0, 1],
        'cat': [0],
        'store': [3, 4],
        'state': [3]
    }
    
    idx = []
    id_key = []
    for name in grp_names:
        idx += grp_dict[name]
        id_key.append(name)
        
    id_key = '_'.join(id_key) + '_id'
    
    tmp = df.copy()[['id']].drop_duplicates()
    tmp.loc[:, id_key] = tmp['id'].apply(lambda x: '_'.join(
        [s for i, s in enumerate(x.split('_')) if i in idx]))
    
    return df.merge(tmp, on='id')


# PRICE CATEGORIES -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def make_cut(x, nmin, ncuts):
    """
    Splits according to price only if nmin is respected.
    """
    c = string.ascii_uppercase
    same = pd.Series([c[ncuts]]*x.shape[0]).reindex(x.index).fillna(c[ncuts])
    if x.shape[0] > (nmin*ncuts):
        split = pd.cut(x, ncuts, labels=[s for s in c[:ncuts]])
        min_split = min(split.value_counts())
        if min_split > nmin:
            return split
        else:
            return same
    else:
        return same             

def add_prc_cat(df, nmin, ncuts):
    """
    Adds the price categories within each dept_id based on the average sell price.
    """
    df = df.join(df.groupby('dept_id')['sell_price'].apply(
            lambda x: make_cut(x, nmin, ncuts)).rename('prc_cat'))
    df.loc[:, 'dept_id'] = df['dept_id'] + df['prc_cat'].astype(str)
    return df.drop('prc_cat', 1)

def create_prc_grp_id(prc_grps, nmin=15, ncuts=2, verbose=True):
    """
    Creates unique ID's for price group within each department. 
    The algorithm iteratively splits the items in each department along the median, until it
    can no longer conduct a split where the price groups have more than nmin items each.
    """
    df = prc_grps.copy()
    ldiff = 1
    l0 = 1
    while ldiff > 0:
        df = add_prc_cat(df, nmin, ncuts)
        counts = df.groupby(['dept_id'])['id'].count()
        ldiff = len(counts) - l0
        l0 = len(counts)
        if verbose:
            print("Min: %d, Max: %d, Num: %d"%(min(counts), max(counts), len(counts)))
    return df.rename(columns={'dept_id':'prc_grp_id'}).drop('sell_price', 1)


# PRICE FEATURES -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def gen_price_features(prc_df, lags=1):
    """
    Creates index-like and per cent change features that can be compared across items, 
    and serve as direct features at the single item level.
    """
    
    # Median-based relative price index
    med_df = prc_df.groupby('id')['sell_price'].median().rename('prc_med').reset_index()
    med_df = prc_df.merge(med_df, on=['id'])
    med_df.loc[:, 'prc_idx'] = med_df['sell_price'] / med_df['prc_med']
    med_df = med_df.drop(['sell_price', 'prc_med'], 1)
    
    # Price changes (and lags)
    chg_df = prc_df.groupby('id')['sell_price'].pct_change().rename('pct_chg_0')
    chg_df = np.log(chg_df + 1)
    chg_df = prc_df.drop('sell_price', 1).join(chg_df)
    new_prod = chg_df['wm_yr_wk'] > min(prc_df['wm_yr_wk']) # treat new products differently
    chg_df.loc[new_prod, 'pct_chg_0'] = chg_df.loc[new_prod, 'pct_chg_0'].fillna(0)
    lags = 1
    for lag in range(1, lags+1):
        chg_df.loc[:, 'pct_chg_'+str(lag)] = chg_df.groupby('id')['pct_chg_'+str(lag-1)
                                                                 ].transform('shift')
        new_prod = chg_df['wm_yr_wk'] > min(prc_df['wm_yr_wk']+lag) 
        chg_df.loc[new_prod, 
                   'pct_chg_'+str(lag)] = chg_df.loc[new_prod, 
                                                     'pct_chg_'+str(lag)].fillna(0)
    chg_df = chg_df.dropna()
    
    return med_df.merge(chg_df, on=['id', 'wm_yr_wk'])


# GROUP FEATURE AGGREGATION -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def gen_agg_prices(grp_list, prc_ft_df, prc_grps):
    """
    Generates aggregate price features for various groupings.
    """
    
    df_list = []
    
    for grp in grp_list:
        if grp == "prc_grp":
            df = prc_ft_df.merge(prc_grps, on=['id'])
            grp_id = grp + '_id'
        else:
            df = add_grp_id(prc_ft_df, grp)
            grp_id = '_'.join(grp)+'_id'
        df = df.groupby(['wm_yr_wk',grp_id])[['prc_idx','pct_chg_0',
                                              'pct_chg_1']].mean().reset_index()
        df_list.append(df.rename(columns={grp_id:'grp_id'}))
        
    return pd.concat(df_list, ignore_index=True)

def gen_agg_sales(train_df, grp_vars, value_vars, pairs, prc_grps):
    """
    Aggregates sales data at various levels which can then be used
    as features for individual item models.    
    """
    
    grp_sales = []
    
    for v in grp_vars:
        df = train_df.groupby(v)[value_vars].sum().reset_index()
        grp_sales.append(df.rename(columns={v:'grp_id'}))

    for p in pairs:
        v = [s + '_id' for s in p]
        df = train_df.groupby(v)[value_vars].sum().reset_index()
        df[v[0]] = df[v[0]] + '_' + df[v[1]]
        df = df.drop(v[1],1).rename(columns={v[0]: 'grp_id'})
        grp_sales.append(df)
        
    df = train_df[value_vars+['id']].merge(prc_grps, on=['id'])
    df = df.groupby('prc_grp_id')[value_vars].sum().reset_index()
    grp_sales.append(df.rename(columns={'prc_grp_id':'grp_id'})) 

    return pd.concat(grp_sales, ignore_index=True)


# ADD FEATURES AT ITEM-LEVEL -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

def get_item_features(item_df, grp_df, grp_vars, pairs):
    """
    """
    
    df = item_df.copy()[grp_vars].drop_duplicates()
    
    for p in pairs:
        v = [s + '_id' for s in p]
        new = '_'.join(p) + '_id'
        df[new] = df[v[0]] + '_' + df[v[1]]
        
    grp_val = df.values[0]
    
    return grp_df[grp_df['GRP'].isin(grp_val)]