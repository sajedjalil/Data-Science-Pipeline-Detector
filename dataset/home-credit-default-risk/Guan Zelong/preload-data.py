import pandas as pd
import numpy as np
import gc

# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
        
        numeric types are converted to the smallest type feasible;
        others converted to categorical type. 
        Values are preserved, thus general enough
    """

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                # suffer precision loss
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else: df[col] = df[col].astype('category')

    return df


# ugly
from functools import lru_cache
@lru_cache(maxsize=1)
def app_cat():
    '''return index object'''
    df = pd.read_csv(f'../input/home-credit-default-risk/application_train.csv')
    col_nuniq = df.nunique()
    candidate_col = col_nuniq[col_nuniq < 7].index
    cols = df.columns
    # NOT touch
    candidate_col = candidate_col.difference(cols[cols.str.contains(r"^AMT_")]) # amount
    candidate_col = candidate_col.difference(cols[cols.str.contains(r"(_AVG|_MODE|_MEDI)$")]) # normalized data
    return candidate_col


def replace_day_outliers(df):
    """Replace 365243 with np.nan in any columns with DAYS"""
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})

    return df


def dataset_specific(df, df_name):
    '''dataset-specific assertions
    I inspected each schema using `df.info()` and correct their result
    '''
    # Categorical
    dict_cat = {
    "application_train": app_cat(),
    "application_test": app_cat().drop(['TARGET']),
    "previous_application": ['NFLAG_LAST_APPL_IN_DAY','NFLAG_INSURED_ON_APPROVAL'],
    }
    if df_name in dict_cat:
        df[dict_cat[df_name]] = df[dict_cat[df_name]].astype('category')
    
    # Time Variables
    df = replace_day_outliers(df) # Reported na repr
    # start_date = pd.Timestamp("2019-12-31") # assumed, to use ft utility
    
    # cols = df.columns
    # d_cols = cols[cols.str.contains(r"^DAYS_")]
    # m_cols = cols[cols.str.contains(r"^MONTHS_")]
    # df[d_cols] = df[d_cols].apply(lambda s: start_date + pd.to_timedelta(s, unit='D'))
    # df[m_cols] = df[m_cols].apply(lambda s: start_date + pd.to_timedelta(s, unit='M'))

    return df


'''For:
    application_train, application_test, bureau, previous_application: 
        idxcol has unique val
    bureau_balance, POS_CASH_balance, installments_payments, credit_card_balance: 
        idxcol val non-unique. balance about previous loan, collected over time
'''
schema_idxcol = {
            'application_train':None,'application_test':None,
            'bureau':None,'previous_application':None,
            #'application_train':'SK_ID_CURR','application_test':'SK_ID_CURR',
            #'bureau':'SK_ID_BUREAU','previous_application':'SK_ID_PREV',
            'bureau_balance':None,'POS_CASH_balance':None,
            'installments_payments':None,'credit_card_balance':None,
            #'bureau_balance':'SK_ID_BUREAU','POS_CASH_balance':'SK_ID_PREV',
            #'installments_payments':'SK_ID_PREV','credit_card_balance':'SK_ID_PREV'
            } 

# increase the overall workflow speed by loading data from optimized .pkl
for schema, index_col in schema_idxcol.items():
    print(f'{schema:-^60}')
    df = pd.read_csv(f'../input/home-credit-default-risk/{schema}.csv',index_col=index_col)

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    df = reduce_mem_usage(df)
    df = dataset_specific(df, schema)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    df.to_pickle(f'{schema}.pkl')
    del df
    gc.collect()


