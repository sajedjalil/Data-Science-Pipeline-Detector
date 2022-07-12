# coding: utf-8
__author__ = 'AJ Cloete: https://kaggle.com/defenestration'

import datetime
from pandas import HDFStore
import pandas as pd

def read_book_data():
    import gc
    pdread = pd.read_csv('../input/train.csv', chunksize=100000, iterator=True, dtype={'orig_destination_distance':object})
    print('Reading data...')
    t = pd.concat([chunk[chunk.is_booking==1] for chunk in pdread])
    with HDFStore('store.h5', complib='blosc') as s:
        s.append('bookings', t)
    print('Stored booking data')
    gc.collect()

    # pdread = pd.read_csv('../input/train.csv', chunksize=100000, iterator=True, dtype={'orig_destination_distance':object})
    # t = pd.concat([chunk[chunk.is_booking==0] for chunk in pdread])
    # with HDFStore('store.h5', complib='blosc') as s:
    #     s.append('clicks',t)
    # print('Stored clicks data')
    # gc.collect()
    
def read_test_data():
    import gc
    pdread = pd.read_csv('../input/test.csv', chunksize=100000, iterator=True, dtype={'orig_destination_distance':object})
    t = pd.concat([chunk for chunk in pdread])
    with HDFStore('store.h5', complib='blosc') as s:
        s.append('test', t)
    print('Stored test data')
    gc.collect()

def aggregator(feature_list, target, from_store, data_key, score_name):
    import pandas as pd
    from pandas import HDFStore
    print('Collecting "',data_key,'" data from HDFStore(',from_store,')',sep='')
    with HDFStore(from_store) as s:
        data = s[data_key]
    print('Aggregator running on ',feature_list)
    ret = (data.groupby(feature_list)[target].value_counts().reset_index()
           .groupby(feature_list)[target].apply(list).reset_index())
    print('Done with aggregation!')
    return ret.rename(columns={'hotel_cluster':score_name})

def looper(lst_ft_lst):
    print('Tallying up most popular combinations of variables.')
    import gc
    target = 'hotel_cluster'
    from_store = 'store.h5'
    data_key = 'bookings'
    ret = {}
    for i,feature_list in enumerate(lst_ft_lst):
        print('Running loop', i+1,'of', len(lst_ft_lst))
        ret[i] = aggregator(feature_list, target, from_store, data_key, score_name = 'hc'+str(i))
        gc.collect()
    return ret

def fillna_list(df, col):
    '''Fill missing values of column of df with empty list
    Returns last column, filled'''
    df = df.copy()
    print('Filling missing values with empty lists')
    df.loc[df[col].isnull(),[col]] = df.loc[df[col].isnull(),col].apply(lambda x: [])
    return df[col].values

def merge_results(dct):
    '''Merge all the scores from Aggregator to submission file'''
    import gc
    with HDFStore('store.h5') as s:
        submit = s.test
    for i in dct.keys():
        print('Merging set',i+1,'of',len(dct.keys()))
        submit = submit.merge(dct[i], how='left')
        col = submit.columns.tolist()[-1]
        submit[col] = fillna_list(submit.ix[:,[0,-1]],col)
        submit[col] = [x[:5] for x in submit[col]]
        print('Done')
        gc.collect()
    return submit        

def unique_entries(row):
    '''Merge unique top entries'''
    hot_list = row[1]
    for i in range(1,len(row)):
        hot_list.extend([x for x in row[i] if x not in hot_list])
    return ' '.join([str(x) for x in hot_list][:5])

def process_file(submit):
    '''Define order in which scores will be combined.
    Returns processed pandas dataframe'''
    import datetime
    cols = ['id']+[str(x) for x in submit.columns.tolist() if 'hc' in x]
    submit = submit[cols]
    print('Processing scores')
    ret = submit.apply(unique_entries, axis=1).reset_index().rename(columns={'index':'id',0:'hotel_cluster'})
    now = datetime.datetime.now()
    now = str(now.strftime("%Y-%m-%d-%H-%M"))
    print('Writing submission file, timestamp', now)
    ret.to_csv('submission'+now+'.csv', index=False)
    return ret

def run_solution(list_of_feature_lists):
    dct = looper(list_of_feature_lists)
    submission = merge_results(dct)
    process_file(submission)
    #return dct, submission

lst_ft_lst = [['user_location_city','orig_destination_distance'],
              ['srch_destination_id', 'hotel_country'],
              ['hotel_market']]

read_book_data()
read_test_data()

run_solution(lst_ft_lst)


