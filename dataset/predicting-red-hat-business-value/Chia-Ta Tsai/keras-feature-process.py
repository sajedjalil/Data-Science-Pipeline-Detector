__name__ = '__main__'

import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold, LabelKFold
import pickle
import gzip

seed = 101472016
np.random.seed(seed)

#keras
dim = int(8 * 4)
hidden1 = int(4 * 8)
hidden2 = int(4 * 4)

default_batch_size = 2 ** 10 * 4
L2_reg = 10 ** -3

path = "../input/"
digit = 6
hash_digit = 2 ** 10 * 16


############feature
def date_process(d, f):
    print('parse date info')

    #d['mm'] = d[f].dt.month
    d['dd'] = d[f].dt.day    
    d['weekday'] = d[f].dt.weekday
    
    #hash_digit = 10 ** 3
    d['yyyy-mm'] = d[f].apply(lambda x: 100 * x.year + x.month)
    
    d['day_no'] = d[f].apply(lambda x: x.day).astype(int)
    d['week_no'] = d[f].apply(lambda x: x.week).astype(int)
           
    d.drop(f, axis=1, inplace=True)
    return d


def act_process(d):
    print('process act')
    #d['null_count'] = d.isnull().sum(axis=1)
    
    d['act_id'] = d['activity_id'].str[3:4]
    d['activity_category'].str.lstrip('type ').fillna('na', inplace = True)
    
    d['act_id_cate'] = d['act_id'] + ':' + d['activity_category']
    d['act_id_cate'] = d['act_id_cate'].apply(lambda x: hash(x) % hash_digit)

    print('hash act concat')
    str_name = 'concat_act_char'
    d['char_10'].str.lstrip('type ').fillna('na', inplace = True)
    d[str_name] = d['char_10'].values
    d.rename(columns={'char_10':'act_char_10'}, inplace=True)
    
    for i in range(1, 10, 1):
        str1 = 'char_' + str(i)
        d[str1].str.lstrip('type ').fillna('na', inplace = True)
        d[str_name] += ':' + d[str1]
        d.drop(str1, axis=1, inplace=True)
        
    d[str_name] = d[str_name].apply(lambda x: hash(x) % hash_digit)
    
    print('after act process: {}'.format(d.columns.tolist()))
    return d


def ppl_concate_process(d):
    print('concat ppl bool concat')
    str_name = 'concat_ppl_bool'
    d[str_name] = ''
    for i in range(10, 38, 1):
        str1 = 'char_' + str(i)
        if str1 in d.columns:
            d[str_name] = d[str_name] + ':' + d[str1].astype(str)
            #d.rename(columns={str1:'ppl_b_' + str1}, inplace=True)
            d.drop(str1, axis=1, inplace=True)
    
    print('hash concated bools')
    d[str_name] = d[str_name].apply(lambda x: hash(x) % hash_digit)
    return d    
    

def concate_process(d, str1, str2):
    f_name = str1 + '_' + str2
    if str1 in d.columns and str2 in d.columns:
        print('add concate, {}, adopted by {} + {}'.format(f_name, str1, str2))
        d[f_name] = d[str1].astype(np.str) + '_' + d[str2].astype(np.str)
        d[f_name] = d[f_name].apply(lambda x: hash(x) % hash_digit)
    
    return d, f_name


##########################################################
def mask_columns(columns = {}, mask = {}):
    for str1 in mask:
        #if columns.count(str1) > 0:
        if str1 in columns:
            columns.remove(str1)
    return columns


def label_encode(data, mask = {}):
    columns = mask_columns(data.columns.tolist(), mask)
    #encode
    for c in columns:
        data[c] = LabelEncoder().fit_transform(data[c].values)
    return data
    

def split_train_valid(X, y, n_folds=5, shuffle=True, random_state=seed):#return np.array

    skf = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    for ind_tr, ind_va in skf:
        y_train, y_valid = y[ind_tr], y[ind_va]
        
        X_train = np.array(X[ind_tr])
        X_valid = np.array(X[ind_va])       
        #X_train = X[ind_tr]
        #X_valid = X[ind_va]
        break
    
    #splice array into samples
    #X_train = [X_train[:,i] for i in range(X.shape[1])]
    #X_valid = [X_valid[:,i] for i in range(X.shape[1])]
    
    print('StratifiedKFold: Length of train and valid: ', X_train.shape[0], ', ', X_valid.shape[0])
    return X_train, y_train, X_valid, y_valid


def split_train_valid_by_label(X, y, label, n_folds=3):#return np.array

    lkf = LabelKFold(label, n_folds=n_folds)
    for ind_tr, ind_va in lkf:
        y_train, y_valid = y[ind_tr], y[ind_va]
        
        X_train = np.array(X[ind_tr])
        X_valid = np.array(X[ind_va])       
        #X_train = X[ind_tr]
        #X_valid = X[ind_va]
        break
    
    #splice array into samples #list-alike 
    #X_train = [X_train[:,i] for i in range(X.shape[1])]
    #X_valid = [X_valid[:,i] for i in range(X.shape[1])]

    print('LabelkFold: Length of train and valid: ', X_train.shape[0], ', ', X_valid.shape[0])
    return X_train, y_train, X_valid, y_valid


def create_submission(activity_id, outcome, prefix, score, digit):
    now = dt.datetime.now()
    filename = 'submission_residual_' + prefix + '_d' + str(dim) + '_a' + str(round(score, digit)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv.gz'
    print('Make submission:{}'.format(filename))    
    
    submission = pd.DataFrame()
    submission['activity_id'] = activity_id
    submission['outcome'] = np.round(outcome, decimals=digit)
    #submission['outcome'] = submission['outcome'].apply(lambda x: round(x, digit))
    submission.to_csv(filename, index=False, compression='gzip')


def main():
    #
    start_time = time.time()

    print('incorp w/ ppl')
    people = pd.read_csv(path+'people.csv', parse_dates=['date'])
    people = people.fillna('null', axis='columns')
    people = date_process(people, 'date')
    people = ppl_concate_process(people)
    people, ppl_g1_d_t = concate_process(people, 'group_1', 'day_no')
    people.rename(columns={'day_no':'ppl_day_no', ppl_g1_d_t:'ppl_' + ppl_g1_d_t}, inplace=True)
    ppl_g1_d_t = 'ppl_' + ppl_g1_d_t    
    people['char_38c'] = people['char_38'].values
    people = label_encode(people, mask = {'people_id', 'char_38'})
    #people = people[['people_id', 'char_38']]
    print('extract {} from ppl: {}\n'.format(len(people.columns), people.columns.tolist()))

    
    print('read in data')
    train = pd.read_csv(path+'act_train.csv', parse_dates=['date'])
    train = date_process(train, 'date')
    train = act_process(train)

    test = pd.read_csv(path+'act_test.csv', parse_dates=['date'])
    test = test.assign(outcome=np.nan)
    test = date_process(test, 'date')
    test = act_process(test)
    
    
    #merge w/ ppl
    train = pd.merge(train, people, how='left', on='people_id').fillna('null', axis='columns')
    X_ppl = train['people_id'].values
    train.drop('people_id', axis=1, inplace=True)
    
    test = pd.merge(test, people, how='left', on='people_id').fillna('null', axis='columns')
    test.drop('people_id', axis=1, inplace=True)
    
    del people

    
    #concate and encode
    print('encode data')
    mask = {'activity_id', 'outcome', 'char_38', 'people_id'}
    data = pd.concat([train, test])
    data, act_g1_d_t = concate_process(data, 'group_1', 'day_no')#add group1 trick my understanding
    data = label_encode(data, mask)
    train = data[:train.shape[0]]
    test = data[train.shape[0]:]

    
    print('Load data: {} minutes'.format(round((time.time() - start_time)/60, 2)))    
    
    #remove target and id by remove its column name
    mask = {'activity_id', 'outcome'}   
    columns = mask_columns(train.columns.tolist(), mask)
    print('Apply {} features: '.format(len(columns)))
    for i, j in enumerate(columns, start=1):
        print(i, ': ', j)        
    
    #y = train['outcome'].values
    #X = train[columns].values
    #train_activity_id = train['activity_id']
    #del train
    
    #X_t = test[columns].values
    #test_activity_id = test['activity_id']
    #del test


    #inner fold
    #with open('train.pkl', 'wb') as f:
        #pickle.dump((X_train, y_train), f, protocol=-1)
    with gzip.open('data.gz', 'wb') as f:
        pickle.dump((train, X_ppl, test), f, protocol=-1)
        #pickle.dump((X_train, y_train, X_valid, y_valid), f, protocol=-1)  
        f.close()
    
    #return
                
main()