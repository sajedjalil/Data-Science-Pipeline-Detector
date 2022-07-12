import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

NUM_FOLDS_OOF = 4
BOOST_ROUNDS = 620
LGBM_LR = 0.045
MAX_FEATURES = 500000

print('ridge,sgd', NUM_FOLDS_OOF, 'oof folds',BOOST_ROUNDS, ' rounds lightgbm with learning rate ',LGBM_LR)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from memory_profiler import memory_usage
import sys
import hashlib
from nltk.corpus import stopwords
import re
from multiprocessing import Pool
import gc
import random
import string
import psutil
import scipy
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from itertools import compress
import time
import math
import warnings
warnings.filterwarnings("ignore")

process = psutil.Process(os.getpid())

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
# Define Stopwords set
cachedStopWords = set(stopwords.words("english"))

brands = set(['apple', 'michael kors', 'louis vuitton', 'pink', 'kate spade',
          'nintendo', 'rae dunn', 'lularoe', 'nike', 'sony'])
          
models = {}
transformers = {}

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5

def remove_stop_words(s):
    return ' '.join([word for word in s.lower().split(' ') if word not in cachedStopWords])

def fill_missing(dataset):
    dataset['name'].fillna('missing', inplace=True)
    dataset['category_name'].fillna('missing', inplace=True)
    dataset['brand_name'].fillna('missing', inplace=True)
    dataset['item_description'].fillna('missing', inplace=True)
    dataset['shipping'] = dataset.shipping.fillna(value=0)
    dataset.loc[~dataset.shipping.isin([0,1]), 'shipping'] = 0
    dataset['shipping'] = dataset.shipping.astype(int)
    dataset['item_condition_id'] = dataset.item_condition_id.fillna(value=1)
    dataset.loc[~dataset.item_condition_id.isin([1,2,3,4,5]), 'item_condition_id'] = 1
    dataset['item_condition_id'] = dataset.item_condition_id.astype(int)
    return dataset
    
def split_cat1(s):
    try: return s.split('/')[0]
    except: return 'cat_missing'

def split_cat2(s):
    try: return s.split('/')[1]
    except: return 'cat_missing'

def split_cat3(s):
    try: return ' '.join(s.split('/')[2:])
    except: return 'cat_missing'
    
def get_categories(dataset):
    dataset['category_1'] = dataset.category_name.map(split_cat1)
    dataset['category_2'] = dataset.category_name.map(split_cat2)
    dataset['category_3'] = dataset.category_name.map(split_cat2)
    return dataset
    
def feature_hash(s, n_bins=10000):
    hf = hashlib.md5(s.lower().encode('utf8')).hexdigest()
    return int(hf, 16) % (n_bins - 1) + 1
    
def label_encoder(s, n_bins=1000000):
    hf = hashlib.md5(s.lower().encode('utf8')).hexdigest()
    return int(hf, 16) % (n_bins - 1) + 1
    
def find_ngrams(s, n):
    input_list = s.lower().split(' ')
    return [i for i in zip(*[input_list[i:] for i in range(n)])]
    
def find_ngrams_pre(s, n=2, mode='concat'):
    input_list = s.lower().split(' ')
    if mode=='list':
        return [i for i in zip(*[input_list[i:] for i in range(n)])]
    elif mode=='concat':
        return ' '.join(['_'.join(i) for i in zip(*[input_list[i:] for i in range(n)])])        

def get_first_bigram(s):
    try:
        return ' '.join(find_ngrams(s, 2)[0])
    except:
        return 'None'

def get_last_bigram(s):
    try:
        return ' '.join(find_ngrams(s, 2)[-1])
    except:
        return 'None'

def get_first_trigram(s):
    try:
        return ' '.join(find_ngrams(s, 3)[0])
    except:
        return 'None'

def get_last_trigram(s):
    try:
        return ' '.join(find_ngrams(s, 3)[-1])
    except:
        return 'None'

def get_last_word(x):
    return [i.lower() for i in x.split(' ')][-1]
    
def get_first_n_words(s, n=4):
    return s.lower().split(' ')[:n]
        
def word_count(x):
    try:
        return len(x.split(' '))
    except:
        return 0.

def sentence_std(x):
    try:
        return np.sqrt(np.var([len(i) for i in x.split(' ')]))
    except:
        return 0.

def hasNumbers(inputString):
    return int(bool(re.search(r'\d', inputString)))
    
def apply_map_fn(x, fn):
    pool = Pool(4)
    results = pool.map(fn, x)
    pool.close()
    pool.join()
    return results
    
def brand_new(s):
    return int('brand new' in s.lower())

def free_shipping(s):
    return int('free shipping' in s.lower())
    
def check_membership(x, s):
    if x.lower() in s:
        return 1
    else:
        return 0
        
def label_encoder(X):
    print('Get values')
    labels = X.values.tolist()

    print('Create labels')
    labels_list = set(labels)
    idx = range(len(labels_list))

    print('label map')
    label_map = dict(zip(labels_list, idx))

    return X.apply(lambda x: label_map[x])
    
def cooc_term_generator(l1, l2):
    for i in range(len(l1)):
        yield (l1[i], l2[i])    

def get_cooc_terms(lst1, lst2, join_str='X'):
    return ' '.join([a+join_str+b for a in lst1 for b in lst2])            

def default_cooc_transformer(df_all, tf_name, feature = 'cooc_terms', ngram_range = (1,1)):
    if tf_name not in transformers:
        transformers[tf_name] = TfidfVectorizer(
                                   ngram_range=ngram_range,
                                   max_features=MAX_FEATURES,
                                   norm='l2',
                                   strip_accents="unicode",
                                   analyzer="word",
                                   token_pattern=r"[0-9a-z_\$\+]{1,}",
                                   use_idf=1,
                                   smooth_idf=1,
                                   sublinear_tf=1
                                )
        transformers[tf_name].fit(df_all[feature])
    return transformers[tf_name].transform(df_all[feature]).astype(np.float16)

def processing_dataset(df_all):
    df_all['category_name'] = df_all.category_name.fillna('missing')
    df_all['item_description'].fillna('', inplace=True)
    df_all['shipping'] = df_all.shipping.fillna(value=0)
    df_all.loc[~df_all.shipping.isin([0,1]), 'shipping'] = 0
    df_all['shipping'] = df_all.shipping.astype(int)
    df_all['item_condition_id'] = df_all.item_condition_id.fillna(value=1)
    df_all.loc[~df_all.item_condition_id.isin([1,2,3,4,5]), 'item_condition_id'] = 1
    df_all['item_condition_id'] = df_all.item_condition_id.astype(int)
    df_all['name'] = df_all['name'].fillna('')
    df_all['brand_name'] = df_all['brand_name'].fillna('brand_missing')
    df_all['desc_len'] = df_all.item_description.str.count('\S+')
    df_all['name_len'] = df_all.name.str.count('\S+')

    get_categories(df_all)    

    encoded_cols = ['desc_len','name_len']
    tf_name = 'scale_cv_ridge'    
    if tf_name not in transformers:
        transformers[tf_name] = StandardScaler()
        transformers[tf_name].fit(df_all[encoded_cols])
    X_encoded = transformers[tf_name].transform(df_all[encoded_cols]).astype(np.float16)
    df_all.drop(encoded_cols, axis=1, inplace=True)
    gc.collect()

    with Pool(4) as pool:
        rmresults = pool.map(remove_stop_words, df_all.item_description.tolist())
    df_all['item_description_clean'] = rmresults
    del rmresults
    gc.collect()   
    df_all['item_description_2'] = df_all['item_description'] + ' ' + df_all['name'] + ' ' + df_all['brand_name']        
    df_all['item_description_3'] = df_all['item_description_clean'] + ' ' + df_all['name']
    with Pool(4) as pool:
        bigrams = pool.map(find_ngrams_pre, df_all.item_description_clean.tolist())
    
    print('transforming name')
    tf_name = 'name_cv_ridge'
    if tf_name not in transformers:
        transformers[tf_name] = CountVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)
        transformers[tf_name].fit(df_all['name'])
    X_name = transformers[tf_name].transform(df_all['name']).astype(np.float16)

    print('transforming category')
    tf_name = 'category_cv_ridge'
    df_all['category_name'] = df_all['category_name'].str.replace('/',' ')
    if tf_name not in transformers:
        transformers[tf_name] = TfidfVectorizer(dtype=np.int16)
        transformers[tf_name].fit(df_all['category_name'])
    X_category_name = transformers[tf_name].transform(df_all['category_name']).astype(np.float16)

    print('brand name')
    tf_name = 'brand_cv_ridge'
    if tf_name not in transformers:
        transformers[tf_name] = CountVectorizer(dtype=np.int16)
        transformers[tf_name].fit(df_all['name'])
    X_brand_name = transformers[tf_name].transform(df_all['brand_name']).astype(np.float16)

    print("shipping & item_condition_id...")
    X_shipping_and_item_condition_id = scipy.sparse.csr_matrix(pd.get_dummies(df_all[['item_condition_id', 'shipping']], sparse=True)).astype(np.float16)
    
    print("item_description ...")
    X_item_description = default_cooc_transformer(df_all, 'desc_cv_ridge', 'item_description_2', (1,2))

    print('Start Cooccurence terms')
    df_all['category_3'] = df_all['category_3'] + ' ' + df_all['brand_name']
    df_all['cooc_terms'] = list(map(lambda lst1,lst2: get_cooc_terms(lst1, lst2), df_all.category_3.str.split(' ').values, df_all.item_description_3.str.split(' ').values))
    X_cooc_terms = default_cooc_transformer(df_all, 'cooc_terms1_ridge')

    print('Start Cooccurence terms 2')
    df_all['cooc_terms'] = list(map(lambda lst1,lst2: get_cooc_terms(lst1, lst2), df_all.category_name.str.split(' ').values, df_all.name.str.split(' ').values))
    X_cooc_terms_2 = default_cooc_transformer(df_all, 'cooc_terms2_ridge')

    print('Start Cooccurence terms 3')
    df_all['cooc_terms'] = list(map(lambda lst1,lst2: get_cooc_terms(lst1, lst2), df_all.category_name.astype(str).str.split(' ').values, df_all.brand_name.str.split(' ').values))
    X_cooc_terms_3 = default_cooc_transformer(df_all, 'cooc_terms3_ridge')

    print('Start Cooccurence terms 4')
    df_all['cooc_terms'] = list(map(lambda lst1,lst2: get_cooc_terms(lst1, lst2), df_all.item_condition_id.astype(str).str.split(' ').values, df_all.category_name.str.split(' ').values))
    X_cooc_terms_4 = default_cooc_transformer(df_all, 'cooc_terms4_ridge')

    print('Start Cooccurence terms 5')
    df_all['cooc_terms'] = list(map(lambda lst1,lst2: get_cooc_terms(lst1, lst2), df_all.item_condition_id.astype(str).str.split(' ').values, df_all.brand_name.str.split(' ').values))
    X_cooc_terms_5 = default_cooc_transformer(df_all, 'cooc_terms5_ridge')

    print('Start Cooccurence terms 6')
    df_all['cooc_terms'] = list(map(lambda lst1,lst2: get_cooc_terms(lst1, lst2), df_all.shipping.astype(str).str.split(' ').values, df_all.category_name.str.split(' ').values))
    X_cooc_terms_6 = default_cooc_transformer(df_all, 'cooc_terms6_ridge')

    df_all.drop(['cooc_terms'], axis=1, inplace=True)
    gc.collect()

    print("hstack all columns")
    X = scipy.sparse.hstack([
                             X_name, 
                             X_brand_name, 
                             X_shipping_and_item_condition_id, 
                             X_category_name, 
                             X_item_description, 
                             X_encoded,
                             X_cooc_terms, 
                             X_cooc_terms_2,
                             X_cooc_terms_3, 
                             X_cooc_terms_4,
                             X_cooc_terms_5, 
                             X_cooc_terms_6
                             ]).tocsr()
    del X_name
    del X_brand_name, X_shipping_and_item_condition_id, X_item_description
    del X_category_name, 
    del X_encoded
    del X_cooc_terms, X_cooc_terms_2, X_cooc_terms_3
    del X_cooc_terms_4, X_cooc_terms_5, X_cooc_terms_6
    gc.collect()

    return X

def make_oof_predictions(X,y_train):
    from sklearn.metrics import mean_squared_error

    num_folds = NUM_FOLDS_OOF
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    oof_prediction = np.zeros((X.shape[0], 1))
    oof_prediction_sgd = np.zeros((X.shape[0], 1))

    ids = range(X.shape[0])
    print('Start Modeling')
    ifold = 0
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        ytrain, y_valid = y_train[train_index], y_train[test_index]
        X_train, X_valid = X[train_index], X[test_index]
        X_train = X_train[np.where(ytrain > 0)]
        ytrain = ytrain.values[ytrain.values > 0]
        
        rmodel = Ridge(alpha=2.0, copy_X=True, fit_intercept=True,
                      max_iter=15.0, normalize=False, random_state=101,
                      solver='sag', tol=0.03)
        print("Fitting Model - Ridge")
        rmodel.fit(X_train, ytrain)
        preds = rmodel.predict(X_valid)
        print('Ridge RMSLE = ', np.sqrt(mean_squared_error(y_valid, preds)))
        oof_prediction[test_index, 0] = np.expm1(preds).clip(3,1e9)
        models['ridge_'+str(ifold)] = rmodel
        
        smodel = SGDRegressor(eta0=0.2, fit_intercept=True, power_t=0.229, alpha=0.0,
                             random_state=1010)
        print("Fitting Model - SGD")
        smodel.fit(X_train, ytrain)
        preds = smodel.predict(X_valid)
        print('SGD RMSLE = ', np.sqrt(mean_squared_error(y_valid, preds)))       
        oof_prediction_sgd[test_index, 0] = np.expm1(preds).clip(3,1e9)
        models['sgd_'+str(ifold)] = smodel

        del X_train, X_valid, ytrain, y_valid
        gc.collect()
        ifold = ifold + 1
    del X
    gc.collect()

    return oof_prediction, oof_prediction_sgd

# Value Counts
def value_counts(X):   
    counts = X.value_counts().to_dict()
    return X.apply(lambda x: counts[x])
    
def dataset_counts(dataset):
    cols = ['category_1', 'category_2', 'category_3',
            'category_1_brand', 'category_2_brand','category_3_brand',
            'brand_name', 'category_name',
            'name', 'item_description', 'name_first_word', 
            'name_last_word', 'first_bigram_name', 'last_bigram_name', 
            'first_bigram_desc', 'last_bigram_desc',
            'first_trigram_name', 'hashed_desc_le', 'hashed_name_le',
            'cond_category_3',
            'last_trigram_name','first_trigram_desc', 'last_trigram_desc', 
            'name_second_word', 'name_third_word',
            'name_fourth_word']
            
    feature = 0 
    pool = Pool(4)
    for x in pool.imap_unordered(value_counts, [dataset[i] for i in cols]):
        dataset['count_feature_{}'.format(feature)] = x
        dataset['count_feature_{}'.format(feature)] = dataset['count_feature_{}'.format(feature)].astype(np.int32)
        feature += 1
    pool.close()
    pool.join()
        
    return dataset
    
               
def get_mapping(x, map):
    try:
        return map[x]
    except:
        return -1.0
        
def target_encode_parallel(X):

    X_train = pd.DataFrame(X[0])
    labels = X[1]
    X_test = pd.DataFrame(X[2])
    col = X[3]

    print(col)

    kf = KFold(n_splits=5, shuffle=True, random_state=0) # Changed to 4 folds
    X_train['price'] = labels
    mean = X_train.groupby(col).price.mean().to_dict()

    X_test[col + '_encode'] = X_test[col].apply(lambda x: get_mapping(x, mean))

    oof_prediction = np.zeros((X_train.shape[0], 1))

    for idx, (train_index, test_index) in enumerate(kf.split(X_train)):
        print('Fold ', idx)
        Xtrain, Xtest = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
        oof_mean = Xtrain.groupby(col).price.mean().to_dict()
        oof_prediction[test_index, 0] = Xtest[col].apply(lambda x: get_mapping(x, oof_mean))

    X_train[col + '_encode'] = oof_prediction[:, 0]
    return (X_train[col + '_encode'].values, X_test[col + '_encode'].values)
    
def lightgbm_model(train, y, test, validate=True):
    
    lgb_params = {
              'max_depth': 11,
              'learning_rate': LGBM_LR,
              'objective': 'regression',
              'metric': 'rmse',
              'bagging_fraction': 0.90, # Best one so far - [884]       valid_0's rmse: 0.434035 - 0.89
              'colsample_bytree': 0.90,
              'num_leaves':  2 ** 11,  # 8192
              'num_threads': 8,
              'min_child_weight': 5,
              'bagging_seed': 0,
              'feature_fraction_seed': 0,
             }
             
    print('gathering validation data')
    if validate:
        
        X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=2017, test_size=0.01)
        del train, y
        gc.collect()
        print('complete')
     
    # Remove Zero Prices 
    idx = (y_train > np.log1p(0))
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    print('lgb data sets')
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)
    print('complete')
    del X_train, y_train, X_test, y_test
    gc.collect()
    
    model = lgb.train( lgb_params
                       , dtrain
                       , num_boost_round=BOOST_ROUNDS
                       , valid_sets=dtest, early_stopping_rounds=100
                       , verbose_eval=10)
                       
    print('model size', sys.getsizeof(model) / 2 ** 30)
                       
    test_prediction = np.array([])
    for data in np.array_split(test, 3):
        test_prediction = np.hstack([test_prediction, np.expm1(model.predict(data, num_iteration=model.best_iteration)).ravel()])
    #test_prediction = model.predict(test, num_iteration=model.best_iteration)
    submission = pd.DataFrame()
    submission['test_id'] = np.array(list(range(len(test))))
    test_prediction = test_prediction.clip(3,1e9) 
    submission['price'] = test_prediction
    #'ridge,sgd', NUM_FOLDS_OOF, 'oof folds',BOOST_ROUNDS, ' rounds lightgbm with learning rate ',LGBM_LR
    sub_file = 'folds_'+str(NUM_FOLDS_OOF)+'_lr_'+str(LGBM_LR) +'_iters_'+str(BOOST_ROUNDS)+'.csv'
    submission.to_csv(sub_file,index=False)
    
    return 1

def data_transforms(dataset):
    
    # Condition - Category 3 Variable
    dataset['cond_category_3'] = dataset['category_3'] + dataset['item_condition_id'].astype(str)
    # dataset['cond_category_3'] = dataset['cond_category_3'].astype('category')
    
    # Hashed
    dataset['hashed_name_le'] = apply_map_fn(dataset.name.tolist(), feature_hash)
    dataset['hashed_desc_le'] = apply_map_fn(dataset.item_description.tolist(), feature_hash)
    
    # Bigrams
    dataset['first_bigram_name'] = apply_map_fn(dataset.name.tolist(), get_first_bigram)
    dataset['last_bigram_name'] = apply_map_fn(dataset.name.tolist(), get_last_bigram)
    dataset['first_bigram_desc'] = apply_map_fn(dataset.item_description.tolist(), get_first_bigram)
    dataset['last_bigram_desc'] = apply_map_fn(dataset.item_description.tolist(), get_last_bigram)
    
    # Trigrams
    dataset['first_trigram_name'] = apply_map_fn(dataset.name.tolist(), get_first_trigram)
    dataset['last_trigram_name'] = apply_map_fn(dataset.name.tolist(), get_last_trigram)
    dataset['first_trigram_desc'] = apply_map_fn(dataset.item_description.tolist(), get_first_trigram)
    dataset['last_trigram_desc'] = apply_map_fn(dataset.item_description.tolist(), get_last_trigram)
    
    # Name last word
    dataset['name_last_word'] = apply_map_fn(dataset.name.tolist(), get_last_word)
    
    # Get N words
    dataset['name_clean'] = apply_map_fn(dataset.name.tolist(), remove_stop_words)
    dataset['first_n_words_name'] = apply_map_fn(dataset.name_clean.tolist(), get_first_n_words)
    
    dataset.drop(['name_clean'], axis=1, inplace=True)
    name_levels = pd.DataFrame(dataset.first_n_words_name.tolist(), index=dataset.index).fillna('None')
    name_levels.columns = ['name_first_word', 'name_second_word', 'name_third_word', 'name_fourth_word']
    dataset.drop(['first_n_words_name'], axis=1, inplace=True)
    # name_levels = name_levels.astype('category')
    
    dataset = pd.concat([dataset, name_levels], axis=1)
    
    del name_levels
    gc.collect()
    
    # Category - Brand Name Co-occurence terms
    brand_freq = dataset[['brand_name', 'category_name']].groupby('brand_name').agg('count').reset_index()
    skip_brands = brand_freq.loc[brand_freq.category_name < 5,'brand_name'].unique()
    X_cbs = []
    min_freq = 4
    i = 0
    for f in ['category_1', 'category_2', 'category_3']:
        fb = f + '_brand'
        dataset[fb] = dataset[f] + dataset.brand_name
        dataset.loc[dataset.brand_name.isin(skip_brands),fb] = ''
        i = i + 1
        
    dataset['has_price'] = 0
    dataset.loc[(dataset.name.str.contains('[rm]', regex=False)) | (dataset.item_description.str.contains('[rm]', regex=False)), 'has_price'] = 1

    dataset['brackets'] = np.where((dataset.name.str.lower().str.contains('\{\{\{\{')) | (dataset.item_description.str.lower().str.contains('\{\{\{\{')), 1, 0)
    dataset['acessories'] = np.where(dataset.name.str.lower().str.contains('cover|charge|glass|usb|defend|protect|case').fillna(False),1,0)
    
    dataset['brand_in_name'] = dataset.apply(lambda row: row['brand_name'].lower() in row['name'].lower(), axis=1)    

    dataset['len_name'] = dataset.name.str.count('\S+').astype(np.int16)
    dataset['len_desc'] = dataset.item_description.str.count('\S+').astype(np.int16)
    dataset['len_bn'] = dataset.brand_name.apply(len).astype(np.int16)
    dataset['len_cn'] = dataset.category_name.apply(len).astype(np.int16)
    
    # Brand New
    dataset['brand_new'] = dataset.item_description.apply(brand_new)

    # Free Shipping
    dataset['free_shipping'] = dataset.item_description.apply(free_shipping)
    
    # Variant Brand
    dataset['variant_brand'] = dataset.brand_name.apply(lambda x: check_membership(x, brands))
    print('Building Stats')
    
    print('word_count')
    dataset['word_count_name'] = apply_map_fn(dataset.name.tolist(), word_count)
    dataset['word_count_desc'] = apply_map_fn(dataset.item_description.tolist(), word_count)
    dataset['word_count_name'] = dataset['word_count_name'].astype(np.float16)
    dataset['word_count_desc'] = dataset['word_count_desc'].astype(np.float16)
    
    print('mean features')
    dataset['sentence_mean_name'] = (dataset['len_name'] / dataset['word_count_name']).astype(np.int16)
    dataset['sentence_mean_desc'] = (dataset['len_desc'] / dataset['word_count_desc']).astype(np.int16)
    
    print('std features')
    dataset['sentence_std_desc'] = apply_map_fn(dataset.item_description.tolist(), sentence_std)
    dataset['sentence_std_name'] = apply_map_fn(dataset.name.tolist(), sentence_std)
    dataset['sentence_std_desc'] = dataset['sentence_std_desc'].astype(np.float16)
    dataset['sentence_std_name'] = dataset['sentence_std_name'].astype(np.float16)
    
    print('has num name')
    dataset['has_num_name'] = dataset.name.apply(hasNumbers)
    dataset['has_num_item'] = dataset.item_description.apply(hasNumbers)
    
    # Label Encoding - Hashing instead
    cols = ['category_1', 'category_2', 'category_1_brand', 'item_description',
            'category_2_brand', 'category_3_brand', 'brand_name', 'category_3',
            'cond_category_3', 'category_name', 'name_first_word',
            'name_last_word', 'first_bigram_name', 'last_bigram_name',
            'first_bigram_desc', 'last_bigram_desc', 'first_trigram_name', 
            'last_trigram_name', 'first_trigram_desc', 'last_trigram_desc',
            'name_second_word', 'name_third_word', 'name_fourth_word', 'name'] 
         
    print('Before Transformation')   
    print(sys.getsizeof(dataset) / 2 ** 30)
    print(dataset.dtypes)
    print(process.memory_info().rss / 2 ** 30)
    
    print('Start Label Encoding')
    X_le = []
    exclude = set(['item_description', 'name'])
    for idx, f in enumerate([dataset[i] for i in cols]):
        print('Label Encoding', cols[idx])
        dataset[cols[idx]] = label_encoder(f)
        dataset[cols[idx]] = dataset[cols[idx]].astype(np.int32)
        print(process.memory_info().rss / 2 ** 30)
    
    print('Size of dataset', sys.getsizeof(dataset) / 2 ** 30)
    
    return dataset

def get_data():
    print("read train data from csv")
    df_train = pd.read_table("../input/train.tsv", engine='c')
    print('Train shape:',df_train.shape)
    y_train = np.log1p(df_train['price'])
    X = processing_dataset(df_train)
    del df_train

    ridge_oof,sgd_oof = make_oof_predictions(X, y_train)
    del X, y_train
    
    print('Starting predict')
    ridge_test = np.array([])
    sgd_test = np.array([])
    reader = pd.read_table("../input/test.tsv", engine='c', chunksize=360000)
    for df_test in reader:
        print('Test shape:',df_test.shape)
        X2 = processing_dataset(df_test)
        del df_test
        ans_test_ridge = 0
        ans_test_sgd = 0
        for i in range(NUM_FOLDS_OOF):
            print('predict with models from ',i,'-th fold')
            ans_test_ridge = ans_test_ridge + np.expm1(models['ridge_'+str(i)].predict(X2)).clip(3,1e9).ravel() / (NUM_FOLDS_OOF*1.0)
            ans_test_sgd = ans_test_sgd + np.expm1(models['sgd_'+str(i)].predict(X2)).clip(3,1e9).ravel() / (NUM_FOLDS_OOF*1.0)
        ridge_test = np.hstack( [ridge_test, ans_test_ridge] )
        sgd_test = np.hstack( [sgd_test, ans_test_sgd] )
        del X2
    gc.collect()
    
    simulate = False
    
    dtypes = {'item_condition_id': np.int8, 
              'price': np.float16,
              'shipping': np.int8}

    train = pd.read_csv('../input/train.tsv', 
                        delimiter='\t',
                        dtype=dtypes)
    train['ridge'] = ridge_oof
    train['sgd'] = sgd_oof
    train['ridge_sgd'] = np.sqrt(ridge_oof * sgd_oof)
    del ridge_oof, sgd_oof
         
    test = pd.read_csv('../input/test.tsv', 
                       delimiter='\t',
                       dtype=dtypes)
    test['ridge'] = ridge_test
    test['sgd'] = sgd_test
    test['ridge_sgd'] = np.sqrt(ridge_test * sgd_test)
    del ridge_test, sgd_test                       
    
    stage_2_test = test
                       
    if simulate:
        # Make a stage 2 test by copying test five times...
        print('Create Test Simulation')
        test1 = test.copy()
        test2 = test.copy()
        test3 = test.copy()
        test4 = test.copy()
        test5 = test.copy()
        stage_2_test = pd.concat([test1, test2, test3, test4, test5], axis=0)
        del test1, test2, test3, test4, test5
        gc.collect()
        
        print(process.memory_info().rss / 2 ** 30)
        
        # ...then introduce random new words
        def introduce_new_unseen_words(desc):
            desc = desc.split(' ')
            if random.randrange(0, 10) == 0: # 10% chance of adding an unseen word
                new_word = ''.join(random.sample(string.ascii_letters, random.randrange(3, 15)))
                desc.insert(0, new_word)
            return ' '.join(desc)
        stage_2_test.item_description = stage_2_test.item_description.apply(introduce_new_unseen_words)
        print(process.memory_info().rss / 2 ** 30)
        print('Stage 2 test created')
    
    del test
    gc.collect()
        
    print('start transformations')
    # Set missing values
    fill_missing(train)
    fill_missing(stage_2_test)
    
    # Get categories
    train = get_categories(train)
    stage_2_test = get_categories(stage_2_test)
    
    print(train.columns)
    print(stage_2_test.columns)
    
    # Hashed, Bigram & Trigrams
    # train = data_transforms(train)
    y = np.log1p(train.price)
    train.drop(['train_id', 'price'], axis=1, inplace=True)
    print(process.memory_info().rss / 2 ** 30)
    
    # stage_2_test = data_transforms(stage_2_test)
    stage_2_test.drop(['test_id'], axis=1, inplace=True)
    
    nrow_train = train.shape[0]
    dataset = pd.concat([train, stage_2_test]).reset_index(drop=True)
    print(process.memory_info().rss / 2 ** 30)
    
    del train, stage_2_test
    gc.collect()
    print(process.memory_info().rss / 2 ** 30)
    
    items = dataset.item_description.str.lower() + ' ' + dataset.name.str.lower()
    items = items.values.tolist()
    items_train = items[:nrow_train]
    items_test = items[nrow_train:]
    del items
    
    name = dataset.name.str.lower()
    name = name.values.tolist()
    name_train = name[:nrow_train]
    name_test = name[nrow_train:]
    del name
    gc.collect()
    
    print(len(items_train))
    print(len(items_test))
    gc.collect()
    
    print('start transformations')
    dataset = data_transforms(dataset)
    print(dataset.columns)
    
    dataset = dataset_counts(dataset)
    print(sys.getsizeof(dataset) / 2 ** 30)
    
    train = dataset.iloc[:nrow_train, :].reset_index(drop=True)
    test = dataset.iloc[nrow_train:, :].reset_index(drop=True)
    del dataset
    gc.collect()
    
    print(process.memory_info().rss / 2 ** 30)

    cols = ['category_1', 'category_2', 'category_3', 
            'category_1_brand', 'category_2_brand','category_3_brand',
            'brand_name', 'category_name',
            'name', 'item_description', 'name_first_word', 
            'name_last_word', 'first_bigram_name', 'last_bigram_name', 
            'first_bigram_desc', 'last_bigram_desc',
            'first_trigram_name',
            'last_trigram_name','first_trigram_desc', 'last_trigram_desc', 
            'name_second_word', 'name_third_word',
            'name_fourth_word']
        
    z = [(train[f], y, test[f], f) for f in cols]

    pool = Pool(processes=4)
    results = []
        
    feature = 0
    for x in pool.imap_unordered(target_encode_parallel, z):
        train['encode_feature_{}'.format(feature)] = x[0]
        train['encode_feature_{}'.format(feature)] = train['encode_feature_{}'.format(feature)].astype(np.float16)
        test['encode_feature_{}'.format(feature)] = x[1]
        test['encode_feature_{}'.format(feature)] = test['encode_feature_{}'.format(feature)].astype(np.float16)
        feature += 1
        pass
    
    pool.close()
    pool.join()
    
    # TFIDF Transformation
    print('tfidf Transformation')
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=150000)
    print('tfidf fit')
    tfidf_vec.fit(items_train)
    print('train transform')
    train_tfidf = tfidf_vec.transform(items_train)
    print('test transform')
    test_tfidf = tfidf_vec.transform(items_test)
    
    print(train_tfidf.shape)
    print(test_tfidf.shape)
    print('train tfidf size', sys.getsizeof(train_tfidf) / 2 ** 30)
    print('test tfidf size', sys.getsizeof(test_tfidf) / 2 ** 30)
    
    n_comp = 20
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack', random_state=0)
    svd_obj.fit(train_tfidf / 1.0)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf)).astype(np.float16)
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf)).astype(np.float16)
    
    print('train svd size', sys.getsizeof(train_svd) / 2 ** 30)
    print('test svd size', sys.getsizeof(test_svd) / 2 ** 30)
    print(train_svd.head())

    train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]

    train = pd.concat([train, train_svd], axis=1)
    test = pd.concat([test, test_svd], axis=1)

    print(train.shape, test.shape)

    train['tf_idf_item_mean'] = np.mean(train_tfidf, axis=1)
    train['tf_idf_item_sum'] = np.sum(train_tfidf, axis=1)
    train['tf_idf_item_mean'] = train['tf_idf_item_mean'].astype(np.float16)
    train['tf_idf_item_sum'] = train['tf_idf_item_sum'].astype(np.float16)

    test['tf_idf_item_mean'] = np.mean(test_tfidf, axis=1)
    test['tf_idf_item_sum'] = np.sum(test_tfidf, axis=1)
    test['tf_idf_item_mean'] = test['tf_idf_item_mean'].astype(np.float16)
    test['tf_idf_item_sum'] = test['tf_idf_item_sum'].astype(np.float16)
    
    del train_svd, test_svd, train_tfidf, test_tfidf, tfidf_vec, svd_obj
    gc.collect()
    
    print('tfidf Transformation')
    count_vec = transformers['name_cv_ridge'] #CountVectorizer(ngram_range=(1, 2), max_features=500000)
    #print('tfidf fit')
    #count_vec.fit(name_train)
    print('train transform')
    train_count = count_vec.transform(name_train)
    print('test transform')
    test_count = count_vec.transform(name_test)
    
    print(train_count.shape)
    print(test_count.shape)
    print('train count size', sys.getsizeof(train_count) / 2 ** 30)
    print('test count size', sys.getsizeof(test_count) / 2 ** 30)
    
    n_comp = 10
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack', random_state=0)
    svd_obj.fit(train_count / 1.0)
    train_svd = pd.DataFrame(svd_obj.transform(train_count)).astype(np.float16)
    test_svd = pd.DataFrame(svd_obj.transform(test_count)).astype(np.float16)
    
    print('train svd size', sys.getsizeof(train_svd) / 2 ** 30)
    print('test svd size', sys.getsizeof(test_svd) / 2 ** 30)
    print(train_svd.head())

    train_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]

    train = pd.concat([train, train_svd], axis=1)
    test = pd.concat([test, test_svd], axis=1)

    print(train.shape, test.shape)

    train['count_name_mean'] = np.mean(train_count, axis=1)
    train['count_name_sum'] = np.sum(train_count, axis=1)
    train['count_name_mean'] = train['count_name_mean'].astype(np.float16)
    train['count_name_sum'] = train['count_name_sum'].astype(np.float16)

    test['count_name_mean'] = np.mean(test_count, axis=1)
    test['count_name_sum'] = np.sum(test_count, axis=1)
    test['count_name_mean'] = test['count_name_mean'].astype(np.float16)
    test['count_name_sum'] = test['count_name_sum'].astype(np.float16)
    
    del train_svd, test_svd, train_count, test_count, count_vec, svd_obj
    gc.collect()
    
    print('end transformations')
    print(train.shape)
    print(test.shape)
    
    print(sys.getsizeof(train) / 2 ** 30)
    print(sys.getsizeof(test) / 2 ** 30)
    
    print('Data to pickle')
    # test.to_pickle('test.pickle', compression='bz2')
    # del test
    gc.collect()

    # LGBM - Modeling
    model_usage = memory_usage(lightgbm_model(train, y, test, validate=True))
    print('lightgbm model usage', model_usage)
    
    
    
    return 1
    

mem_usage = memory_usage(get_data())
print(mem_usage)
print(process.memory_info().rss / 2 ** 30)