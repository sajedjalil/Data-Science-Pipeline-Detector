# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
import gc
import time
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer , HashingVectorizer
from sklearn.preprocessing import LabelBinarizer , LabelEncoder , MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import re
import string
stop_words = stopwords.words('english')

NAME_MIN_DF = 20
MAX_FEATURES_ITEM_DESCRIPTION =10000

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_pred), 2)))


def find_cat2(text):
    cat2 =''
    if text.count('/') > 0 :
        cat2  = text.split("/", -1)[1]
    else:
        cat2 = 'missing'
    return cat2    

def find_cat3(text):
    cat3 =''
    if text.count('/') > 1 :
        cat3  = text.split("/", -1)[2]
    else:
        cat3 = 'missing'        
    return cat3  

def find_cat4(text):
    cat4 =''
    if text.count('/') > 2 :
        cat4  = text.split("/", -1)[3]
    else:
        cat4 ='missing'        
    return cat4 

def find_cat5(text):
    cat5 =''
    if text.count('/') > 3 :
        cat5  = text.split("/", -1)[4]
    else:
        cat5 = 'missing'         
    return cat5 
def find_brand1(text):
    cat3 =''
    cat3  = text.split(" ", -1)[0]
    if len(cat3)==1:
        cat3 = text
    return cat3.lower()  

def find_brand2(text):
    cat4 =''
    if text.count(' ') > 0 :
        cat4  = text.split(" ", -1)[1]
    else:
        cat4 ='missing'        
    return cat4.lower() 
def impute_brand(x , brand):
    if (x['brand_name']=='missing') and (str(x['name']).find(brand) > -1):
        x['brand_name'] = brand
    return x
def remove_stop(text , list_brand):
    stop = stopwords.words('english')
    text = re.sub(r'([^\s\w]|_)+', '', text.lower())
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop]
    filtered_sentence = [w for w in filtered_sentence if w.isalpha()]
    filtered_sentence = [w for w in filtered_sentence if len(w) > 1]
    filtered_sentence = [w for w in filtered_sentence if w in list_brand]
    filtered_sentence = [w for w in filtered_sentence if w in filtered_sentence[0]]
    return   filtered_sentence 

def impute_brand1(df):
    df_missing_brand = df[df['brand_name']=='missing']
    df_brand_train = df[df['brand_name']!='missing']

    brand1  = df_brand_train['brand_name'].apply(lambda x :find_brand1(str(x)))

    top_300_brand =brand1.value_counts().sort_values(ascending =False)[0:1000]

    list_brand  = top_300_brand.index.tolist()
    if 'new' in list_brand :
        list_brand.remove('new' )
    mod_text =df_missing_brand['name'].apply(lambda x: remove_stop(x , list_brand ))

    result_brand  = mod_text[mod_text.astype(str) != '[]']

    result_brand = result_brand.apply(lambda list1 : ''.join(list1))
    
    df['brand1']  = df['brand_name'].apply(lambda x :find_brand1(str(x)))
    
    df.loc[df['brand1']=='missing', 'brand1']=result_brand
    
    df['brand1'] = df['brand1'].fillna('missing')
    
    return df 

def wordCount(text):
    try:
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        words = [w for w in txt.split(" ") \
                 if not w in stop_words and len(w)>2]
        return len(words)
    except: 
        return 0

def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e) 
    
def nol(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
def ol(data, m=3):
    return data[(data - np.mean(data)) >= m * np.std(data)]
    
def model_lgbm(X_train , y_train , params):
    
    start_time = time.time()
   # feat_col  =  X_train.columns.tolist()
    
    train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.1, random_state = 144) 
    d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)

#    d_train = lgb.Dataset(X_train, label=y_train, max_bin=8192)
   
    start_time = time.time()
    gc.collect()
    print(params)
    model = lgb.train(params, train_set= d_train ,valid_sets= [d_train, d_valid], valid_names = ['train' , 'eval' ] ,
                      early_stopping_rounds=500, num_boost_round=20000 ,  verbose_eval=500) 

#    model = lgb.cv(params, train_set= d_train,nfold=5,show_stdv=True,seed=144, early_stopping_rounds=50, num_boost_round=20000 ,  verbose_eval=500) 

    print('[{}] Finished Training '.format(time.time() - start_time))
    return model 
    
def model_ridge(X , y , X_test):
    
    start_time = time.time()
            
    model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)
    model.fit(X, y)
    predsR = model.predict(X_test)
    
    #model2 = Ridge(solver="lsqr", fit_intercept=True, random_state=145, alpha = 3)
    #model2.fit(X, y)
    #predsR2 = model2.predict(X_test)

    print('[{}] Finished Training '.format(time.time() - start_time))
    
    return predsR 

def model_train_predict(df , params , params2):
    
    start_time = time.time()
      
    df = impute_brand1(df)
     
    df['desc_len'] = df['item_description'].apply(lambda x: wordCount(x))
    df['name_len'] = df['name'].apply(lambda x: wordCount(x))
    
    le = LabelEncoder()
    
    if len(df['cat4'].unique()) == 1 :
        df = df.drop('cat4' , axis=1)
    else :  
        df['cat4'] = le.fit_transform(df['cat4'])

        
    if len(df['cat5'].unique()) == 1 :
        df =  df.drop('cat5'  , axis=1)
    else :  
        df['cat5'] = le.fit_transform(df['cat5'])
    
    df['brand_name'] = le.fit_transform(df['brand_name'])
    
    df['brand1'] = df['brand1'].fillna('missing') 
    df['brand1'] = df['brand1'].astype('category')
    df['brand1'] = le.fit_transform(df['brand1'])


    df['cat2'] = le.fit_transform(df['cat2'])
    df['cat3'] = le.fit_transform(df['cat3'])
     
    df['item_condition_id'] = le.fit_transform(df['item_condition_id'])
    df['shipping'] = le.fit_transform(df['shipping'])
   
    df = pd.get_dummies(df, columns=['item_condition_id'])
    df = pd.get_dummies(df, columns=['shipping'])
    df = pd.get_dummies(df, columns=['cat2'])
    df = pd.get_dummies(df, columns=['cat3'])

    X_train_df = df[df['source']=='train']
    X_test_df = df[df['source']=='test']
    
    
    print(X_train_df.shape)
    #X_train_df = X_train_df.ix[nol(X_train_df['price']).index]
    #print(X_train_df.shape)
    
    y_train =  np.log1p(X_train_df['price'])
    
    remove_column = ['train_id','test_id','index','source','price','name','item_description','category_name','cat1' ] 

    print(str(datetime.now()))
    
    cv = CountVectorizer(min_df=5 , stop_words='english' ,  ngram_range=(1,1))
    cv.fit(df['name'])
    X_name_train  =  cv.transform(X_train_df['name'])
    X_name_test  =  cv.transform(X_test_df['name'])

    vectorizer = TfidfVectorizer(min_df=5,max_features=MAX_FEATURES_ITEM_DESCRIPTION, tokenizer=tokenize,ngram_range=(1, 1))
    
    vz = vectorizer.fit_transform(df[df['source']=='train']['item_description'])

    temp1  =vectorizer.get_feature_names()
    
    vz = vectorizer.fit_transform(df[df['source']=='test']['item_description'])

    temp2  =vectorizer.get_feature_names()
    
    com_feature  = list(set(temp1).intersection(temp2))
    print(len(com_feature))
    
    hv = TfidfVectorizer(vocabulary=com_feature , ngram_range=(1,1))
    X_description_train = hv.fit_transform(X_train_df['item_description'])
    X_description_test = hv.fit_transform(X_test_df['item_description'])

    X_train_df = X_train_df.drop(remove_column, axis=1).astype('float64')
    X_test_df = X_test_df.drop(remove_column , axis=1).astype('float64')
    
    sparse_merge_train = hstack((csr_matrix(X_train_df.values),X_name_train , X_description_train )).tocsr()

    sparse_merge_test = hstack((csr_matrix(X_test_df.values),X_name_test,X_description_test)).tocsr()

    print(sparse_merge_train.shape ,sparse_merge_test.shape )
    
    train_X, valid_X, train_y, valid_y = train_test_split(sparse_merge_train, y_train, test_size = 0.1, random_state = 42) 
    
    del df
    del X_name_train
    del X_description_train
    del X_name_test
    del X_description_test
       
    print(str(datetime.now()))
    
    model =  model_lgbm(sparse_merge_train , y_train , params)
    predsL1 = model.predict(sparse_merge_test)
    
    model =  model_lgbm(sparse_merge_train , y_train , params2)
    predL2  =   model.predict(sparse_merge_test)
 
    pred  =  0.5*(predsL1 + predL2 )

    gc.collect()
    return np.expm1(pred)



def main() :
    print(str(datetime.now()))
    kernel_start_time = time.time()
    
    train = pd.read_table('../input/train.tsv', sep='\t' ,    engine="python")

    test = pd.read_table('../input/test.tsv', sep="\t" ,   engine="python")
    
   # train = pd.read_table('train.tsv', sep='\t' ,  encoding='utf-8' ,  engine="python")
    
   # test = pd.read_table('test.tsv', sep="\t" , encoding='utf-8' ,  engine="python")

    train['source']='train'
    test['source']='test'

    train = train.drop(train[(train.price < 1.0)].index)

    submission = pd.DataFrame(columns=['test_id' , 'price'])
    merge: pd.DataFrame = pd.concat([train, test])

    handle_missing_inplace(merge)

    to_categorical(merge)

    merge['cat1'] = merge.category_name.str.extract('([^/]+)')
    merge['cat2'] = merge['category_name'].apply(lambda x : find_cat2(str(x)))
    merge['cat3'] = merge['category_name'].apply(lambda x : find_cat3(str(x)))
    merge['cat4'] = merge['category_name'].apply(lambda x : find_cat4(str(x)))
    merge['cat5'] = merge['category_name'].apply(lambda x : find_cat5(str(x)))

    merge_group  =  merge.groupby('cat1')

    women_group = merge_group.get_group('Women').reset_index()
    beauty_group = merge_group.get_group('Beauty').reset_index()
    kids_group = merge_group.get_group('Kids').reset_index()
    elect_group = merge_group.get_group('Electronics').reset_index()
    men_group = merge_group.get_group('Men').reset_index()
    home_group = merge_group.get_group('Home').reset_index()
    vintage_group = merge_group.get_group('Vintage & Collectibles').reset_index()
    others_group = merge_group.get_group('Other').reset_index()
    handmade_group = merge_group.get_group('Handmade').reset_index()
    sports_group = merge_group.get_group('Sports & Outdoors').reset_index()
    missing_group = merge_group.get_group('missing').reset_index()

    del merge_group
    del train
    del test

    print(str(datetime.now()))


    params = {
        'learning_rate': 0.1,
        'application': 'regression',
        'max_depth': 5,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        'tree_learner':'data',
        'data_random_seed': 1,
        'bagging_fraction': 1.0,
        'nthread': 4


    }

    params2 = {
        'learning_rate': 0.2,
        'application': 'regression',
        'max_depth': 5,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        'min_child_samples':5,
        'data_random_seed': 1,
        'bagging_fraction': 1.0,
        'tree_learner':'data',
        'nthread': 4
    }

    params3 = {
        'learning_rate': 0.5,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        'min_child_samples':10,
        'data_random_seed': 2,
        'bagging_fraction': 1.0,
        'tree_learner':'data',
        'nthread': 4
    }

    params4 = {
        'learning_rate': 0.5,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        #'min_child_samples':10,
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'tree_learner':'data',
        'nthread': 4
    }

    print('----- Women------')

    pred = model_train_predict(women_group,params3,params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = women_group[women_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del women_group

    print('----- Beauty------') 
    pred = model_train_predict(beauty_group,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = beauty_group[beauty_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del beauty_group
    print('----- Kids------')

    pred = model_train_predict(kids_group,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = kids_group[kids_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del kids_group
    print('----- Electronics------')

    pred = model_train_predict(elect_group, params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = elect_group[elect_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del elect_group   
    print('----- Men------')

    pred = model_train_predict(men_group, params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = men_group[men_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del men_group
    print('----- Home------')

    pred = model_train_predict(home_group, params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = home_group[home_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del home_group
    print('----- Vintage------')

    pred = model_train_predict(vintage_group,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = vintage_group[vintage_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del vintage_group    
    print('----- Others------')

    pred = model_train_predict(others_group,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = others_group[others_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del others_group
    print('----- HandMade------')

    pred = model_train_predict(handmade_group,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = handmade_group[handmade_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del handmade_group    
    print('----- Sport------')

    pred = model_train_predict(sports_group ,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = sports_group[sports_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del sports_group    
 
    print('----- Missing------')
    pred = model_train_predict(missing_group,params3, params4)
    temp = pd.DataFrame(columns=['test_id' , 'price'])
    temp['test_id'] = missing_group[missing_group['source']=='test']['test_id'].astype('int64')
    temp['price']=pred
    submission = submission.append(temp)

    del missing_group   
    
    del temp
    print(str(datetime.now()))

    print('[{}] Finished Training '.format(time.time() - kernel_start_time))
    submission.info()

    submission = submission.sort_values(by='test_id').reset_index()

    submission['test_id'] =submission['test_id'].astype('int64')

    submission.drop('index' , inplace=True , axis=1)

    submission.to_csv("submission_lgbm_ridge.csv", index=False)


    print(str(datetime.now()))

if __name__ == '__main__':
    main()