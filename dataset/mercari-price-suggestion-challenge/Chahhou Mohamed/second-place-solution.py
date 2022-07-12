from __future__ import division
import pyximport
pyximport.install()
import os
import random
import numpy as np
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=1)
from keras import backend
tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, BatchNormalization, PReLU
from keras.initializers import he_uniform
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam, SGD
from keras.models import Model

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from itertools import combinations
from sklearn.linear_model import LinearRegression
import re
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from multiprocessing import Pool
import gc
import time
import nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

###################################################################
#GLOBAL VARIABLES
path = '../input/'
split = -1#1400000 # use -1 for submission, otherwise tha value of split is the number of instances in train 
cores = 4
max_text_length=60###################
min_df_one=5
min_df_bi=5

def clean_str(text):
    try:
        text = ' '.join( [w for w in text.split()[:max_text_length]] )        
        text = text.lower()
        text = re.sub(u"é", u"e", text)
        text = re.sub(u"ē", u"e", text)
        text = re.sub(u"è", u"e", text)
        text = re.sub(u"ê", u"e", text)
        text = re.sub(u"à", u"a", text)
        text = re.sub(u"â", u"a", text)
        text = re.sub(u"ô", u"o", text)
        text = re.sub(u"ō", u"o", text)
        text = re.sub(u"ü", u"u", text)
        text = re.sub(u"ï", u"i", text)
        text = re.sub(u"ç", u"c", text)
        text = re.sub(u"\u2019", u"'", text)
        text = re.sub(u"\xed", u"i", text)
        text = re.sub(u"w\/", u" with ", text)
        
        text = re.sub(u"[^a-z0-9]", " ", text)
        text = u" ".join(re.split('(\d+)',text) )
        text = re.sub( u"\s+", u" ", text ).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text

def load_data( ):
    print ('LOAD 1.4M ROWS FOR TRAIN')
    df_train = pd.read_csv(path+'train.tsv', sep='\t', encoding='utf-8')
    df_train['item_condition_id'].fillna(2, inplace=True)
    df_train['shipping'].fillna(0, inplace=True)
    if split>0:
        df_train = df_train.loc[:split].reset_index(drop=True)
    df_train = df_train.loc[df_train.price>0].reset_index(drop=True)
    df_train['price'] = np.log1p(df_train['price']).astype(np.float32)
    df_train.drop('train_id', axis=1, inplace=True)
    return df_train

def create_count_features(df_data):
    def lg(text):
        text = [x for x in text.split() if x!='']
        return len(text)
    df_data['nb_words_item_description'] = df_data['item_description'].apply(lg).astype(np.uint16)

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cores)
    pool = Pool(cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
def clean_str_df(df):
    return df.apply( lambda s : clean_str(s))

def prepare_data(df_data, train=True):
    print ('Prepare data....')
    
    def fill_brand_name(x):
        try:
            k=[]
            for n in [4,3,2,1]:
                temp =  [' '.join(xi) for xi in ngrams(x.split(' '), n) if ' '.join(xi) in   brand_names  ] 
                if len(temp)>0:
                    k = k+temp
            if len(k) > 0:
                return k[0]
            else:
                return np.NaN
        except:
            return np.NaN
        
    def fill_cat(x, i, new=False):
        try:
            if new:
                return x.split('/')[i-1].strip()
            else:
                return ' '.join( x.split('/') ).strip()
        except:
            return ''
        
    df_data['name'].fillna('', inplace=True)
    df_data['item_description'].fillna('', inplace=True)
    df_data['item_description'] = df_data['item_description'].apply(lambda x : x.replace('No description yet',''))
    
    #create 3 categories and remove / from category name and replace nan
    df_data['category_name'].fillna('//', inplace=True)
    df_data['category1'] = df_data.category_name.apply(lambda x : x.split('/')[0].strip())
    df_data['category2'] = df_data.category_name.apply(lambda x : x.split('/')[1].strip())
    df_data['category3'] = df_data.category_name.apply(lambda x : x.split('/')[2].strip())
    df_data['category_name'] = df_data['category_name'].apply( lambda x : ' '.join( x.split('/') ).strip() )

    create_count_features(df_data)     
    df_data['nb_words_item_description'] /= max_text_length

    df_data['brand_name'] = parallelize_dataframe(df_data['brand_name'], clean_str_df)  
    df_data['name'] = parallelize_dataframe(df_data['name'], clean_str_df)  
    df_data['item_description'] = parallelize_dataframe(df_data['item_description'], clean_str_df)                                                                            
    
    df_data.loc[df_data['brand_name'].isnull(), 'brand_name'] = df_data.loc[df_data['brand_name'].isnull(),
                                                                            'name'].apply(fill_brand_name)
    df_data['brand_name'].fillna('', inplace=True)
    
    if train:        
        for feat in ['brand_name', 'category_name', 'category1', 'category2', 'category3']:
            temp = df_data[feat].unique()
            lb = LabelEncoder()
            df_data[feat] = lb.fit_transform(df_data[feat]).astype(np.uint16)
            labels_dict[feat] = (lb, temp)
    else:   
        for feat in ['brand_name', 'category1', 'category2', 'category3', 'category_name']  :
            idx = labels_dict[feat][1]
            df_data.loc[ -df_data[feat].isin(idx), feat ] = ''
            df_data[feat] = labels_dict[feat][0].transform(df_data[feat]).astype(np.uint16)

    df_data['name_old'] = df_data['name'].copy()    
    
    df_data['brand_cat']  = 'cat1_'+df_data['category1'].astype(str)+' '+\
    'cat2_'+df_data['category2'].astype(str)+' '+\
    'cat3_'+df_data['category3'].astype(str)+' '+\
    'brand_'+df_data['brand_name'].astype(str) 
    
    df_data['name']  = df_data['brand_cat']  + ' ' + df_data['name']
    
    df_data['name_desc']  = df_data['name'] + ' ' +\
    df_data['item_description'].apply( lambda x : ' '.join( x.split()[:5] ) )
    
    df_data['item_condition_id'] = df_data['item_condition_id']/5.
    return df_data

def word_count(text, dc):
    text = set( text.split(' ') ) 
    for w in text:
        dc[w]+=1
def remove_low_freq(text, dc):
    return ' '.join( [w for w in text.split() if w in dc] )
    
def create_bigrams(text):
    try:
        text = np.unique( [ wordnet_lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words ] )
        lst_bi = []
        for combo in combinations(text, 2):
            cb1=combo[0]+combo[1]
            cb2=combo[1]+combo[0]
            in_dict=False
            if cb1 in word_count_dict_one:
                new_word = cb1
                in_dict=True
            if cb2 in word_count_dict_one:
                new_word = cb2
                in_dict=True
            if not in_dict:
                new_word = combo[0]+'___'+combo[1]
            if len(cb1)>=0:
                lst_bi.append(new_word)
        return ' '.join( lst_bi )
    except:
        return ' '
def create_bigrams_df(df):
    return df.apply( create_bigrams )
##########################################################################################################
############################  TRAIN PROCESSING  #####################################
##########################################################################################################
print('*'*50)
labels_dict = dict()
df_train = load_data()
print (df_train.shape)
brand_names = df_train.groupby('brand_name').size()  
start_time = time.time()
df_train = prepare_data(df_train, train=True)
print('[{}] Finished TRAIN DATA PREPARATION'.format(time.time() - start_time))

#STORE ALL WORDS  FREQUENCY and Filter
#################################################
start_time = time.time()
word_count_dict_one = defaultdict(np.uint32)
for feat in ['name','item_description' ]:
    df_train[feat].apply(             lambda x : word_count(x, word_count_dict_one) )
rare_words = [key for key in word_count_dict_one if  word_count_dict_one[key]<min_df_one ]
for key in rare_words :
    word_count_dict_one.pop(key, None)
for feat in ['name','item_description' ]:
    df_train[feat]      = df_train[feat].apply( lambda x : remove_low_freq(x, word_count_dict_one) )
word_count_dict_one=dict(word_count_dict_one)
print('[{}] Finished COUNTING WORDS FOR NAME AND DESCRIPTION...'.format(time.time() - start_time))

#Create ALL 2_ways combinations (Custom Bigrams)
#################################################
start_time = time.time()
word_count_dict_bi=defaultdict(np.uint32)
def word_count_bi(text):
    text =  text.split(' ') 
    for w in text:
        word_count_dict_bi[w]+=1
df_train['name_bi']      = parallelize_dataframe( df_train['name_desc'],  create_bigrams_df )
df_train['name_bi'].apply(word_count_bi )
rare_words = [key for key in word_count_dict_bi if  word_count_dict_bi[key]<min_df_bi ]
for key in rare_words :
    word_count_dict_bi.pop(key, None)
df_train['name_bi']      = df_train['name_bi'].apply( lambda x : remove_low_freq(x, word_count_dict_bi) )
print('[{}] Finished CREATING BIGRAMS...'.format(time.time() - start_time))

#####################################

start_time = time.time()
word_count_dict_bi = dict(word_count_dict_bi)
vocabulary_one = word_count_dict_one.copy()
vocabulary_bi = word_count_dict_bi.copy()
for dc in [vocabulary_one,  vocabulary_bi]:
    cpt=0
    for key in dc:
        dc[key]=cpt
        cpt+=1
print('[{}] Finished CREATING VOCABULARY ...'.format(time.time() - start_time))

#####################################

mean_dc=dict()
for feat in ['category1', 'category2', 'category3', 'category_name', 'brand_name'  ]:
    mean_dc[feat] = df_train.groupby(feat)['price'].mean().astype(np.float32)
    mean_dc[feat] /= np.max(mean_dc[feat])
    df_train['mean_price_'+feat] = df_train[feat].map(mean_dc[feat]).astype(np.float32)
    df_train['mean_price_'+feat].fillna( mean_dc[feat].mean(), inplace=True  )
    
##################################### vectorizers
def tokenize(text):
    return [w for w in text.split()]
start_time = time.time()
vect_name_one            = CountVectorizer(vocabulary= vocabulary_one,   dtype=np.uint8,
                                           tokenizer=tokenize, binary=True ) 
train_name_one  = vect_name_one.fit_transform( df_train['name'] )
print (train_name_one.shape)
print('[{}] Finished Vectorizing Onegram Name'.format(time.time() - start_time))

start_time = time.time()
vect_item_one            = CountVectorizer(vocabulary= vocabulary_one,   dtype=np.uint8, 
                                           tokenizer=tokenize, binary=True ) 
train_item_one  = vect_item_one.fit_transform( df_train['item_description']  )
print (train_item_one.shape)
print('[{}] Finished Vectorizing Onegram Item Description'.format(time.time() - start_time))

start_time = time.time()
vect_name_bi           = CountVectorizer(vocabulary= vocabulary_bi,   dtype=np.uint8, 
                                         tokenizer=tokenize, binary=True ) 
train_name_bi  = vect_name_bi.fit_transform( df_train['name_bi']  )
print (train_name_bi.shape)
print('[{}] Finished Vectorizing BiGram Name'.format(time.time() - start_time))

    
#############################################################################################################
##############################TRAIN AND PREDICT################################
#############################################################################################################
keep = ['item_condition_id', 'shipping', 'nb_words_item_description',
       'mean_price_category1', 'mean_price_category2', 'mean_price_category3',
        'mean_price_category_name', 'mean_price_brand_name']

#RIDGE MODEL 1
dtrain_y = df_train.price.values
dtrain  = hstack((df_train[keep].values, train_name_one, train_item_one, train_name_bi  )).tocsr()
print ('RIDGE MATRIX SIZE : ',dtrain.shape)
start_time=time.time()
model_ridge_name = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',
                    max_iter=100,   normalize=False, random_state=0,  tol=0.0025)
model_ridge_name.fit(dtrain, dtrain_y)
print ('ridge time : ',time.time()-start_time)
del dtrain, train_name_bi, train_name_one, train_item_one
df_train.drop('name_bi', axis=1, inplace=True)
gc.collect()

####################################################################################################
#SPARSE NN MODEL
def tokenize(text):
    return [ w for w in text.split()]

start_time = time.time()
vect_sparse = CountVectorizer(lowercase=False, min_df=5, ngram_range=(1,2), max_features=200000,
                                    dtype=np.uint8,       tokenizer=tokenize, strip_accents=None, binary=True )
vect_sparse.fit( df_train['name']+' '+df_train['item_description']  )
def get_keras_sparse(df):
    X = {'sparse_data': vect_sparse.transform( df['name']+' '+df['item_description']  ) ,
        'item_condition': np.array(df['item_condition_id']),
        'shipping': np.array(df["shipping"]),
        'temp': np.array(df["mean_price_category2"]),
        'temp2': np.array(df["nb_words_item_description"])
    }
    return X
train_keras      = get_keras_sparse(df_train)

def sparseNN():                                             
    sparse_data = Input( shape=[train_keras["sparse_data"].shape[1]], 
        dtype = 'float32',   sparse = True, name='sparse_data')  

    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[1], name="shipping")
    temp = Input(shape=[1], name="temp")
    temp2 = Input(shape=[1], name="temp2")
    
    x = Dense(200 , kernel_initializer=he_uniform(seed=0) )(sparse_data)    
    x = PReLU()(x)
    x = concatenate( [x, item_condition, shipping, temp, temp2] ) 
    x = Dense(200 , kernel_initializer=he_uniform(seed=0) )(x)
    x = PReLU()(x)
    x = Dense(100 , kernel_initializer=he_uniform(seed=0) )(x)
    x = PReLU()(x)
    x= Dense(1)(x)
    
    model = Model([sparse_data, item_condition, shipping,   temp, temp2],x)
    
    optimizer = Adam(.0011)
    model.compile(loss="mse", optimizer=optimizer)
    return model

BATCH_SIZE = 2000
epochs = 3

sparse_nn = sparseNN()

print("Fitting SPARSE NN model ...")

mean_price = np.mean(df_train.price.values)

for ep in range(epochs):
    BATCH_SIZE = int(BATCH_SIZE*2)
    sparse_nn.fit(  train_keras, (df_train.price.values-mean_price), 
                      batch_size=BATCH_SIZE, epochs=1, verbose=10 )

del train_keras
gc.collect

##########################################################################################################

# FASTTEXT AND RCNN DATA PREPARATION
MAX_NAME_SEQ = 20
MAX_ITEM_DESC_SEQ = 30
MAX_TEXT = len(vocabulary_one)+ 1
MAX_CATEGORY    = np.max(df_train['category_name'].max()) + 1
MAX_CATEGORY1   = np.max(df_train['category1'].max()) + 1
MAX_CATEGORY2   = np.max(df_train['category2'].max()) + 1
MAX_CATEGORY3   = np.max(df_train['category3'].max()) + 1
MAX_BRAND       = np.max(df_train['brand_name'].max()) + 1

def preprocess_keras(text):
    return [ vocabulary_one[w] for w in (text.split())[:MAX_ITEM_DESC_SEQ] ]
def preprocess_keras_df(df):
    return df.apply( preprocess_keras )

start_time = time.time()
df_train['seq_name'] = parallelize_dataframe(df_train['name'], preprocess_keras_df)
df_train['seq_item_description'] = parallelize_dataframe(df_train['item_description'], preprocess_keras_df)

def get_keras_fasttext(df):
    X = {
        'name': pad_sequences(df['seq_name'], maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(df['seq_item_description'], maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(df['brand_name']),
        'category_name': np.array(df['category_name']),
        'category1': np.array(df['category1']),
        'category2': np.array(df['category2']),
        'category3': np.array(df['category3']),
        'item_condition': np.array(df['item_condition_id']),
        'shipping': np.array(df["shipping"]),
        'temp': np.array(df["mean_price_category2"]),
        'temp2': np.array(df["nb_words_item_description"]),
        
    }
    return X
train_keras      = get_keras_fasttext(df_train)
print('[{}] Finished Converting to Sequence for FASTTEXT NN'.format(time.time() - start_time))

#FASTTEXT MODEL
def fasttext_model():
    name = Input(shape=[train_keras["name"].shape[1]], name="name")
    item_desc = Input(shape=[train_keras["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    category1 = Input(shape=[1], name="category1")
    category2 = Input(shape=[1], name="category2")
    category3 = Input(shape=[1], name="category3")
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[1], name="shipping")
    temp = Input(shape=[1], name="temp")
    temp2 = Input(shape=[1], name="temp2")
    
    shared_embedding = Embedding(MAX_TEXT, 50)    
    emb_name = shared_embedding (name)
    emb_item_desc = shared_embedding (item_desc)
    
    val=10
    emb_brand_name = Flatten() ( Embedding(MAX_BRAND, val)(brand_name)    )
    emb_category1 = Flatten() ( Embedding(MAX_CATEGORY, val)(category1) )
    emb_category2 = Flatten() ( Embedding(MAX_CATEGORY, val)(category2) )
    emb_category3 = Flatten() ( Embedding(MAX_CATEGORY, val)(category3) )
    
    emb_name = GlobalAveragePooling1D( name='output_name_max' )(emb_name)
    emb_item_desc = GlobalAveragePooling1D(name='output_item_max' )(emb_item_desc)

    x = concatenate([  item_condition , shipping, emb_name, emb_item_desc, 
    emb_brand_name, emb_category1, emb_category2, emb_category3    ])
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)

    x = Dense(1, activation="linear") (x)
    model = Model([name, item_desc,  brand_name , category_name,
                   category1, category2, category3, item_condition, shipping, temp, temp2],
                   x)
    optimizer = Adam(.002)
    model.compile(loss="mse", optimizer=optimizer)

    return model

BATCH_SIZE = 128
epochs = 4

fasttext_model = fasttext_model()

print("Fitting FASTTEXT NN model ...")

for ep in range(epochs):
    BATCH_SIZE = int(BATCH_SIZE*2)
    fasttext_model.fit(  train_keras, (df_train.price.values-mean_price), 
                      batch_size=BATCH_SIZE, epochs=1, verbose=10 )
del train_keras
gc.collect()
##################################################################################################################
#DATA PREPARATION FOR CHAR NGRAMS NN
start_time = time.time()
vect_sparse_char = CountVectorizer(lowercase=False, min_df=5, ngram_range=(1,4), analyzer='char',
                              max_features=100000,     dtype=np.uint8, 
                              strip_accents=None, binary=True )
vect_sparse_char.fit( df_train['name_old'] )

vect_sparse_word = CountVectorizer(lowercase=False, min_df=2, ngram_range=(1,3), 
                              max_features=100000,     dtype=np.uint8, 
                              tokenizer=tokenize, strip_accents=None, binary=True )
vect_sparse_word.fit( df_train['brand_cat'] )

vect_sparse_desc = CountVectorizer(lowercase=False, min_df=20, ngram_range=(1,1), 
                              max_features=100000,     dtype=np.uint8, 
                              tokenizer=tokenize, strip_accents=None, binary=True )
vect_sparse_desc.fit( df_train['item_description'] ) 
                     
def get_keras_sparse_char(df):
    X = {'sparse_data_char': vect_sparse_char.transform( df['name_old'] ) ,
         'sparse_data_word': vect_sparse_word.transform( df['brand_cat'] ) ,
         'sparse_data_desc' : vect_sparse_desc.transform( df['item_description'] ) ,
        'item_condition': np.array(df['item_condition_id']),
        'shipping': np.array(df["shipping"]),
        'temp': np.array(df["mean_price_category2"]),
        'temp2': np.array(df["nb_words_item_description"])
    }
    return X
train_keras      = get_keras_sparse_char(df_train)
print('[{}] Finished Converting to Sequence for CHAR NGRAMS NN'.format(time.time() - start_time))

#CHAR NGRAM NN MODEL
def sparse_char_model():                                             
    sparse_data_char = Input( shape=[train_keras["sparse_data_char"].shape[1]], 
        dtype = 'float32',   sparse = True, name='sparse_data_char')
    sparse_data_word = Input( shape=[train_keras["sparse_data_word"].shape[1]], 
        dtype = 'float32',   sparse = True, name='sparse_data_word')
    sparse_data_desc = Input( shape=[train_keras["sparse_data_desc"].shape[1]], 
        dtype = 'float32',   sparse = True, name='sparse_data_desc')
    
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[1], name="shipping")
    temp = Input(shape=[1], name="temp")
    temp2 = Input(shape=[1], name="temp2")
    
    x = Dense(100  )(sparse_data_char)    
    x = PReLU()(x)
    
    y = Dense(100  )(sparse_data_word)    
    y = PReLU()(y)
    
    z = Dense(100  )(sparse_data_desc)    
    z = PReLU()(z)

    x = concatenate( [x, y, z, item_condition, shipping, temp, temp2] )    
    x = Dense(50)(x)
    x = PReLU()(x)

    x= Dense(1)(x)
    model = Model([sparse_data_char, sparse_data_word, sparse_data_desc,
                   item_condition, shipping,   temp, temp2],x)
    
    optimizer = Adam(0.002)
    model.compile(loss="mse", optimizer=optimizer)
   
    return model

BATCH_SIZE = 2048
epochs = 3

sparse_char = sparse_char_model()

print("Fitting SPARSE CHAR NGRAM NN model ...")

for ep in range(epochs):
    BATCH_SIZE = (BATCH_SIZE*2)
    sparse_char.fit(  train_keras, (df_train.price.values-mean_price), 
                      batch_size=BATCH_SIZE, epochs=1, verbose=10 )
                      
del train_keras, df_train
gc.collect()

##################################################################################################################
##############################SUBMISSION PREPROCESSING #########################################
##################################################################################################################

start_time = time.time()
models_predictions = defaultdict(list)
submission_idx = []
chunk_counter=1
for df_submission in pd.read_csv(path+'test.tsv', sep='\t', encoding='utf-8' , chunksize=800000):
    df_submission['price']=-99
    print(' ')
    print ('LOADING CHUNK ', chunk_counter)
    chunk_counter += 1
    submission_idx += list(df_submission.test_id.values)
    
    if split>0:
        print ('USING HOLDOUT AS SUBMISSION')
        df_submission = pd.read_csv(path+'train.tsv', sep='\t',  encoding='utf-8')[split:]
        df_submission = df_submission.loc[df_submission.price>0].reset_index(drop=True)
        df_submission['price'] = np.log1p(df_submission['price']).astype(np.float32)
        df_submission.drop('train_id', axis=1, inplace=True)
        sub_price = df_submission.price.values
        
    df_submission['item_condition_id'].fillna(2, inplace=True)
    df_submission['shipping'].fillna(0, inplace=True)   

    print ('SUBMISSION CHUNK SIZE : ',df_submission.shape)

    df_submission = prepare_data(df_submission, train=False)
    
    start_time = time.time()
    for feat in ['name', 'item_description']:
        df_submission[feat]             = df_submission[feat].apply( lambda x : remove_low_freq(x, vocabulary_one) )
        
    df_submission['name_bi']      = parallelize_dataframe( df_submission['name_desc'], create_bigrams_df)
    df_submission['name_bi']      = df_submission['name_bi'].apply( lambda x : remove_low_freq(x, word_count_dict_bi) )
    print('[{}] Finished NAME BIGRAMS and REMOVING NEW WORDS...'.format(time.time() - start_time))

    for feat in ['category1', 'category2', 'category3', 'category_name', 'brand_name'  ]:
        df_submission['mean_price_'+feat] = df_submission[feat].map(mean_dc[feat]).astype(np.float32)
        df_submission['mean_price_'+feat].fillna( mean_dc[feat].mean(), inplace=True  )
    
    start_time = time.time()
    submission_name_one = vect_name_one.transform(     df_submission['name'] )
    submission_item_one = vect_item_one.transform(     df_submission['item_description']  )
    submission_name_bi = vect_name_bi.transform(     df_submission['name_bi']  )
    print('[{}] Finished VECTORIZING SUBMISSION...'.format(time.time() - start_time))
    
    df_submission['seq_name'] = parallelize_dataframe(df_submission['name'], preprocess_keras_df)
    df_submission['seq_item_description'] = parallelize_dataframe(df_submission['item_description'], preprocess_keras_df)
    
    
    
    #RIDGE MODEL 1
    dsubmit_y = df_submission.price.values
    dsubmit  = hstack((df_submission[keep].values, submission_name_one,submission_item_one,  submission_name_bi  )).tocsr()
    preds = model_ridge_name.predict(dsubmit)
    models_predictions['RIDGE1'] += list( preds )
    if split>0:
        print ('RIDGE SCORE :',np.sqrt(mean_squared_error( dsubmit_y, preds )))
    del dsubmit, submission_name_bi, submission_name_one, submission_item_one
    gc.collect()

        
    #SPARSE NN MODEL
    submission_keras = get_keras_sparse(df_submission)
    preds = sparse_nn.predict(submission_keras, batch_size=100000)+mean_price
    models_predictions['SPARSENN'] += list( preds.reshape((1,-1))[0] )
    if split>0:
        print (time.time()-start_time, 'SPARSE NN : ',  np.sqrt(mean_squared_error(df_submission.price.values,preds)) )
        

    #FASTTEXT NN MODEL
    submission_keras = get_keras_fasttext(df_submission)
    preds = fasttext_model.predict(submission_keras, batch_size=100000)+mean_price
    models_predictions['FASTTEXT'] += list( preds.reshape((1,-1))[0] )
    if split>0:
        print (time.time()-start_time, 'FASTTEXT : ',  np.sqrt(mean_squared_error(df_submission.price.values,preds)) )
        
    
    #SPARSE NN MODEL
    submission_keras = get_keras_sparse_char(df_submission)
    preds = sparse_char.predict(submission_keras, batch_size=100000)+mean_price
    models_predictions['SPARSENN2'] += list( preds.reshape((1,-1))[0] )
    if split>0:
        print (time.time()-start_time, 'SPARSE NN CHAR NGRAM : ',  np.sqrt(mean_squared_error(df_submission.price.values,preds)) )

    del submission_keras, df_submission
    gc.collect()
        
submission_preds_df = pd.DataFrame(models_predictions)

if split>0:
    print ('ENSEMBLE MEAN SCORE :',np.sqrt(mean_squared_error( sub_price, submission_preds_df.mean(axis=1) )))
    print(' ')
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit( submission_preds_df.values, sub_price)
    preds = lr.predict(submission_preds_df.values)
    print ('ENSEMBLE LR SCORE :', np.sqrt(mean_squared_error(sub_price, preds)) )
    print (lr.coef_)

if split==-1:    
    mysubmission=pd.DataFrame()
    mysubmission['test_id']=submission_idx
    preds = np.expm1( submission_preds_df.mean(axis=1) )
    preds[preds<3]=3
    preds[preds>1000]=1000
    mysubmission['price'] = preds
    mysubmission.to_csv('mean.csv', index=False)
   
    
    print (mysubmission.shape)