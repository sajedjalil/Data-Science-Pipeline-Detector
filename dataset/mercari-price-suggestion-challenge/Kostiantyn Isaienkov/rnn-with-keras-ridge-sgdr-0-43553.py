# coding: utf-8

# mainly forking from notebook
# https://www.kaggle.com/johnfarrell/simple-rnn-with-keras-script

# ADDED
# 5x scaled test set
# category name embedding
# some small changes like lr, decay, batch_size~

import os
import gc
import time
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDRegressor

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 40000

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


df_train = train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = test = pd.read_csv('../input/test.tsv', sep='\t')

train['target'] = np.log1p(train['price'])


print(train.shape)
print('5 folds scaling the test_df')
print(test.shape)
test_len = test.shape[0]
def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test
test = simulate_test(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))

#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)

print('[{}] Finished handling missing data...'.format(time.time() - start_time))



#PROCESS CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
train.head(3)


#PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([train.category_name.str.lower(), 
                      train.item_description.str.lower(), 
                      train.name.str.lower()])

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head(3)

print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))


#EXTRACT DEVELOPTMENT TEST
from sklearn.model_selection import train_test_split
dtrain, dvalid = train_test_split(train, random_state=666, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)


#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 20 #17
MAX_ITEM_DESC_SEQ = 60 #269
MAX_CATEGORY_NAME_SEQ = 20 #8
MAX_TEXT = np.max([np.max(train.seq_name.max())
                   , np.max(test.seq_name.max())
                   , np.max(train.seq_category_name.max())
                   , np.max(test.seq_category_name.max())
                   , np.max(train.seq_item_description.max())
                   , np.max(test.seq_item_description.max())])+2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()])+1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(), 
                        test.item_condition_id.max()])+1

print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))


#KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        ,'category_name': pad_sequences(dataset.seq_category_name
                                        , maxlen=MAX_CATEGORY_NAME_SEQ)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)

print('[{}] Finished DATA PREPARARTION...'.format(time.time() - start_time))



#KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers

def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

dr = 0.25

def get_model():
    #params
    dr_r = dr
    
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]], 
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_size = 60
    
    emb_name = Embedding(MAX_TEXT, emb_size//3)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_category_name = Embedding(MAX_TEXT, emb_size//3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_category_name)
    rnn_layer3 = GRU(8) (emb_name)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    main_l = Dropout(0.3)(Dense(512,activation='relu') (main_l))
    main_l = Dropout(0.2)(Dense(88,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand
                   , category, category_name
                   , item_condition, num_vars], output)
    #optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mse", 
                  optimizer=optimizer)
    return model

def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)
    
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))
    return v_rmsle
#fin_lr=init_lr * (1/(1+decay))**(steps-1)
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))


gc.collect()
#FITTING THE MODEL
epochs = 2
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.006
lr_decay = exp_decay(lr_init, lr_fin, steps)
log_subdir = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

history = model.fit(X_train, dtrain.target
                    , epochs=epochs
                    , batch_size=BATCH_SIZE
                    , validation_split=0.01
                    #, callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=10
                    )
print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))
#EVLUEATE THE MODEL ON DEV TEST
v_rmsle = eval_model(model)
print('[{}] Finished predicting valid set...'.format(time.time() - start_time))


#CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len]*0.8
print('[{}] Finished predicting test set...'.format(time.time() - start_time))

del train
del test
gc.collect()

#Ridge https://www.kaggle.com/apapiu/ridge-script

nrow_train = df_train.shape[0]
y = np.log1p(df_train["price"])
merge: pd.DataFrame = pd.concat([df_train, df_test])

del df_train
del df_test
gc.collect()

handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))

cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
X_description = tv.fit_transform(merge['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]
model = Ridge(solver="sag", fit_intercept=True, alpha = 3.5, random_state=666)
model.fit(X, y)
print('[{}] Finished to train ridge'.format(time.time() - start_time))
predsR = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))
predsR = np.expm1(predsR)
predsR = predsR*0.14
submission["price"] += predsR


model = Ridge(solver="sag", fit_intercept=False, alpha = 1.5, random_state=666)
model.fit(X, y)
print('[{}] Finished to train ridge'.format(time.time() - start_time))
predsRR = model.predict(X=X_test)
print('[{}] Finished to predict ridge'.format(time.time() - start_time))
predsRR = np.expm1(predsRR)
predsRR = predsRR*0.06
submission["price"] += predsRR

submission.to_csv("submission_rnn_ridge.csv", index = False)