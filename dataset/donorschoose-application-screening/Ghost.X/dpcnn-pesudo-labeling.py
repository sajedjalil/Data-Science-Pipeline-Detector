# In the recently concluded Toxic Comments Classification Challenge
# @neptune shared his solutions https://github.com/neptune-ml/open-solution-toxic-comments
# @neptune's solution includes the implementation of Deep Pyramid Convolutional Neural Networks(DPCNN),
# http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf,
# a low-complexity, high-efficiently neural network, deep but runs fast.
# @MichaelS also implemented DPCNN in his kernel https://www.kaggle.com/michaelsnell/conv1d-dpcnn-in-keras

import os; os.environ['OMP_NUM_THREADS'] = '4'
import re
import gc
import random
import numpy as np
np.random.seed(7)
import pandas as pd

from functools import reduce
from functools import partial

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import MaxPooling1D, Conv1D, add, Dropout, PReLU, BatchNormalization, Flatten
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM, Dropout, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder




#-------------Kernel Config--------------------
# Set quick_run to True to run this code on a small part of training data.
quick_run = False

epochs = 5
batch_size = 32
embed_size = 300
max_features = 20000

project_maxlen = 180
resouse_max_len = 20
maxlen = project_maxlen + resouse_max_len

if quick_run == True:
    max_features = 1000
    epochs = 1

EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
#-------------End of Kernel Config--------------------


#-------------Data Preprocessing--------------------
train_df = pd.read_csv('../input/donorschoose-application-screening/train.csv')
test_df = pd.read_csv('../input/donorschoose-application-screening/test.csv')
resouse_df = pd.read_csv('../input/donorschoose-application-screening/resources.csv')

resouse_df['description'].fillna('', inplace=True)
res_nums = pd.DataFrame(resouse_df[['id', 'price']].groupby('id').price.agg(['count', 
                                                                             'sum', 
                                                                             'min', 
                                                                             'max', 
                                                                             'mean',  
                                                                             'std', 
                                                                             lambda x: len(np.unique(x)),])).reset_index()
res_nums = res_nums.rename(columns={'count': 'res_count', 
                                    'sum': 'res_sum',
                                    'min':  'res_min', 
                                    'max':  'res_max',
                                    'mean': 'res_mean', 
                                    'std':  'res_std',
                                    '<lambda>': 'res_unique' })
res_descp = resouse_df[['id', 'description']].groupby('id').description.agg([ lambda x: ' '.join(x) ]).reset_index().rename(columns={'<lambda>':'res_description'})
resouse_df = res_nums.merge(res_descp, on='id', how='left')
train_df = train_df.merge(resouse_df, on='id', how='left')
test_df = test_df.merge(resouse_df, on='id', how='left')
del res_nums
del res_descp
del resouse_df

if quick_run == True:
    train_df = train_df[:10000]
    test_df = test_df[:100]
train_target = train_df['project_is_approved'].values
del train_df['project_is_approved']
gc.collect()

# project_essay preprocessing
print('Essay cols...')
essay_cols = ['project_essay_1', 'project_essay_2','project_essay_3', 'project_essay_4']
essay_length_cols = [item+'_len' for item in essay_cols]

def count_essay_length(df):
    for col in essay_cols:
        df[col] = df[col].fillna('')
        df[col+'_len'] = df[col].apply(len)
    return df
train_df = count_essay_length(train_df)
test_df = count_essay_length(test_df)

train_df['project_essay'] = ''
test_df['project_essay'] = ''
for col in essay_cols:
    train_df['project_essay'] += train_df[col] + ' '
    test_df['project_essay'] += test_df[col] + ' '
train_df = train_df.drop(essay_cols, axis=1)
test_df = test_df.drop(essay_cols, axis=1)

# time features
print('Time cols...')
time_cols = ['sub_year', 'sub_month', 'sub_day', 'sub_hour',  'sub_dayofweek', 'sub_dayofyear']
def time_stamp_features(df):
    time_df = pd.to_datetime(df['project_submitted_datetime'])
    df['sub_year'] = time_df.apply(lambda x: x.year)
    df['sub_month'] = time_df.apply(lambda x: x.month)
    df['sub_day'] = time_df.apply(lambda x: x.day)
    df['sub_hour'] = time_df.apply(lambda x: x.hour)
    df['sub_dayofweek'] = time_df.apply(lambda x: x.dayofweek)
    df['sub_dayofyear'] = time_df.apply(lambda x: x.dayofyear)
    return df
train_df = time_stamp_features(train_df)
test_df = time_stamp_features(test_df)

# string and num cols 
print('filling lost')
str_cols = ['teacher_prefix', 'school_state',
            'project_submitted_datetime', 'project_grade_category',
            'project_subject_categories', 'project_subject_subcategories',
            'project_title', 'project_resource_summary','res_description', 'project_essay']
num_cols = ['teacher_number_of_previously_posted_projects', 
            'res_count', 'res_sum', 'res_min', 'res_max', 'res_mean', 'res_std', 'res_unique'] + essay_length_cols
train_df[str_cols] =train_df[str_cols].fillna('unknown')
train_df[num_cols] = train_df[num_cols].fillna(0)
test_df[str_cols] =test_df[str_cols].fillna('unknown')
test_df[num_cols] = test_df[num_cols].fillna(0)
for col in str_cols:
    train_df[col] = train_df[col].str.lower()
    test_df[col] = test_df[col].str.lower()

scaler = MinMaxScaler()
train_none_text_features = scaler.fit_transform(train_df[num_cols].values)
test_none_text_features = scaler.transform(test_df[num_cols].values)
train_df = train_df.drop(num_cols, axis=1)
test_df = test_df.drop(num_cols, axis=1)
del scaler

train_df['project_descp'] = train_df['project_subject_categories'] + ' ' + train_df['project_subject_subcategories'] + ' ' + train_df['project_title'] + ' ' + train_df['project_resource_summary'] + ' ' + train_df['project_essay']
test_df['project_descp'] = test_df['project_subject_categories'] + ' ' + test_df['project_subject_subcategories'] + ' ' + test_df['project_title'] + ' ' + test_df['project_resource_summary'] + ' ' + test_df['project_essay']
train_df = train_df.drop(['project_title', 'project_resource_summary', 'project_essay'], axis=1)
test_df = test_df.drop(['project_title', 'project_resource_summary', 'project_essay'], axis=1)
gc.collect()

# remove punctuations and numbers
print("remove punctuations and numbers")
def clean_descp(descp):
    low_case = re.compile('([a-z]*)')
    words = low_case.findall(descp)
    return ' '.join(words)
train_df['project_descp']  = train_df['project_descp'].apply(clean_descp)
test_df['project_descp']  = test_df['project_descp'].apply(clean_descp)

# Label Encoding
label_cols = ['teacher_prefix',  'school_state',  'project_grade_category', 'project_subject_categories', 'project_subject_subcategories'] + time_cols
for col in label_cols:
    le = LabelEncoder()
    le.fit(np.hstack([train_df[col].values, test_df[col].values]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
train_label_features = train_df[label_cols].values
test_label_features = test_df[label_cols].values
train_df = train_df.drop(label_cols, axis=1)
test_df = test_df.drop(label_cols, axis=1)
del le
gc.collect()

# text tokenize
print('Text Tokenize...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_df['project_descp']) + list(test_df['project_descp']) + list(train_df['res_description']) + list(test_df['res_description']))
train_pj = sequence.pad_sequences(tokenizer.texts_to_sequences(train_df['project_descp']), maxlen=project_maxlen)
test_pj = sequence.pad_sequences(tokenizer.texts_to_sequences(test_df['project_descp']), maxlen=project_maxlen)
train_res = sequence.pad_sequences(tokenizer.texts_to_sequences(train_df['res_description']), maxlen=resouse_max_len)
test_res = sequence.pad_sequences(tokenizer.texts_to_sequences(test_df['res_description']), maxlen=resouse_max_len)

train_num_features = np.hstack([train_none_text_features, train_label_features])
test_num_features = np.hstack([test_none_text_features, test_label_features])
train_seq = np.hstack([train_pj, train_res, train_num_features])
test_seq = np.hstack([test_pj, test_res, test_num_features])

del train_num_features
del test_num_features
del train_none_text_features
del train_label_features
del test_none_text_features
del test_label_features
del train_df
del test_df
gc.collect()
#-------------End of Data Preprocessing--------------------


#-------------DPCNN Model--------------------
def get_model():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

    filter_nr = 128
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 256
    spatial_dropout = 0.4
    dense_dropout = 0.3
    train_embed = False
    
    pj_repeat = 5     # dpcnn block repeated times on project text 
    rs_repeat = 1     # dpcnn block repeated times on resources text 

    project = Input(shape=(project_maxlen,), name='project')
    emb_project = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(project)
    emb_project = SpatialDropout1D(spatial_dropout)(emb_project)
    pj_block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_project)
    pj_block1 = BatchNormalization()(pj_block1)
    pj_block1 = PReLU()(pj_block1)
    pj_block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(pj_block1)
    pj_block1 = BatchNormalization()(pj_block1)
    pj_block1 = PReLU()(pj_block1)
    pj_resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_project)
    pj_resize_emb = PReLU()(pj_resize_emb)
    pj_block1_output = add([pj_block1, pj_resize_emb])
    for _ in range(pj_repeat):  
        pj_block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(pj_block1_output)
        pj_block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(pj_block1_output)
        pj_block2 = BatchNormalization()(pj_block2)
        pj_block2 = PReLU()(pj_block2)
        pj_block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(pj_block2)
        pj_block2 = BatchNormalization()(pj_block2)
        pj_block2 = PReLU()(pj_block2)
        pj_block1_output = add([pj_block2, pj_block1_output])
    
    resouse = Input(shape=(resouse_max_len,), name='resouse')
    emb_resouse = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(resouse)
    emb_resouse = SpatialDropout1D(spatial_dropout)(emb_resouse)
    rs_block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_resouse)
    rs_block1 = BatchNormalization()(rs_block1)
    rs_block1 = PReLU()(rs_block1)
    rs_block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(rs_block1)
    rs_block1 = BatchNormalization()(rs_block1)
    rs_block1 = PReLU()(rs_block1)
    rs_resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_resouse)
    rs_resize_emb = PReLU()(rs_resize_emb)

    rs_block1_output = add([rs_block1, rs_resize_emb])
    for _ in range(rs_repeat):  
        rs_block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(rs_block1_output)
        rs_block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(rs_block1_output)
        rs_block2 = BatchNormalization()(rs_block2)
        rs_block2 = PReLU()(rs_block2)
        rs_block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(rs_block2)
        rs_block2 = BatchNormalization()(rs_block2)
        rs_block2 = PReLU()(rs_block2)
        rs_block1_output = add([rs_block2, rs_block1_output])
        
    pj_output = GlobalMaxPooling1D()(pj_block1_output)
    rs_output = GlobalMaxPooling1D()(rs_block1_output)
    pj_output = BatchNormalization()(pj_output)
    rs_output = BatchNormalization()(rs_output)
    
    inp_num = Input(shape=(train_seq.shape[1]-maxlen, ), name='num_input')
    conc = concatenate([pj_output, rs_output, inp_num])
    
    output = Dense(dense_nr, activation='linear')(conc)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=[project, resouse, inp_num], outputs=output)
    model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()
    return model

#-------------End of DPCNN Model--------------------


#-------------Embedding--------------------
print('Creating embedding_matrix')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
del embeddings_index
gc.collect()
#-------------End of Embedding--------------------


#-------------Add pseudo label to training data-----------------------
pl_sample_rate = 0.4
pl_df = pd.read_csv('../input/dcas-lgb-rework/dcas_lgb_sub.csv')
pl_labels = pl_df['project_is_approved'].values
test_len = pl_df.shape[0]
pl_len = int(min(test_len*pl_sample_rate, test_seq.shape[0]))
pl_index = random.sample(list(range(pl_len)), int(pl_len))

X_tra, X_val, y_tra, y_val = train_test_split(train_seq, train_target, train_size=0.95, random_state=233)
X_tra = np.vstack([X_tra, test_seq[pl_index,:]])
y_tra = np.vstack([y_tra.reshape(-1,1), pl_labels[pl_index].reshape(-1,1)])
y_tra = y_tra.reshape((y_tra.shape[0],))
#-------------End of pseudo labeling-----------------------

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = score
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
def get_dpcnn_data(X_tra):
    return {'project' : X_tra[:,:project_maxlen], 
            'resouse' : X_tra[:,project_maxlen:project_maxlen+resouse_max_len], 
            'num_input' : X_tra[:,maxlen:]  }
            

X_tra = get_dpcnn_data(X_tra)
X_val = get_dpcnn_data(X_val)
test_data = get_dpcnn_data(test_seq)

model = get_model()
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
early_stopping = EarlyStopping(monitor='roc_auc_val', patience=3, mode='max',min_delta=0.0005)  
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                              callbacks=[early_stopping, RocAuc], 
                              verbose=2)

if quick_run == False:
    predict_test = model.predict(test_data, batch_size=1024)[:, 0]
    sample_df = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')
    sample_df['project_is_approved'] = predict_test
    sample_df.to_csv('dpcnn_pl_sub.csv', index=False)
