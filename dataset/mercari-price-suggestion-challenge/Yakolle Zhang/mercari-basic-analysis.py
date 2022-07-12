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

import gc
import os
import string
import warnings
from multiprocessing import current_process, Process, Queue
from time import time

import pandas as pd
import lightgbm as lgb
import numpy as np
from keras.initializers import glorot_normal
from keras.layers import Input, Dense, concatenate, GRU, Embedding, Flatten, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from scipy.sparse import hstack, vstack
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

data_dir = '../input'
train_df = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t')
train_df.fillna('0', inplace=True)

# tvr = TfidfVectorizer(token_pattern=r'\w+', dtype=np.float32)
# tr_x_name = tvr.fit_transform(train_df['name'])
# print(tr_x_name.shape)
# del tvr,tr_x_name
# gc.collect()

# tvr = TfidfVectorizer(token_pattern=r'\w+', min_df=2, dtype=np.float32)
# tr_x_name = tvr.fit_transform(train_df['name'])
# print(tr_x_name.shape)
# del tvr,tr_x_name
# gc.collect()


# tvr = TfidfVectorizer(token_pattern=r'\w+', dtype=np.float32)
# tr_x_desc = tvr.fit_transform(train_df.item_description)
# print(tr_x_desc.shape)
# del tvr,tr_x_desc
# gc.collect()

# tvr = TfidfVectorizer(token_pattern=r'\w+', min_df=2, dtype=np.float32)
# tr_x_desc = tvr.fit_transform(train_df.item_description)
# print(tr_x_desc.shape)
# del tvr,tr_x_desc
# gc.collect()

# tvr = TfidfVectorizer(token_pattern=r'\w+', min_df=3, dtype=np.float32)
# tr_x_desc = tvr.fit_transform(train_df.item_description)
# print(tr_x_desc.shape)
# del tvr,tr_x_desc
# gc.collect()

# tvr = TfidfVectorizer(token_pattern=r'\w+', ngram_range=(1, 2), dtype=np.float32)
# tr_x_desc = tvr.fit_transform(train_df.item_description)
# print(tr_x_desc.shape)
# del tvr,tr_x_desc
# gc.collect()

# tvr = TfidfVectorizer(token_pattern=r'\w+', ngram_range=(1, 2), min_df=2, dtype=np.float32)
# tr_x_desc = tvr.fit_transform(train_df.item_description)
# print(tr_x_desc.shape)
# del tvr,tr_x_desc
# gc.collect()

# tvr = TfidfVectorizer(token_pattern=r'\w+', ngram_range=(1, 2), min_df=3, dtype=np.float32)
# tr_x_desc = tvr.fit_transform(train_df.item_description)
# print(tr_x_desc.shape)
# del tvr,tr_x_desc
# gc.collect()

tvr = TfidfVectorizer(token_pattern=r'\w+', ngram_range=(1, 2), min_df=30, dtype=np.float32)
tr_x_desc = tvr.fit_transform(train_df.item_description)
print(tr_x_desc.shape)
del tvr,tr_x_desc
gc.collect()