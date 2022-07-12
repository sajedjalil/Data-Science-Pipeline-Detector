# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os 
import matplotlib.pyplot as plt 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn 
import tensorflow as tf 
import transformers


# Function to get the length of the tweet text col 
def get_str_len(txt):
    if type(txt) != type('check'):
        return txt
    else:
        return len(txt.split(' '))

# Difference Encode the sentiment column 
def sent_diff_encode(txt):
    ret_val = 0
    if txt == 'neutral':
        ret_val = 0
    elif txt == 'positive':
        ret_val = 1
    else:
        ret_val = 2
    return ret_val 

def check_link(txt):
    if type(txt) != type('string'):
        return 0
    if 'http' in txt and type(txt) == type('string'):
        return 1
    else:
        return 0 

def punc_check(txt):
    if type(txt) != type('string'):
        return 0 
    puncs = '<>?/:;\}}[]_=+-@$%#&()!'
    ret = 0
    for i in txt:
        if i in puncs:
            ret = 1
    return ret 

# sentance1  = "AI is our friend and it has been friendly"
# sentance2 = "AI and humans have always been friendly"

# sent1 = ["Our", "Is", "It"]
# both = ["AI", "Has", "Been", "And"]
# sent2 = ["human", "always"]

# score = len(both) / (len(both) + len(sent1) + len(sent2))
# print(0.5)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
file_loc = '/kaggle/input/tweet-sentiment-extraction/'

# Formatting some columns and passing the function 
df = pd.read_csv(file_loc+'train.csv')
test_df = pd.read_csv(file_loc+'test.csv')
df['text'] = df['text'].str.strip()
df['text_len'] = df['text'].apply(get_str_len)
max_words = int(df['text_len'].max())
df['sent_encode'] = df['sentiment'].apply(sent_diff_encode)
df['has_link'] = df['text'].apply(check_link)
df['punctuations'] = df['text'].apply(punc_check)
df['select_text_length'] = df['selected_text'].apply(get_str_len)

# Check how many null values there are in the data 
null_vals = df['text'].isnull().sum()

# There is one null value in all the text 
df = df.dropna()
df.reset_index(drop=True, inplace=True)


y_data = df['sent_encode']
x_data = df['text']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                    test_size=0.3, random_state=4)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)


train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

train_data = pad_sequences(train_sequences, maxlen=max_words, padding='post')
test_data = pad_sequences(test_sequences, maxlen=max_words, padding='post')

train_labels = np.asarray(y_train)
test_labels = np.asarray(y_test)

compute_loss = losses.SparseCategoricalCrossentropy()

model = transformers.RobertaModel(transformers.RobertaConfig())
bert_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
print(bert_tokenizer.encode(["Hello", "people", "how", "are", "people", "today"]))


# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# with tpu_strategy.scope():
# test_mod = Sequential([
#     layers.Embedding(10000, 80), 
#     layers.Conv1D(50, 5, activation='relu'), 
#     layers.MaxPooling1D(5), 
#     layers.Conv1D(50, 5, activation='relu'), 
#     layers.GlobalAveragePooling1D(), 
#     layers.Flatten(), 
#     layers.Dense(20, activation='relu'), 
#     layers.Dense(3, activation='softmax')
# ])

# tester = train_data[0].reshape(1,100)
# print(test_mod(tester))

# test_mod.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
#                  loss=compute_loss, 
#                 metrics=['acc'])

# hist = test_mod.fit(train_data, train_labels, epochs=50, 
#                     validation_split=0.2, batch_size=84)

# test_loss, test_acc = test_mod.evaluate(test_data, test_labels)
# acc = hist.history['acc']
# loss = hist.history['loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training Acc')
# plt.title("Training Accuracy")
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.title('Training loss')
# plt.legend()

# plt.show()

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session