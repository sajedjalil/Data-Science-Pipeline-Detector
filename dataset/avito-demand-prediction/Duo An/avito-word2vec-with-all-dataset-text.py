# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
input_dir = '../input/'
files = ['train_active','test_active','train','test']

from keras.preprocessing.text import text_to_word_sequence
from gensim.models import Word2Vec
from tqdm import tqdm
import gc
import logging

logging.basicConfig(level=logging.INFO)


model = Word2Vec(size=100, window=5, max_vocab_size=500000)
def fit_w2v(sentences, update=False):
    sentences = [text_to_word_sequence(sentence) for sentence in tqdm(sentences)]
    model.build_vocab(sentences, update=update)
    model.train(sentences, total_examples=model.corpus_count, epochs=3)
    
    
chuncksize = 1000000
usecols = ['param_1','param_2','param_3','title', 'description']
update = False
for k in range(15):
    for file in files:
        print(20 * '=' + 'Epoch {}, File {}'.format(k, file) + 20 * '=')
        for df in pd.read_csv(input_dir + file + '.csv', chunksize=chuncksize, usecols=usecols):
            df['text'] = df['param_1'].str.cat([df.param_2,df.param_3,df.title,df.description], sep=' ',na_rep='')
            sentences = df['text'].values
            fit_w2v(sentences, update)
            del sentences, df
            gc.collect()
            update = True
    model.save('avito.w2v'.format(k))
    print(80 * '=')