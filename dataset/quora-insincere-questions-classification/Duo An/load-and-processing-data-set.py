import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import h5py
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataLoader:
    
    cache_file = 'data.h5fs'
    
    def __init__(self, max_features=95000, maxlen=70):
        self.max_features = max_features
        self.maxlen = maxlen
        print('DataLoader init max_features:{},maxlen:{}'.format(max_features,maxlen))
    
    def load_all(self, update=False):
        if not os.path.exists(self.cache_file) or update:
            train_X, test_X, train_y, word_index = self.load_and_processing()
            em1 = self.load_embedding_matrix('glove',word_index)
            em2 = self.load_embedding_matrix('fasttext',word_index)
            em3 = self.load_embedding_matrix('paragram',word_index)
            
            pickle.dump(word_index, open('word_index.pk','wb'))
            with h5py.File(self.cache_file, 'w') as cache:
                cache['train_X'] = train_X
                cache['train_y'] = train_y
                cache['test_X']  = test_X
                cache['embedding_matrix_glove'] = em1
                cache['embedding_matrix_fasttext'] = em2
                cache['embedding_matrix_paragram'] = em3
            return train_X, test_X, train_y, word_index, em1, em2, em3
        else:
            cache = h5py.File(self.cache_file, 'r')
            train_X = cache['train_X'].value
            train_y = cache['train_y'].value
            test_X = cache['test_X'].value
            em1 = cache['embedding_matrix_glove'].value
            em2 = cache['embedding_matrix_fasttext'].value
            em3 = cache['embedding_matrix_paragram'].value
                
            word_index = pickle.load(open('word_index.pk','rb'))
            return train_X, test_X, train_y, word_index, em1, em2, em3
    
    def load_and_processing(self):
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
        print("Train shape : ",train_df.shape)
        print("Test shape : ",test_df.shape)
        ## fill up the missing values
        train_X = train_df["question_text"].fillna("_##_").values
        test_X = test_df["question_text"].fillna("_##_").values

        ## Tokenize the sentences
        print('Tokenizing the sentences')
        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(list(train_X))
        train_X = tokenizer.texts_to_sequences(train_X)
        test_X = tokenizer.texts_to_sequences(test_X)

        ## Pad the sentences 
        print('Padding the sentences')
        train_X = pad_sequences(train_X, maxlen=self.maxlen)
        test_X = pad_sequences(test_X, maxlen=self.maxlen)

        ## Get the target values
        train_y = train_df['target'].values

        #shuffling the data
        np.random.seed(2018)
        trn_idx = np.random.permutation(len(train_X))

        train_X = train_X[trn_idx]
        train_y = train_y[trn_idx]

        return train_X, test_X, train_y, tokenizer.word_index
    
    def load_embedding_matrix(self, name, word_index):
        print('Building {} embedding index'.format(name))
        if name == 'glove':
            embeddings_index = self.load_embedding_index_glove(word_index)
        elif name == 'fasttext':
            embeddings_index = self.load_embedding_index_fasttext(word_index)
        else:
            embeddings_index = self.load_embedding_index_paragram(word_index)
        
        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        
        # word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in tqdm(word_index.items(), desc='Building {} embedding matrix'.format(name)):
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix 
    
    def load_embedding_index_glove(self, word_index):
        EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
        return embeddings_index
        
    def load_embedding_index_fasttext(self, word_index):    
        EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
        return embeddings_index

    def load_embedding_index_paragram(self, word_index):
        EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)
        return embeddings_index
        

data_loader = DataLoader()
train_X, test_X, train_y, word_index, em1, em2, em3 = data_loader.load_all(True)