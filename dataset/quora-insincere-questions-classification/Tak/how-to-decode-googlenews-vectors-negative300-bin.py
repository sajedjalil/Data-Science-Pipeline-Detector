"""
Thanks a lot, strideradu! See this kernel if you are interested in modeling with gensim.
https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go

This kernel is related to this discussion.
https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/70991

gensim document
https://radimrehurek.com/gensim/models/keyedvectors.html
"""



import numpy as np

filepath = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

from gensim.models import KeyedVectors
wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True) 

embeddings_index = {}
for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
    coefs = np.asarray(vector, dtype='float32')
    embeddings_index[word] = coefs