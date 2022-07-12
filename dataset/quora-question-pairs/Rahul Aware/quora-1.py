# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim
from nltk.tokenize import word_tokenize
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
# print(check_output(["ls", "../working"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv',nrows=100)
df_dup = df_train[df_train.is_duplicate == 1]

train_qs = pd.Series(df_dup['question1'].tolist()).astype(str)
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in train_qs]

dictionary = gensim.corpora.Dictionary(gen_docs)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
# print(gen_docs[1])
# print(corpus[1])
tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.Similarity('../working',tf_idf[corpus],num_features=len(dictionary))

query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
# print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
# print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
# print(query_doc_tf_idf)

sims[query_doc_tf_idf]
                                      