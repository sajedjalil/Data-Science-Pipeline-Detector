"""
NOTE: I ran this to update it for the data change, output matricies should once again be the same size as the train and test data

I have built this to try to make use of Kaggle's new feature where you can use outputs from one kernel
as inputs for another kernel. This way I won't have to create a sparse matrix in each new kernel
and hopefully this will help avoid hitting the time cap for any algorithms I run.

This is a modified version of Jeremy Howard's NB-SVM baseline and basic EDA (0.06 lb) script:
https://www.kaggle.com/jhoward/nb-svm-baseline-and-basic-eda-0-06-lb

I've extracted only the code needed to generate and save an n_gram term document matrix 
of 1-3 words in length in Compressed Sparse Row format. 

The processed data are saved to a sparse matrix (.npz format) using the scipy.sparse.save_npz() function.
With the new Kaggle i/o function these files be loaded into other kernels. 
Click on the 'New Kernel using this data' button below this script.
Then import the matrices using the following code:

#load in the processed data from train_and_test_to_matrix.py
from scipy import sparse

train_sparse = sparse.load_npz('../input/sparse_train_punc.npz')
test_sparse = sparse.load_npz('../input/sparse_test_punc.npz')


"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re, string
from scipy import sparse

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_sub = pd.read_csv('../input/sample_submission.csv')


#fillna
train['comment_text'].fillna("unk", inplace=True)

test['comment_text'].fillna("unk", inplace=True)


# get the list of tokenizers 
# this is the way to split text into tokens, 
# also split on whitespace in the tokenize functino
symbols = f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])'

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 
	return re_tok.sub(r' \1 ', s).split()

"""
#make sure that works as advertized
tokenize(train['comment_text'][1])


?CountVectorizer
Convert a collection of text documents to a matrix of token counts

This implementation produces a sparse representation of the counts using
scipy.sparse.csr_matrix.
"""

vec = CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, max_features=1500000)

train_sparse = vec.fit_transform(train['comment_text'])
test_sparse = vec.transform(test['comment_text'])

#save the matrix to a file so we can start back here
sparse.save_npz('sparse_train_punc.npz', train_sparse)
sparse.save_npz('sparse_test_punc.npz', test_sparse)

# sparse.load_npz()