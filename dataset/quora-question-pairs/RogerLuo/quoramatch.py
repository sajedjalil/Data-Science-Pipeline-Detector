PUNCTUATION='["\'?,\.()/]'
ABBR_DICT={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    PUNCTUATION:'',
    '\s+':' ', # replace multi space with one single space
}


import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

import re
import gensim
from gensim import corpora
from nltk.stem.porter import *

from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

stop_words = stopwords.words('english')

	







# feature of string
def add_features(data):
    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    return data

def tokenize_questions(data):
    question_1_tokenized = []
    question_2_tokenized = []

    for q in data.question1.tolist():
        question_1_tokenized.append(q.lower().split())

    for q in data.question2.tolist():
        question_2_tokenized.append(q.lower().split())

    data["Question_1_tok"] = question_1_tokenized
    data["Question_2_tok"] = question_2_tokenized
    
    return data


def train_dictionary(data):
    questions_tokenized = data.Question_1_tok.tolist() + data.Question_2_tok.tolist()
    
    dictionary = corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.8)
    dictionary.compactify()
    
    return dictionary


def get_vectors(data, dictionary):
    
    df_train = tokenize_questions(data)
    dictionary = train_dictionary(data)

    question1_vec = [dictionary.doc2bow(text) for text in data.Question_1_tok.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in data.Question_2_tok.tolist()]
    
    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    
    return question1_csc.transpose(),question2_csc.transpose()



def get_similarity_values(q1_csc, q2_csc):
    minkowski_dis = DistanceMetric.get_metric('minkowski')
    mms_scale_man = MinMaxScaler()
    mms_scale_euc = MinMaxScaler()
    mms_scale_mink = MinMaxScaler()
    
    cosine_sim = []
    manhattan_dis = []
    eucledian_dis = []
    jaccard_dis = []
    minkowsk_dis = []
    
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])
        sim = md(i,j)
        manhattan_dis.append(sim[0][0])
        sim = ed(i,j)
        eucledian_dis.append(sim[0][0])
        i_ = i.toarray()
        j_ = j.toarray()
        try:
            sim = jsc(i_,j_)
            jaccard_dis.append(sim)
        except:
            jaccard_dis.append(0)
            
        sim = minkowski_dis.pairwise(i_,j_)
        minkowsk_dis.append(sim[0][0])
    
    return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis 



def process_data(file_name):
    TRAIN_ROW_N = 1000
    
    data=pd.read_csv(file_name)
    data.question1=data.question1.str.lower() # conver to lower case
    data.question2=data.question2.str.lower()
    data.question1=data.question1.astype(str)
    data.question2=data.question2.astype(str)
    data.replace(fil.ABBR_DICT,regex=True,inplace=True)
    
    # drop not neccessary attributes
    data = data.drop(['id', 'qid1', 'qid2'], axis=1)
    return data


def feature(data):
    added_feature_data = add_features(data)
    
    df_train = tokenize_questions(data)
    dictionary = train_dictionary(df_train)
    
    q1_csc, q2_csc = get_vectors(df_train, dictionary)
    cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = get_similarity_values(q1_csc, q2_csc)
    
    result = added_feature_data.drop(['Question_1_tok', 'Question_2_tok'], axis=1)
    
    result['cosine_sim'] = cosine_sim
    result['manhattan_dis'] = manhattan_dis
    result['eucledian_dis'] = eucledian_dis
    result['jaccard_dis'] = jaccard_dis
    result['minkowsk_dis'] = minkowsk_dis
    
    return result
#-------------------------------------------------------------
#                           COMPILING
#-------------------------------------------------------------
    TRAINING_DATA = "../input/train.csv"
    
    raw_data = process_data(TRAINING_DATA)
    result = feature(raw_data)
    
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4
    
    
    y_train = result['is_duplicate'].values
    
    x_train = result.drop(['question1', 'question2', 'is_duplicate'], axis=1)
    print (x_train)
    
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
