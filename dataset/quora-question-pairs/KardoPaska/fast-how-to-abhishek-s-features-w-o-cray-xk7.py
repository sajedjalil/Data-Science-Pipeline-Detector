# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
r"""
Created on Sat May 27 20:18:55 2017
NOW TAKES 2025.74 secs (~34 mins) for train set

--some inefficiencies with keeping the df's separate and appending at the end
-- still don't know if pandas is actually concatenating on the index (f'ing pandas)
-- hopefully col names aren't messed up
-- USES INFO FROM MY OTHER KERNEL
https://www.kaggle.com/kardopaska/fast-gensim-word2vec-w-googlenews

@author: Kardo Paska
"""

"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur (slow coder)
"""
#import pyemd as emd
import pickle #import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm#, tqdm_pandas
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
#from nltk import word_tokenize
import os, time, string


###### FOR SAVING LOCALLY
#            BASE_DIR = r'\i_bet_you_use_linux/'
#            TRAIN_FILE = os.path.join(BASE_DIR, r'traincsv', r'train.csv')
#            
#            Q1_W2V_FILE2SAVE = os.path.join(BASE_DIR, r'FeatureEngineering', r'q1_w2v_train.pkl')
#            Q2_W2V_FILE2SAVE = os.path.join(BASE_DIR, r'FeatureEngineering', r'q2_w2v_train.pkl')
#            ABISHEKS_OUTPUT = os.path.join(BASE_DIR, r'FeatureEngineering', r'quora_featuresindian_train_id.csv')
#            WORDMOVEDIST_OUTPUT = os.path.join(BASE_DIR, r'FeatureEngineering', r'quora_featuresindian_3_WMD.csv')
#            FUZZ_WITH_ID_OUTPUT = os.path.join(BASE_DIR, r'FeatureEngineering', r'quora_featuresindian_3_fuzz.csv')
#            GENSIM_LIMIT_ROWS = None
#            
#            # SEE MY OTHER KERNAL FOR THIS MMAP STUFF
#            BIGASS_GOOG_mmap = r'SmallAssGoogleNews.gnsm'
#            BIGASS_GOOG_NORM_mmap = r'SmallAssGoogleNews_normalized.gnsm'
TRAIN_FILE = r'../input/train.csv'
start_time = time.time()
begin_time = time.time()

##### JUST CALCULATE IT ONCE! DUH
STOP_WORDS = stopwords.words('english')
CHARS2REPLACE = string.punctuation + string.digits


def prnt_updt(what_u_b_updtng, the_start):
    kardo_time = '{:*^32.22}'.format(time.strftime("%a %b %d %I:%M:%S %p"))
    elpsed = '{:,G}'.format(time.time() - the_start)
    print(kardo_time + '{:^32.32}'.format(what_u_b_updtng.upper()) + '{:<32.32}'.format(elpsed))


def new_sent2vec(words_smart):
    ##### this is still slow as shit
    #words = str(s).lower().decode('utf-8') #python 2 crap
    #words = word_tokenize(words)
    words = [w for w in words_smart if not w in STOP_WORDS]
    #words = [w for w in words if w.isalpha()] # obsolete --> pandas str.translate
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def smart_do_fuzz(a_dataframe):
    #a_dataframe.dropna(axis=0, how='any', inplace=True)
    a_dataframe.fillna('dingalingalingalinga', inplace=True)
    fuzz_q1_lower = a_dataframe['question1'].str.lower()
    fuzz_q2_lower = a_dataframe['question2'].str.lower()
    #slice_the_testid = a_dataframe['test_id'] # for test set
    slice_the_testid = a_dataframe['id'] # for train set
    #slice_is_a_dupe = a_dataframe['is_duplicate']
    list_of_dicts_for_pandas = []
    with tqdm(total=len(slice_the_testid), desc='doin fuzz') as pbar:
        for q1_strings, q2_strings, the_id in zip(fuzz_q1_lower, fuzz_q2_lower, slice_the_testid):
            fuzz_qratio = fuzz.QRatio(q1_strings, q2_strings)
            fuzz_WRatio = fuzz.WRatio(q1_strings, q2_strings)
            fuzz_partial_ratio = fuzz.partial_ratio(q1_strings, q2_strings)
            fuzz_partial_token_set_ratio = fuzz.partial_token_set_ratio(q1_strings, q2_strings)
            fuzz_partial_token_sort_ratio = fuzz.partial_token_sort_ratio(q1_strings, q2_strings)
            fuzz_token_set_ratio = fuzz.token_set_ratio(q1_strings, q2_strings)
            fuzz_token_sort_ratio = fuzz.token_sort_ratio(q1_strings, q2_strings)
            hdr_list = ['id', 'fuzz_qratio',
                        'fuzz_WRatio', 'fuzz_partial_ratio',
                        'fuzz_partial_token_set_ratio',
                        'fuzz_partial_token_sort_ratio',
                        'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
            scores_list = [the_id, fuzz_qratio,
                           fuzz_WRatio, fuzz_partial_ratio,
                           fuzz_partial_token_set_ratio,
                           fuzz_partial_token_sort_ratio,
                           fuzz_token_set_ratio, fuzz_token_sort_ratio]

            list_of_dicts_for_pandas.append(dict(zip(hdr_list, scores_list)))
            pbar.update()
    pbar.close()
    return pd.DataFrame(list_of_dicts_for_pandas)


def better_wmd(df_q1_l_s, df_q2_l_s, df_of_id):
    list_of_dicts_for_pandas = []
    with tqdm(total=len(df_of_id), desc='doin better WMD') as pbar:
        for some_id, ss1, ss2 in zip(df_of_id, df_q1_l_s, df_q2_l_s):
            s1 = [w for w in ss1 if w not in STOP_WORDS]
            s2 = [w for w in ss2 if w not in STOP_WORDS]
            wmd_dist = model.wmdistance(s1, s2)
            normwmd_dist = norm_model.wmdistance(s1, s2)
            hdr_list = ['id', 'wmd_dist', 'normwmd_dist']
            scores_list = [some_id, wmd_dist, normwmd_dist]
            list_of_dicts_for_pandas.append(dict(zip(hdr_list, scores_list)))
            pbar.update()
    return pd.DataFrame(list_of_dicts_for_pandas)


#####

GENSIM_LIMIT_ROWS = None

prnt_updt('STARTING UP NOW', start_time)

data = pd.read_csv(TRAIN_FILE, encoding="utf8")#, nrows=1000)
#data = data.drop(['id', 'qid1', 'qid2'], axis=1) ## SERIOUSLY DUDE DON'T BE A DICK
data.fillna('dingalingalingalinga', inplace=True)

tqdm.pandas(desc='{:>20.20}'.format('Doin common words so slow...'))

df_q1_lower_split = data['question1'].str.lower().str.split()
df_q2_lower_split = data['question2'].str.lower().str.split()

data['len_q1'] = data['question1'].str.len() # total length of question chars incl whitespace
data['len_q2'] = data['question2'].str.len()
data['diff_len'] = data.len_q1 - data.len_q2
data['len_char_q1'] = df_q1_lower_split.str.join("").str.len() # total length of chars no whitespace
data['len_char_q2'] = df_q2_lower_split.str.join("").str.len()
data['len_word_q1'] = df_q1_lower_split.str.len() # num of words
data['len_word_q2'] = df_q2_lower_split.str.len()
data['common_words'] = data.progress_apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

prnt_updt('STARTING SMART FUZZ', start_time)


fuzz_with_id = smart_do_fuzz(data) ## JOIN DF AT THE END
prnt_updt('DONE FUZZ, LOADING GOOG1', start_time)

#
###
#######
#############
###################
#############################
#####################################
###########################################
#####################################################
#############################################################
#######################################################################
############# CAN'T DO GENSIM STUFF W/O WORD2VEC FROM MY OTHER KERNEL SORRY (KAGGLE NO HAVE SORRY)
############# CAN'T DO GENSIM STUFF W/O WORD2VEC FROM MY OTHER KERNEL SORRY (KAGGLE NO HAVE SORRY)
############# CAN'T DO GENSIM STUFF W/O WORD2VEC FROM MY OTHER KERNEL SORRY (KAGGLE NO HAVE SORRY)
#######################################################################
#############################################################
#####################################################
###########################################
#####################################
#############################
###################
#############
#######
###
#

#        
#        #start_time = time.time()
#        model = gensim.models.KeyedVectors.load(r'SmallAssGoogleNews.gnsm', mmap='r')
#        prnt_updt('DONE GOOG1, LOADING GOOG2', start_time)
#        
#        
#        
#        #start_time = time.time()
#        norm_model = gensim.models.KeyedVectors.load(r'SmallAssGoogleNews_normalized.gnsm', mmap='r')
#        prnt_updt('DONE GOOG2, STARTING WMD', start_time)
#        
#        
#        df_id_wmd_normwmd = better_wmd(df_q1_lower_split, df_q2_lower_split, data['id']) ## JOIN DF AT THE END
#        #df_id_wmd_normwmd = better_wmd(df_q1_lower_split, df_q2_lower_split, data['test_id']) ## JOIN DF AT THE END
#        prnt_updt('DONE WMD, STARTING SENT2VEC Q1', start_time)
#        
#        
#        
#        question1_vectors = np.zeros((data.shape[0], 300))
#        shit5 = data['question1'].str.lower().str.translate(str.maketrans({key: ' ' for key in CHARS2REPLACE})).str.split()
#        for i, q in enumerate(tqdm(shit5, desc='{:>20.20}'.format('Q1 vecs'))):
#            question1_vectors[i, :] = new_sent2vec(q)
#        prnt_updt('DONE Q1, STARTING SENT2VEC Q2', start_time)
#        
#        
#        
#        shit52 = data['question2'].str.lower().str.translate(str.maketrans({key: ' ' for key in CHARS2REPLACE})).str.split()
#        question2_vectors  = np.zeros((data.shape[0], 300))
#        for i, q in enumerate(tqdm(shit52, desc='{:>20.20}'.format('Q2 vecs dawg'))):
#            question2_vectors[i, :] = new_sent2vec(q)
#        prnt_updt('DONE Q2, STARTING DIST CALCS', start_time)
#        
#        
#        
#        #start_time = time.time()
#        data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(tqdm(np.nan_to_num(question1_vectors)),
#                                                                  np.nan_to_num(question2_vectors))]
#        
#        prnt_updt('DONE DIST, DOING SKEW AND KURTOSIS', start_time)
#        data['skew_q1vec'] = [skew(x) for x in tqdm(np.nan_to_num(question1_vectors))]
#        data['skew_q2vec'] = [skew(x) for x in tqdm(np.nan_to_num(question2_vectors))]
#        data['kur_q1vec'] = [kurtosis(x) for x in tqdm(np.nan_to_num(question1_vectors))]
#        data['kur_q2vec'] = [kurtosis(x) for x in tqdm(np.nan_to_num(question2_vectors))]


#
###
#######
#############
###################
#############################
#####################################
###########################################
#####################################################
#############################################################
#######################################################################
############# IF YOU WANT TO SAVE FOR LATER
############# IF YOU WANT TO SAVE FOR LATER
############# IF YOU WANT TO SAVE FOR LATER
#######################################################################
#############################################################
#####################################################
###########################################
#####################################
#############################
###################
#############
#######
###
#
#        prnt_updt('DONE ALL CALCS, SAVING...', start_time)
#        pickle.dump(question1_vectors, open(Q1_W2V_FILE2SAVE, 'wb'), -1)
#        pickle.dump(question2_vectors, open(Q1_W2V_FILE2SAVE, 'wb'), -1)
#        prnt_updt('FINISHED PICKLING VECS', start_time)

#        #data.set_index('id', inplace=True)
#        fuzz_with_id.set_index('id', inplace=True)
#        df_id_wmd_normwmd.set_index('id', inplace=True)
#        
#        ##### THE JOIN
#        prnt_updt('STARTING MAIN SAVE', start_time)
#        data2 = pd.concat([data, fuzz_with_id, df_id_wmd_normwmd], axis=1)
#        #        data2.to_csv(ABISHEKS_OUTPUT, header=True, index=True, encoding="utf-8")    
#        prnt_updt('DONE MAIN SAVE', start_time)


##### OPTIONAL SAVES
#df_id_wmd_normwmd.to_csv(WORDMOVEDIST_OUTPUT, header=True, index=True, encoding="utf-8")
#fuzz_with_id.to_csv(FUZZ_WITH_ID_OUTPUT, header=True, index=True, encoding="utf-8")



total_time = time.time() - begin_time
print('THANK YOU ABISHEK U DA BEST! ' + str(total_time))
