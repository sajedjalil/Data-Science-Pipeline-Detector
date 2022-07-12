import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import xgboost as xgb
from nltk.corpus import wordnet
import re
import difflib
import nltk
from nltk import bigrams
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

eng_stopwords = set(stopwords.words('english'))
color = sns.color_palette()

pd.options.mode.chained_assignment = None  # default='warn'

train_df_all = pd.read_csv("../input/train.csv")
train_df = train_df_all[54000:56000]
test_df_all = pd.read_csv("../input/test.csv")
test_df = test_df_all[54000:56000]

from nltk.stem import PorterStemmer
ps = PorterStemmer()

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

tfidf_txt = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist() + test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)
tfidf.fit_transform(tfidf_txt)


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

total = 1
def feature_extraction(row):
    global total
    print("processing" + str(total))
    total = total + 1
    out_list = []
    
    q1 = re.sub('\W+', ' ', str(row['question1']).lower())
    q2 = re.sub('\W+', ' ', str(row['question2']).lower())
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    w1_stem = set([ps.stem(word) for word in w1 if word not in eng_stopwords and word != "aed"])
    w2_stem = set([ps.stem(word) for word in w2 if word not in eng_stopwords and word != "aed"])
    
    # get unigram features #
    unigrams_que1 = [word for word in w1]
    unigrams_que2 = [word for word in w2]
    
    unigrams_que1_stem = [word for word in w1_stem]
    unigrams_que2_stem = [word for word in w2_stem]
    common_unigrams_len = len(set(unigrams_que1_stem).intersection(set(unigrams_que2_stem)))
    common_unigrams_ratio = float(common_unigrams_len) / max(len(set(unigrams_que1_stem).union(set(unigrams_que2_stem))),1)
    #out_list.extend([common_unigrams_len, common_unigrams_ratio])

    # get bigram features #
    bigrams_que1 = [i for i in ngrams(unigrams_que1, 2)]
    bigrams_que2 = [i for i in ngrams(unigrams_que2, 2)]
    common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
    common_bigrams_ratio = float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))),1)
    #out_list.extend([common_bigrams_len, common_bigrams_ratio])

    # get trigram features #
    #trigrams_que1 = [i for i in ngrams(unigrams_que1, 3)]
    #trigrams_que2 = [i for i in ngrams(unigrams_que2, 3)]
    #common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
    #common_trigrams_ratio = float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))),1)
    #out_list.extend([common_trigrams_len, common_trigrams_ratio])
    
    
    # nouns #
    #question1_nouns = [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(q1).lower())) if t[:1] in ['N']]
    #question2_nouns = [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(q2).lower())) if t[:1] in ['N']]
    #z_noun_match = sum([1 for w in question1_nouns if w in question2_nouns])  #takes long
    
    # lengths #
    #z_len1 = len(str(q1))
    #z_len2 = len(str(q2))
    #z_word_len1 = len(str(q1).split())
    #z_word_len2 = len(str(q2).split())
    
    # difflib #
    z_match_ratio = diff_ratios(q1, q2)  #takes long

    # tfidf #
    z_tfidf_sum1 = np.sum(tfidf.transform([str(q1)]).data)
    z_tfidf_sum2 = np.sum(tfidf.transform([str(q2)]).data)
    z_tfidf_mean1 = np.mean(tfidf.transform([str(q1)]).data)
    z_tfidf_mean2 = np.mean(tfidf.transform([str(q2)]).data)
    #z_tfidf_len1 = len(tfidf.transform([str(q1)]).data)
    #z_tfidf_len2 = len(tfidf.transform([str(q2)]).data)
    
    out_list.extend([common_unigrams_ratio,common_bigrams_ratio])
    out_list.extend([z_match_ratio,z_tfidf_sum1,z_tfidf_sum2,z_tfidf_mean1,z_tfidf_mean2])
    return out_list

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'logloss'
        params["eta"] = 0.02
        params["subsample"] = 0.7
        params["min_child_weight"] = 1
        params["colsample_bytree"] = 0.7
        params["max_depth"] = 4
        params["silent"] = 1
        params["seed"] = seed_val
        params["early_stopping_rounds"]=10
        num_rounds = 2000 
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=10)
        else:
                xgtest = xgb.DMatrix(test_X)
                model = xgb.train(plst, xgtrain, num_rounds)
                
        pred_test_y = model.predict(xgtest)

        loss = 1
        if test_y is not None:
                loss = log_loss(test_y, pred_test_y)
                return pred_test_y, loss, model
        else:
            return pred_test_y, loss, model


train_X = np.vstack( np.array(train_df.apply(lambda row: feature_extraction(row), axis=1)) ) 
test_X = np.vstack( np.array(test_df.apply(lambda row: feature_extraction(row), axis=1)) )
train_y = np.array(train_df["is_duplicate"])
test_id = np.array(test_df["test_id"])

# rebalance data #
train_X_dup = train_X[train_y==1]
train_X_non_dup = train_X[train_y==0]

train_X = np.vstack([train_X_non_dup, train_X_dup, train_X_non_dup, train_X_non_dup])
train_y = np.array([0]*train_X_non_dup.shape[0] + [1]*train_X_dup.shape[0] + [0]*train_X_non_dup.shape[0] + [0]*train_X_non_dup.shape[0])
del train_X_dup
del train_X_non_dup
print("Mean target rate : ",train_y.mean())

# xgbosst training #
kf = KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds, lloss, model = runXGB(dev_X, dev_y, val_X, val_y)
    break

import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = (30.0, 30.0)
plt.rcParams.update({'font.size': 22})
plt.xlabel('xlabel', fontsize=22)
plt.ylabel('ylabel', fontsize=22)
plt.legend(loc=2,prop={'size':22})
xgb.plot_importance(model); plt.show()

#import matplotlib.pyplot as plt
#import seaborn as sns
#plt.rcParams['figure.figsize'] = (100.0, 100.0)
#xgb.plot_tree(model, num_trees=0); plt.show()

# get testing result #
#xgtest = xgb.DMatrix(test_X)
#preds = model.predict(xgtest)

#out_df = pd.DataFrame({"test_id":test_id, "is_duplicate":preds})
#out_df.to_csv("../output/xgb_20170526.csv", index=False)