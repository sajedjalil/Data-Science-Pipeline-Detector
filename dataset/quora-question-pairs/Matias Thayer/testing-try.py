import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import log_loss
from stop_words import get_stop_words
from unicodedata import normalize
from sklearn.cross_validation import train_test_split

def load_data():
    df_train = pd.read_csv('../input/train.csv')#.sample(1000)
    df_test = pd.read_csv('../input/test.csv')#.sample(1000)
    return df_train, df_test

def get_weight(count, eps=10000, min_count=2):
    ''' If a word appears only once, we ignore it completely (likely a typo)
     Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller'''
    if count < min_count:
        return 0
    else:
        return 1. / float(count + eps)

def count_word_match_share(row):
    #stops = get_stop_words('english')
    #stops = [normalize('NFKD', stop).encode('ascii','ignore') for stop in stops]
    stops = {'here', 'o', 'own', 'being', 'did', 'up', 'under', 'below', 're', 'itself', 'why', 'for', 'the', 'more', 'do', 'just', 'these', 'himself', 'with', 'off', 't', 'further', 'only', 'doesn', 'hasn', 'each', 'doing', 'theirs', 'm', 'same', 'their', 'because', 'nor', 'myself', 'how', 's', 'am', 'into', 'after', 'any', 'ours', 'our', 'most', 'no', 'too', 'does', 'as', 'all', 'herself', 'are', 'they', 'then', 'very', 'she', 'ourselves', 'few', 'where', 'needn', 'some', 'had', 'haven', 'if', 'her', 'it', 'against', 'yourselves', 'what', 'now', 'again', 'before', 'should', 'mustn', 'hadn', 'will', 'weren', 'be', 'y', 'is', 'mightn', 'than', 'them', 'at', 'i', 'in', 'there', 'its', 'about', 'through', 'such', 'out', 'on', 'ma', 'have', 've', 'was', 'and', 'were', 'so', 'ain', 'until', 'hers', 'yours', 'he', 'aren', 'you', 'can', 'not', 'has', 'shouldn', 'which', 'when', 'his', 'themselves', 'don', 'above', 'down', 'other', 'd', 'didn', 'to', 'we', 'those', 'who', 'll', 'shan', 'once', 'your', 'couldn', 'yourself', 'wouldn', 'while', 'been', 'that', 'between', 'a', 'him', 'both', 'isn', 'but', 'or', 'whom', 'having', 'this', 'from', 'during', 'me', 'by', 'an', 'my', 'won', 'over', 'of', 'wasn'}
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    return q1words, q2words

def word_match_share(row):
    q1words, q2words = count_word_match_share(row)
    # The computer-generated chaff includes a few questions that are nothing but stopwords
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float(len(q1words) + len(q2words))
    return R

def tfidf_word_match_share(row):
    q1words, q2words = count_word_match_share(row)
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / float(np.sum(total_weights))
    return R


def plot_distribution_word_match(df,): 
    word_match = df['word_match']
    plt.hist(word_match[df['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
    plt.hist(word_match[df['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
    plt.legend()
    plt.title('Label distribution over word_match_share', fontsize=15)
    plt.xlabel('word_match_share', fontsize=15)
    plt.show()

def get_xgb_params():
    params = {
            'num_rounds':1760000,
            "objective": "binary:logistic",
            "booster": "gbtree",
            "eval_metric": 'logloss',
            'seed':1,
            'silent': 1,
            "max_depth": 4,
            'eta': 0.02,
           # "subsample": 0.6,
           # "alpha": 0,
           # "lambda": 1,
           # "colsample_bytree": 0.7,
           # 'min_child_weight': 15,
            }
    return params

def rebalancing_data(x_train, y_train):
    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]

    # Now we oversample the negative class
    # There is likely a much more elegant way to do this...
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print(len(pos_train) / (len(pos_train) + len(neg_train)))

    x_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train
    return x_train, y_train

def add_features(df):
    x_df = pd.DataFrame()
    x_df['word_match'] = df.apply(word_match_share, axis=1, raw=True)   
    x_df['tfidf_word_match'] = df.apply(tfidf_word_match_share, axis=1, raw=True)  
    return x_df

if __name__ == '__main__': 
    df_train, df_test = load_data()
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
   
    x_train = add_features(df_train)
    x_test = add_features(df_test)

    #plot_distribution_word_match(df_train)
    y_train = df_train['is_duplicate'].values
    x_train, y_train = rebalancing_data(x_train, np.array(y_train))

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    params = get_xgb_params()
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    bst = xgb.train(params, d_train, 680, watchlist, early_stopping_rounds=50, verbose_eval=10)

    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)

    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('simple_xgb_better.csv', index=False)
