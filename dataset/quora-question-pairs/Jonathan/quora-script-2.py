import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from difflib import SequenceMatcher

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


## Training set
train_df = pd.read_csv('../input/train.csv')

## Test Set",
test_df = pd.read_csv('../input/test.csv')

print(train_df.shape[0],test_df.shape[0])



def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R



from collections import Counter
# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def stop_ratio(question):
    q = set(question)
    if len(q) == 0:
        return 0
    qwords = q.difference(stops)
    qstops = q.intersection(stops)
    return len(qstops) / len(q)

def uniq1_ratio(row):
    uniq_1 = set(row["question1"].lower())
    uniq_2 = set(row["question2"].lower())
    return len(uniq_1) / len(uniq_1 | uniq_2)

def uniq2_ratio(row):
    uniq_1 = set(row["question1"].lower())
    uniq_2 = set(row["question2"].lower())
    return len(uniq_2) / len(uniq_1 | uniq_2)

def create_features(df):
    df["question1"].fillna("", inplace=True)
    df["question2"].fillna("", inplace=True)

    df["question1"] = df["question1"].apply(str)
    df["question2"] = df["question2"].apply(str)
    
    print("len")
    df["q1_len"] = df["question1"].apply(len)
    df["q2_len"] = df["question1"].apply(len)
    df["diff_len"] = abs(df["q1_len"] - df["q2_len"])
    
    print("len word")
    df["q1_len_word"] = df["question1"].apply(lambda x: len(x.split()))
    df["q2_len_word"] = df["question1"].apply(lambda x: len(x.split()))
    df["diff_len_word"] = abs(df["q1_len_word"] - df["q2_len_word"])
    
    print("avg len word")
    df['q1_avg_len_word'] = df['q1_len'] / df['q1_len_word']
    df['q2_avg_len_word'] = df['q2_len'] / df['q2_len_word']
    df['diff_avg_len_word'] = abs(df['q1_avg_len_word'] - df['q2_avg_len_word'])
    
    print("n unique char")
    df["q1_n_uniquechar"] = df["question1"].apply(lambda x: len("".join(set(x.replace(" ","")))))
    df["q2_n_uniquechar"] = df["question2"].apply(lambda x: len("".join(set(x.replace(" ","")))))
    df["diff_n_uniquechar"] = abs(df["q1_n_uniquechar"] - df["q2_n_uniquechar"])

    print("W words")
    df["q1_how"]   = df["question1"].apply(lambda x : "how"   in x.lower())
    df["q1_who"]   = df["question1"].apply(lambda x : "who"   in x.lower())
    df["q1_why"]   = df["question1"].apply(lambda x : "why"   in x.lower())
    df["q1_what"]  = df["question1"].apply(lambda x : "what"  in x.lower())
    df["q1_where"] = df["question1"].apply(lambda x : "where" in x.lower())
    df["q1_which"] = df["question1"].apply(lambda x : "which" in x.lower())

    df["q2_how"]   = df["question2"].apply(lambda x : "how"   in x.lower())
    df["q2_who"]   = df["question2"].apply(lambda x : "who"   in x.lower())
    df["q2_why"]   = df["question2"].apply(lambda x : "why"   in x.lower())
    df["q2_what"]  = df["question2"].apply(lambda x : "what"  in x.lower())
    df["q2_where"] = df["question2"].apply(lambda x : "where" in x.lower())
    df["q2_which"] = df["question2"].apply(lambda x : "which" in x.lower())
    
    df["q1q2_how"]   = df["q1_how"]   == df["q2_how"]
    df["q1q2_who"]   = df["q1_who"]   == df["q2_who"]
    df["q1q2_why"]   = df["q1_why"]   == df["q2_why"]
    df["q1q2_what"]  = df["q1_what"]  == df["q2_what"]
    df["q1q2_where"] = df["q1_where"] == df["q2_where"]
    df["q1q2_which"] = df["q1_which"] == df["q2_which"]
    
    print("stop ratio")
    df["q1_stop_ratio"] = df["question1"].apply(stop_ratio)
    df["q2_stop_ratio"] = df["question2"].apply(stop_ratio)
    df["diff_stop_ratio"] = abs(df["q1_stop_ratio"] - df["q2_stop_ratio"])

    print("math")
    df["q1_math"] = df["question1"].apply(lambda x: '[math]' in x)
    df["q2_math"] = df["question2"].apply(lambda x: '[math]' in x)
    
    print("n qmark")
    df["q1_nqmark"] = df["question1"].apply(lambda x: x.count('?'))
    df["q2_nqmark"] = df["question2"].apply(lambda x: x.count('?'))
    df["diff_nqmark"] = abs(df["q1_nqmark"] - df["q2_nqmark"])
    
    print("n period")
    df["q1_nperiod"] = df["question1"].apply(lambda x: x.count('.'))
    df["q2_nperiod"] = df["question2"].apply(lambda x: x.count('.'))
    df["diff_nperiod"] = abs(df["q1_nperiod"] - df["q2_nperiod"])

    print("capital first")
    df["q1_capitalfirst"] = df["question1"].apply(lambda x: x[0].isupper() if len(x) > 0 else False)
    df["q2_capitalfirst"] = df["question2"].apply(lambda x: x[0].isupper() if len(x) > 0 else False)
    df["q1q2_capitalfirst"] = df["q1_capitalfirst"] == df["q2_capitalfirst"]

    print("has capital")
    df["q1_has_capital"] = df["question1"].apply(lambda x: any([l.isupper() for l in x]))
    df["q2_has_capital"] = df["question2"].apply(lambda x: any([l.isupper() for l in x]))
    df["q1q2_has_capital"] = df["q1_has_capital"] == df["q2_has_capital"]

    print("n capitals")
    df["q1_n_capitals"] = df["question1"].apply(lambda x: sum([1 for c in x if c.isupper()]))
    df["q2_n_capitals"] = df["question2"].apply(lambda x: sum([1 for c in x if c.isupper()]))
    df["diff_n_capitals"] = abs(df["q1_n_capitals"] - df["q2_n_capitals"])
    
    print("is identical")
    df["is_identical"] = (df["question1"].apply(lambda x: x.lower()) == df["question2"].apply(lambda x: x.lower()))    

    print("unique ratio")
    df["q1_unique_ratio"] = df.apply(uniq1_ratio ,axis=1)
    df["q2_unique_ratio"] = df.apply(uniq2_ratio ,axis=1)

    #print("similarity")
    #df["similarity_prob"] = df.apply(lambda row: SequenceMatcher(None, row["question1"],row["question2"]).ratio(),axis=1)



create_features(train_df)
create_features(test_df)

train_df.head()

plt.figure()
plt.hist(train_df[train_df["is_duplicate"]==0]["diff_n_uniquechar"],bins=100,range=(0,0.3),alpha=0.5,normed=True)
plt.hist(train_df[train_df["is_duplicate"]==1]["diff_n_uniquechar"],bins=100,range=(0,0.3),alpha=0.5,normed=True)
plt.savefig("uniqchar.png")