# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

##################################### nltk start #####################################
#sentence = """What is the step by step guide to invest in share market in india?."""
#
#tokens = nltk.word_tokenize(sentence)
#print(tokens)
#
#tagged = nltk.pos_tag(tokens)
#print(tagged)

##################################### nltk end #####################################

df_train = pd.read_csv('../input/train.csv').head(2000)
#print(df_train.head())

df_test = pd.read_csv('../input/test.csv').head(100)
#sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': 0.38888888})
#sub.to_csv('test_submission.csv', index=False)
#sub.head()

############################ 



def spec_word_compare(row):
#    matchObj1 = re.search( r'([0-9]+)', str(row['question1']) , re.M|re.I)
#    matchObj2 = re.search( r'([0-9]+)', str(row['question2']) , re.M|re.I)
#    if matchObj1 and matchObj2:
#        return (int(matchObj1.group(1))+0.1)/(int(matchObj2.group(1))+0.1)
#    elif str(row['question1']).lower().split()[0] == str(row['question2']).lower().split()[0]:
#        return 1
#    else:
#        return 0
    word_wrb1 = ""
    word_nn1 = ""
    word_vb1 = ""
    sens = nltk.sent_tokenize(str(row['question1']).lower())
    words = []
    for sen in sens:
        words.append(nltk.word_tokenize(sen)) 
    tags = []
    for word in words:
        tags.append(nltk.pos_tag(word))
    for tag in tags[0]:
        if tag[1][0:3] == "WRB":
            word_wrb1 = tag[0]
        elif tag[1][0:2] == "VB":
            word_vb1 = tag[0]
        elif tag[1][0:2] == "NN":
            word_nn1 = tag[0]

    word_wrb2 = ""
    word_nn2 = ""
    word_vb2 = ""
    sens = nltk.sent_tokenize(str(row['question2']).lower())
    words = []
    for sen in sens:
        words.append(nltk.word_tokenize(sen)) 
    tags = []
    for word in words:
        tags.append(nltk.pos_tag(word))
    for tag in tags[0]:
        if tag[1][0:3] == "WRB":
            word_wrb2 = tag[0]
        elif tag[1][0:2] == "VB":
            word_vb2 = tag[0]
        elif tag[1][0:2] == "NN":
            word_nn2 = tag[0]
    match1 = 0
    match2 = 0
    match3 = 0
    if word_wrb1 == word_wrb2:
        match1 = 1
    if word_vb1 == word_vb2:
        match2 = 1
    if word_nn1 == word_nn2:
        match3 = 1
    return (match1 + match2 + match3)/3
train_spec_word = df_train.apply(spec_word_compare, axis=1, raw=True)
######################### train_spec_word #############################


#################### train_word_match #############################
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
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
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)


#################### tfidf_train_word_match #############################
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)

from collections import Counter
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
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
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_train['spec_word_compare'] = train_spec_word
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
x_test['spec_word_compare'] = df_test.apply(spec_word_compare, axis=1, raw=True)

y_train = df_train['is_duplicate'].values

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

from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

############################### XGBoost Start #############################################
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('whiplash.csv', index=False)

# Any results you write to the current directory are saved as output.