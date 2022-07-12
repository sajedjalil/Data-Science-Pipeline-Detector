import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import scipy
import xgboost as xgb
import difflib
import re
from nltk.corpus import stopwords
from nltk.metrics import jaccard_distance

#Reading and processing of data
train=pd.read_csv('../input/train.csv').fillna("")
#train=pd.read_csv('../input/train.csv').dropna()
stops = set(stopwords.words("english"))
y=train['is_duplicate']
train=train.drop(['id', 'qid1', 'is_duplicate','qid2'], axis=1)

#Cleaning up the data
#Removing ? mark and non ASCII characters
def cleanup(data):
    data['question1'] = data['question1'].apply(lambda x: x.rstrip('?'))
    data['question2'] = data['question2'].apply(lambda x: x.rstrip('?'))
    # Removing non ASCII chars
    data['question1']=data['question1'].apply(lambda x: x.replace(r'[^\x00-\x7f]',r' '))
    data['question2']=data['question2'].apply(lambda x: x.replace(r'[^\x00-\x7f]',r' ')) 
    # Pad punctuation with spaces on both sides
    '''
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        x = x.replace(char, ' ' + char + ' ')
    '''
    contractions = {
      "ain't": "am not",  "aren't": "are not",  "can't": "cannot",  "can't've": "cannot have",  "'cause": "because",  "could've": "could have",  "couldn't": "could not",
      "couldn't've": "could not have",  "didn't": "did not",  "doesn't": "does not",  "don't": "do not",  "hadn't": "had not",  "hadn't've": "had not have","hasn't": "has not",  "haven't": "have not",  "he'd": "he would",  "he'd've": "he would have",  "he'll": "he will",  "he'll've": "he will have",  "he's": "he is",  "how'd": "how did",
      "how'd'y": "how do you",  "how'll": "how will",  "how's": "how is",  "I'd": "I would",  "I'd've": "I would have",  "I'll": "I will",  "I'll've": "I will have",  "I'm": "I am",
      "I've": "I have",  "isn't": "is not",  "it'd": "it had",  "it'd've": "it would have",  "it'll": "it will",  "it'll've": "it will have",  "it's": "it is",  "let's": "let us","ma'am": "madam",  "mayn't": "may not",  "might've": "might have",  "mightn't": "might not",  "mightn't've": "might not have",  "must've": "must have",  "mustn't": "must not",
      "mustn't've": "must not have",  "needn't": "need not",  "needn't've": "need not have",  "o'clock": "of the clock",  "oughtn't": "ought not",  "oughtn't've": "ought not have",      "shan't": "shall not",  "sha'n't": "shall not",  "shan't've": "shall not have",  "she'd": "she would",  "she'd've": "she would have",  "she'll": "she will",  "she'll've": "she will have",
      "she's": "she is",  "should've": "should have",  "shouldn't": "should not",  "shouldn't've": "should not have",  "so've": "so have",  "so's": "so is",  "that'd": "that would",  "that'd've": "that would have",
      "that's": "that is",  "there'd": "there had",  "there'd've": "there would have",  "there's": "there is",  "they'd": "they would",  "they'd've": "they would have",      "they'll": "they will",  "they'll've": "they will have",  "they're": "they are",  "they've": "they have",  "to've": "to have",  "wasn't": "was not",  "we'd": "we had",  "we'd've": "we would have",
      "we'll": "we will",  "we'll've": "we will have",  "we're": "we are",  "we've": "we have",  "weren't": "were not",  "what'll": "what will",  "what'll've": "what will have",      "what're": "what are",  "what's": "what is",  "what've": "what have",  "when's": "when is",  "when've": "when have",  "where'd": "where did",  "where's": "where is",  "where've": "where have",
      "who'll": "who will",  "who'll've": "who will have",  "who's": "who is",  "who've": "who have",  "why's": "why is",  "why've": "why have",  "will've": "will have",  "won't": "will not",      "won't've": "will not have",  "would've": "would have",  "wouldn't": "would not",  "wouldn't've": "would not have",  "y'all": "you all",  "y'alls": "you alls",  "y'all'd": "you all would",  "y'all'd've": "you all would have",
      "y'all're": "you all are",  "y'all've": "you all have",  "you'd": "you had",  "you'd've": "you would have",  "you'll": "you you will",  "you'll've": "you you will have",  "you're": "you are",  "you've": "you have"
      }
    contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
    def expand_contractions(s, contractions_dict=contractions):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)     
    data['question1']=data['question1'].apply(lambda x: expand_contractions(x.lower()) )
    data['question2']=data['question2'].apply(lambda x: expand_contractions(x.lower()) )
    return data
train=cleanup(train)

def build_dict(sentences):
    #Dictionary of train words --> word index: word freq
    print('Building dictionary using train words..')
    wordcount = dict()
    #For each worn in each sentence, cummulate frequency
    for ss in sentences:
        for w in ss:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1    
    worddict = dict()
    for idx, w in enumerate(sorted(wordcount.items(), key = lambda x: x[1], reverse=True)):
        worddict[w[0]] = idx+2  # leave 0 and 1 (UNK)
    return worddict, wordcount

def generate_sequence(sentences, dictionary):
    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in ss]
    return seqs

def tokenize(x):
    return x.lower().split()

questions = train['question1'].tolist() + train['question2'].tolist()
tok_questions = [tokenize(s) for s in questions]
worddict, wordcount = build_dict(tok_questions)
print(np.sum(list(wordcount.values())), ' total words ', len(worddict), ' unique words')

#Metrics for sentence comparison
def jc(x):
    return jaccard_distance(set(x['Q1seq']),set(x['Q2seq']))

def cosine_d(x):
    a = set(x['Q1seq'])
    b = set(x['Q2seq'])
    d = len(a)*len(b)
    if (d == 0):
        return 0
    else: 
        return len(a.intersection(b))/d

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.quick_ratio()

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
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf.fit_transform(questions)

def get_features(df_features):    
    #Question length
    print('question lengths....')
    df_features['Qlen1'] = df_features.question1.map(lambda x: len(str(x)))
    df_features['Qlen2'] = df_features.question2.map(lambda x: len(str(x)))
    df_features['diffQlen'] = df_features['Qlen1']-df_features['Qlen2']    
    #Question number of words
    df_features['Qwords1'] = df_features.question1.map(lambda x: len(str(x).split()))
    df_features['Qwords2'] = df_features.question2.map(lambda x: len(str(x).split()))
    df_features['diffQword'] = df_features['Qwords1'] -df_features['Qwords2']    
    print('jaccard...')
    df_features['Q1seq'] = generate_sequence(df_features['question1'].apply(tokenize),worddict)
    df_features['Q2seq'] = generate_sequence(df_features['question2'].apply(tokenize),worddict)
    df_features['jaccard'] = df_features.apply(jc,axis = 1)    
    print('cosine....')
    df_features['cosine'] = df_features.apply(cosine_d,axis = 1)        
    #matching the sequences
    print('difflib...')
    df_features['SeqMatchRatio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
    #percentage of common words in both questions
    print('word match...')    
    df_features['WordMatch'] = df_features.apply(word_match_share, axis=1, raw=True)    
    print('tfidf...')
    question1_tfidf = tfidf.transform(df_features.question1.tolist())
    question2_tfidf = tfidf.transform(df_features.question2.tolist())    
    df_features['tfidfSum1'] = scipy.sparse.csr_matrix(question1_tfidf).sum(axis=1)
    df_features['tfidfSum2'] = scipy.sparse.csr_matrix(question2_tfidf).sum(axis=1)
    df_features['tfidfMean1'] = scipy.sparse.csr_matrix(question1_tfidf).mean(axis=1)
    df_features['tfidfMean2'] = scipy.sparse.csr_matrix(question2_tfidf).mean(axis=1)
    df_features['tfidfLen1'] = (question1_tfidf != 0).sum(axis = 1)
    df_features['tfidfLen2'] = (question2_tfidf != 0).sum(axis = 1)
    #Exactly same questions
    df_features['exactly_same'] = (df_features['question1'] == df_features['question2']).astype(int)
    return df_features.fillna(0.0)

df_train = get_features(train)
feats = df_train.columns.values.tolist()
feats=[x for x in feats if x not in ['question1','question2','Q1seq','Q2seq','id','qid1','qid2','is_duplicate']]
print("features",feats)

x_train, x_valid, y_train, y_valid = train_test_split(df_train[feats], y, test_size=0.3, random_state=0)
#XGBoost model
params = {"objective":"binary:logistic",'eval_metric':'logloss',"eta": 0.11,
          "subsample":0.7,"min_child_weight":1,"colsample_bytree": 0.7,
          "max_depth":5,"silent":1,"seed":2017}

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50,verbose_eval=100) #change to higher #s
print('training done')

print("log loss for training data set",log_loss(y, bst.predict(xgb.DMatrix(df_train[feats]))))
#Predicting for test data set
sub = pd.DataFrame() # Submission data frame
sub['test_id'] = []
sub['is_duplicate'] = []
header=['test_id','question1','question2','id','qid1','qid2','is_duplicate']
test=pd.read_csv('../input/test.csv').fillna("")
print("cleaning test")
df_test=cleanup(test)
print("feature engineering for test")
df_test = get_features(df_test)
sub=pd.DataFrame({'test_id':df_test['test_id'], 'is_duplicate':bst.predict(xgb.DMatrix(df_test[feats]))})
sub.to_csv('quora_submission_xgb_11.csv', index=False)
