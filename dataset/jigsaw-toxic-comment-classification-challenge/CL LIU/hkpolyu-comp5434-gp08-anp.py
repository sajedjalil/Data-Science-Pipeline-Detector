"""
Kaggle : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
COMP5434 : http://www4.comp.polyu.edu.hk/~cskchung/COMP5434/Competition.html

Ultimate guide to deal with Text Data (using Python)
https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
"""

############################
# IMPORT LIBRARIES
############################

import numpy as np
import pandas as pd

# For Data Cleaning
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('punkt')
#nltk.download('stopwords')
from sklearn.feature_extraction import text as sklearn_text

# For Feature Extraction
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# For Model Building
from sklearn.linear_model import LogisticRegression

# For Model Evaluation
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

############################
# DATA IMPORT
############################

base_path_input = '../input/'
#base_path_input = ''

print('### Import data ###')
train = pd.read_csv(base_path_input + 'jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv(base_path_input + 'jigsaw-toxic-comment-classification-challenge/test.csv')
sample_submission = pd.read_csv(base_path_input + 'jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
 
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_text = train['comment_text'].fillna(' ')
test_text = test['comment_text'].fillna(' ')

train_text[:20]

############################
# COMMON WORD REPLACEMENT
############################

# Build Dictionarys
# replacements is a dictionary loaded from https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view 
replacement = { "aren't" : "are not",
                "can't" : "cannot",
                "couldn't" : "could not",
                "didn't" : "did not",
                "doesn't" : "does not",
                "don't" : "do not",
                "hadn't" : "had not",
                "hasn't" : "has not",
                "haven't" : "have not",
                "he'd" : "he would",
                "he'll" : "he will",
                "he's" : "he is",
                "i'd" : "I would",
                "i'd" : "I had",
                "i'll" : "I will",
                "i'm" : "I am",
                "isn't" : "is not",
                "it's" : "it is",
                "it'll":"it will",
                "i've" : "I have",
                "let's" : "let us",
                "mightn't" : "might not",
                "mustn't" : "must not",
                "shan't" : "shall not",
                "she'd" : "she would",
                "she'll" : "she will",
                "she's" : "she is",
                "shouldn't" : "should not",
                "that's" : "that is",
                "there's" : "there is",
                "they'd" : "they would",
                "they'll" : "they will",
                "they're" : "they are",
                "they've" : "they have",
                "we'd" : "we would",
                "we're" : "we are",
                "weren't" : "were not",
                "we've" : "we have",
                "what'll" : "what will",
                "what're" : "what are",
                "what's" : "what is",
                "what've" : "what have",
                "where's" : "where is",
                "who'd" : "who would",
                "who'll" : "who will",
                "who're" : "who are",
                "who's" : "who is",
                "who've" : "who have",
                "won't" : "will not",
                "wouldn't" : "would not",
                "you'd" : "you would",
                "you'll" : "you will",
                "you're" : "you are",
                "you've" : "you have",
                "'re": " are",
                "wasn't": "was not",
                "we'll":" will",
                "didn't": "did not"
              }

replacement.update({"im" : "i am", "youre" : "you are", "ur" : "you are",
                    "theyre" : "they are", "pls" : "please", "fk" : "fuck"})


print('\n#### Data Cleaning ####')

def replace_comment(comment):
    comment=comment.lower()
    
    # Replace words like gooood to good
    comment = re.sub(r'(\w)\1{2,}', r'\1\1', comment)
    
    # Normalize common abbreviations
    words=comment.split(' ')
    words=[replacement[word] if word in replacement else word for word in words]

    comment_repl=" ".join(words)
    return comment_repl

# Lower the case and replace common abbreviation
train_text = train_text.apply(lambda x: replace_comment(x))
test_text = test_text.apply(lambda x: replace_comment(x))

############################
# DATA CLEANING
############################

# For checking Regexp: https://regex101.com/
def standardize_text(datafile):
    datafile = datafile.str.lower()
    # Remove website link
    datafile = datafile.str.replace(r"http\S+", "")
    datafile = datafile.str.replace(r"https\S+", "")
    datafile = datafile.str.replace(r"http", "")
    datafile = datafile.str.replace(r"https", "")
    # Remove name tag
    datafile = datafile.str.replace(r"@\S+", "")
    # Remove time related text
    datafile = datafile.str.replace(r'\w{3}[+-][0-9]{1,2}\:[0-9]{2}\b', "") # e.g. UTC+09:00
    datafile = datafile.str.replace(r'\d{1,2}\:\d{2}\:\d{2}', "")            # e.g. 18:09:01
    datafile = datafile.str.replace(r'\d{1,2}\:\d{2}', "")                  # e.g. 18:09
    # Remove date related text
        # e.g. 11/12/19, 11-1-19, 1.12.19, 11/12/2019  
    datafile = datafile.str.replace(r'\d{1,2}(?:\/|\-|\.)\d{1,2}(?:\/|\-|\.)\d{2,4}', "")
        # e.g. 11 dec, 2019   11 dec 2019   dec 11, 2019
    datafile = datafile.str.replace(r"([\d]{1,2}\s(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s[\d]{1,2})(\s|\,|\,\s|\s\,)[\d]{2,4}", "")
        # e.g. 11 december, 2019   11 december 2019   december 11, 2019
    datafile = datafile.str.replace(r"[\d]{1,2}\s(january|february|march|april|may|june|july|august|september|october|november|december)(\s|\,|\,\s|\s\,)[\d]{2,4}", "")
        # Remove line breaks
    datafile = datafile.str.replace("\r"," ")
    datafile = datafile.str.replace("\n"," ")
    # Remove special characters
    datafile = datafile.str.replace(r"[^A-Za-z0-9(),.!?@\`\"\_ ]", "")
    datafile = datafile.str.replace(' "" ','')
    # Remove phone number and IP address
    datafile = datafile.str.replace(r'\d{8,}', "")
    datafile = datafile.str.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "")
    # Adjust common abbreviation
    datafile = datafile.str.replace(r" you re ", " you are ")
    datafile = datafile.str.replace(r" we re ", " we are ")
    datafile = datafile.str.replace(r" they re ", " they are ")
    datafile = datafile.str.replace(r"@", "at")
    return datafile

# Use regular expressions to clean up pour data.
train_text = standardize_text(train_text)
test_text = standardize_text(test_text)

############################
# STOP WORD REMOVAL
############################
stopwords_list = nltk.corpus.stopwords.words('english') # stopwords from nltk

# Exclude from stopwords: not, cannot
stopwords_list_rev = list(filter(lambda x: x not in ('not','cannot'), stopwords_list)) 
stopwords_list_rev.sort()

train_text = train_text.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords_list_rev))
test_text = test_text.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords_list_rev))

train_text[:20]

############################
# CHECKING COMMON WORDS
############################

""" Revised method based on:
    Ref 1: https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
    Ref 2: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
"""

# Word 1-gram
word_vectorizer_1 = TfidfVectorizer(sublinear_tf=True,
                                    strip_accents='unicode',
                                    analyzer='word',
                                    token_pattern=r'\w{1,}',
                                    stop_words='english',
                                    ngram_range=(1, 1),
                                    max_features=10000
                                    )
# Word 2-gram
word_vectorizer_2 = TfidfVectorizer(sublinear_tf=True,
                                    strip_accents='unicode',
                                    analyzer='word',
                                    token_pattern=r'\w{1,}',
                                    stop_words=stopwords_list_rev,
                                    ngram_range=(2, 2),
                                    max_features=5000
                                    )
# Char 2 to 6-gram
char_vectorizer = TfidfVectorizer(  sublinear_tf = True,
                                    strip_accents = 'unicode',
                                    analyzer = 'char',
                                    stop_words = 'english',
                                    ngram_range = (2, 6),
                                    max_features = 50000
                                 )

# Fit vectorizer by all text
all_text = pd.concat([train_text, test_text])
print('\n### Vectorizer fitting ###')
word_vectorizer_1.fit(all_text)
word_vectorizer_2.fit(all_text)
char_vectorizer.fit(all_text)

# Transform dataset to document-term matrix
print('\n### DTM Transforming (train) ###')
train_word_features1 = word_vectorizer_1.transform(train_text)
train_word_features2 = word_vectorizer_2.transform(train_text)
train_char_features  = char_vectorizer.transform(train_text)
print('\n### DTM Transforming (test) ###')
test_word_features1 = word_vectorizer_1.transform(test_text)
test_word_features2 = word_vectorizer_2.transform(test_text)
test_char_features  = char_vectorizer.transform(test_text)

# Merge features
print('\n### Merging Features Martix ###')
train_features = hstack([train_word_features1, train_word_features2, train_char_features])
test_features = hstack([test_word_features1, test_word_features2, test_char_features])



############################
# BASE MODELS
############################

""" model 1 : Simple Logistic Regression (Well Calibrated)
    https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
    # 0.9792
""" 
scores = []

LR = pd.DataFrame.from_dict({'id': sample_submission['id']}).sort_values('id')

print('\n### Model 1: Simple Logistic Regression ###')
for label in labels:
    pred_model = LogisticRegression(C=0.1, solver='sag')
    #AUC
    score = np.mean(cross_val_score( pred_model, train_features, train[label], cv=3, scoring='roc_auc'))
    scores.append(score)
    print('For {}, AUC is {}.'.format(label, score))
    
    pred_model.fit(train_features, train[label])
    LR[label] = pred_model.predict_proba(test_features)[:, 1]
    
print('\nOverall CV score is {}'.format(np.mean(scores)))


####################################
# IMPORT OTHER HIGH SCORE MODELS
####################################

""" Ref
GRU ( Gated Recurrent Unit Networks ):
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

LSTM ( Long Short Term Memory Networks ):
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

LGBM ( Light(High Speed) Gradient Boosting Framework ): - can easily overfit small data with < 10000 rows
https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
"""

""" INSERT THE DATASET FROM KERNEL BEFORE RUNNIG BELOW'S PART """

''' Gated Recurrent Unit (GRU) / Long and Short Term Memory Unit (LSTM) '''
# https://www.kaggle.com/yekenot/pooled-gru-fasttext
gru_lstm = pd.read_csv(base_path_input + 'pooled-gru-fasttext/submission.csv') # PL score 0.9829

# https://www.kaggle.com/jhoward/minimal-lstm-nb-svm-baseline-ensemble
lstm_nb_svm = pd.read_csv(base_path_input + 'minimal-lstm-nb-svm-baseline-ensemble/submission.csv') # 0.9811

''' Light Gradient Boosting Framework '''
# https://www.kaggle.com/peterhurford/lightgbm-with-select-k-best-on-tfidf
lgbm = pd.read_csv(base_path_input + 'lightgbm-with-select-k-best-on-tfidf/lgb_submission.csv') # 0.9785

# Blended Data: 0.9852


############################
# Testing Correlations
############################

print('\n#### Testing Correlations ####')
for label in labels:
    r = spearmanr(lstm_nb_svm[label],gru_lstm[label])
    print( label + " - correlation: {:.5f}, pvalue:{:.3f}".format(r.correlation, r.pvalue) )
    
############################
# DATA BLENDING
############################
# https://www.kaggle.com/reppic/lazy-ensembling-algorithm

# Controls weights when combining predictions
# 0: equal average of all inputs; 1: up to 50% of weight going to least correlated input
DENSITY_COEFF = 0
assert DENSITY_COEFF >= 0.0 and DENSITY_COEFF <= 1.0

# When merging 2 files with corr > OVER_CORR_CUTOFF 
# the result's weight is the max instead of the sum of the merged files' weights
OVER_CORR_CUTOFF = 0.98
assert OVER_CORR_CUTOFF >= 0.0 and OVER_CORR_CUTOFF <= 1.0

###############################################

def load_submissions():
    csv_files = {'gru': base_path_input + 'pooled-gru-fasttext/submission.csv',
                 'lstm': base_path_input + 'minimal-lstm-nb-svm-baseline-ensemble/submission.csv',
                 'lgbm': base_path_input + 'lightgbm-with-select-k-best-on-tfidf/lgb_submission.csv'
                }
    frames = { f:pd.read_csv(f).sort_values('id') for f in csv_files.values() }
    models = [ m for m in csv_files.keys() ]
    data = dict(zip(models, frames.values()))
    # Adding LR model to import models
    data = dict(data,**{'LR': LR})
    del frames
    return data

def get_corr_mat(frames, label):
    c = pd.DataFrame()
    for datafile, values in frames.items():
        c[datafile] = values[label]
    cor = c.corr()
    
    # Set the diagonal correlation to zero for merging
    for index, name in enumerate(cor):
        cor.iat[index,index] = 0.0
    del c
    return cor


def highest_corr(mat):
    n_cor = np.array(mat.values)
    corr = np.max(n_cor)
    idx = np.unravel_index(np.argmax(n_cor, axis=None), n_cor.shape)
    f1 = mat.columns[idx[0]]
    f2 = mat.columns[idx[1]]
    return corr,f1,f2


def get_merge_weights(m1,m2,densities):
    d1 = densities[m1]
    d2 = densities[m2]
    d_tot = d1 + d2
    weights1 = 0.5*DENSITY_COEFF + (d1/d_tot)*(1-DENSITY_COEFF)
    weights2 = 0.5*DENSITY_COEFF + (d2/d_tot)*(1-DENSITY_COEFF)
    return weights1, weights2


def ensemble_col(label,frames,densities):
    if len(frames) == 1:
        model, value = frames.popitem() # Pop the last item
        return value[label]
    else:
        corr_mat = get_corr_mat(frames, label)
        
        corr, merge1, merge2 = highest_corr(corr_mat)
        w1,w2 = get_merge_weights(merge1,merge2,densities)
        
        comb_model = pd.DataFrame()
        comb_model[label] = (frames[merge1][label]*w1) + (frames[merge2][label]*w2)
    
        comb_col = merge1 + '_' + merge2
        frames[comb_col] = comb_model
    
        if corr >= OVER_CORR_CUTOFF:
            print('\t',merge1,merge2,'  (OVER CORR)')
            densities[comb_col] = max(densities[merge1],densities[merge2])
        else:
            densities[comb_col] = densities[merge1] + densities[merge2]
        
        del frames[merge1]
        del frames[merge2]
        del densities[merge1]
        del densities[merge2]
        return ensemble_col(label, frames, densities)

print('\n#### Data Blending ####')

final_submission = pd.DataFrame.from_dict({'id': sample_submission['id']}).sort_values('id')

for label in labels:
    frames = load_submissions()
    densities = { k: 1.0 for k in frames.keys() }   # Pre-set density as 1 to all models
    
    print('\n\n # ', label)
    final_submission[label] = ensemble_col(label, frames, densities)
    
############################
# Output
############################

final_submission.to_csv('submission.csv',index=False)