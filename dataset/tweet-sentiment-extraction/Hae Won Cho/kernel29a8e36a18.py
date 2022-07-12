# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
import string
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.metrics import accuracy_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

train = train.drop(314) 

train['text'] = train['text'].str.lower()
test['text'] = test['text'].str.lower()





def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

adding_stop_words = ['i','to','the','a','my','you','and','it','is','in','for','im',
                     'of','me','on','so','have','that','be','its','with','day','at',
                     'was','...','..','from','was','were','just','almost','also','mile','miles','yup']

ENGLISH_STOP_WORDS = set([ 'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 
                          'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 
                          'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 
                          'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 
                          'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldn', 'couldnt', 'cry', 
                          'd', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 
                          'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 
                          'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 
                          'front', 'full', 'further', 'get', 'give', 'go', 'had', 'hadn', 'has', 'hasn', 'hasnt', 'have', 'haven', 'having', 'he', 'hence', 'her', 'here', 
                          'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 
                          'indeed', 'interest', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'll', 'ltd', 'm', 
                          'ma', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mightn', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'mustn', 
                          'my', 'myself', 'name', 'namely', 'needn', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 
                          'now', 'nowhere', 'o', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 
                          'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 
                          'shan', 'she', 'should', 'shouldn', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 
                          'somewhere', 'still', 'such', 'system', 't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 
                          'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 
                          'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 've', 'very', 'via', 'was', 
                          'wasn', 'we', 'well', 'were', 'weren', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 
                          'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'won', 'would', 
                          'wouldn', 'y', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves' ])
my_stop_words = ENGLISH_STOP_WORDS.union(adding_stop_words)


vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2,
        stop_words=my_stop_words,
        preprocessor=clean_text,
        max_features = 3000
    )

X = vectorizer.fit_transform(train['text'].values.astype('U'))
X = X.toarray()


train_rows_pos = train['sentiment'].str.count("positive")
train_rows_neg = train['sentiment'].str.count("negative")
train_rows_neu = train['sentiment'].str.count("neutral")
num_pos = train_rows_pos.sum(axis = 0, skipna = True)
num_neg = train_rows_neg.sum(axis = 0, skipna = True)
num_neu = train_rows_neu.sum(axis = 0, skipna = True)
p_neg = num_neg/27480
p_pos = num_pos/27480
P_neu = num_neu/27480

train_rows_pos = np.array([train_rows_pos])
pos_word_v = np.dot(train_rows_pos, X)
pos_total_words = pos_word_v.sum()
#print(pos_total_words)
pos_total_words += X.shape[1]

train_rows_neg = np.array([train_rows_neg])
neg_word_v = np.dot(train_rows_neg, X)
neg_total_words = neg_word_v.sum()
#print(neg_total_words)
neg_total_words += X.shape[1]

train_rows_neu = np.array([train_rows_neu])
neu_word_v = np.dot(train_rows_neu, X)
neu_total_words = neu_word_v.sum()
#print(neg_total_words)
neu_total_words += X.shape[1]


pos_word_v = (pos_word_v + 1)
neg_word_v = (neg_word_v + 1)
neu_word_v = (neu_word_v + 1)

pos_P_word_v = np.divide(pos_word_v, pos_total_words)
neg_P_word_v = np.divide(neg_word_v, neg_total_words)
neu_P_word_v = np.divide(neu_word_v, neu_total_words)




pos_words = {}
neu_words = {}
neg_words = {}


for k,i in zip(vectorizer.get_feature_names(),range(X.shape[1])):
    pos_words[k] = pos_P_word_v[0][i] 
    neg_words[k] = neg_P_word_v[0][i] 
    neu_words[k] = neu_P_word_v[0][i] 
    

neg_words_adj = {}
pos_words_adj = {}
neu_words_adj = {}

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neu_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neu_words[key] + neg_words[key])
    
for key, value in neu_words.items():
    neu_words_adj[key] = neu_words[key] - (neg_words[key] + pos_words[key])

    
    
def calculate_selected_text(df_row, tol = 0):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    
    if(sentiment == 'neutral'):
        return tweet
    
    elif(sentiment == 'positive'):
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
    elif(sentiment == 'negative'):
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    
    
    for i in range(len(subsets)):
        
        new_sum = 0 # Sum for the current substring
        
        # Calculate the sum of weights for each word in the substring
        for p in range(len(lst[i])):
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
            
        # If the sum is greater than the score, update our current selection
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
    return ' '.join(selection_str)


pd.options.mode.chained_assignment = None

tol = 0.001


for index, row in test.iterrows():
    
    selected_text = calculate_selected_text(row, tol)
    
    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text


sample.to_csv('submission.csv', index = False)