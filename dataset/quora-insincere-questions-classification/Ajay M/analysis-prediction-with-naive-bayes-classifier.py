# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


########### Library Imports #################
import pandas as pd
import os
import numpy as np
from nltk.tokenize	import	word_tokenize  
from nltk.corpus	import	stopwords
from nltk import ngrams
from collections import defaultdict
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
######################################################

########## Inputs ###############################

path = "../input"
num_of_max_values_to_per_ngram = 20 ## Display these many ngrams 
stopwords_list = stopwords.words('english')
thresholds = [0.1,0.2,0.5,0.6,0.7,0.8] ####### These are thresholds at which predictions need to be generated
####################################################


################ Function Definitions ##########################



def generate_tokenized_sentence_without_stopwords(sentence):
    """ Generates Tokenized Sentence removing stopwords """
    tokens = [w	for	w in	word_tokenize(sentence.lower()) if	w.isalpha()]            
    tokenized_sen =  [t for t in tokens if t not in stopwords_list]
    return tokenized_sen

def ngram(sentence,ngram_len):
    """ Return Ngram if ngram doesn't exist return empty list  """
    try:
        return(list(ngrams(sentence,ngram_len)))
    except RuntimeError:
        return([])


##################################################################





train_df = pd.read_csv(os.path.join(path,'train.csv'))#,nrows=250) ## currently load only 25 que for development
test_df = pd.read_csv(os.path.join(path,'test.csv'))#,nrows=250) ## currently load only 25 que for development



train_df['One_gram'] = train_df['question_text'].apply(lambda que: generate_tokenized_sentence_without_stopwords(que)) ### Get tokenized que
#train_df = train_df[['qid','target','One_gram']]


train_df['Two_gram'] = train_df['One_gram'].apply(lambda x: ngram(x,2))
train_df['Three_gram'] = train_df['One_gram'].apply(lambda x: ngram(x,3))


freq_count_df = []
for target_val in [0,1] :
    for ngram_val in ['One_gram','Two_gram','Three_gram']:
        freq_dict = defaultdict(int)
        for sentence in train_df[train_df['target']==target_val][ngram_val]:
            for token in sentence:
                freq_dict[token] += 1
        freq_df = pd.DataFrame.from_dict(freq_dict, orient='index',columns=['Count']).reset_index()
        freq_df= freq_df[['index','Count']].sort_values(by ='Count' ,ascending = False).head(num_of_max_values_to_per_ngram).reset_index(drop = True)
        freq_df['target'] = target_val 
        freq_df['Ngram'] = ngram_val
        freq_count_df.append(freq_df)

freq_count_df = pd.concat(freq_count_df)   



sns.set(style="whitegrid")
f, axes = plt.subplots(3,2,figsize=(20,30))

for target_val in [0,1] :
    row = 0
    for ngram_val in ['One_gram','Two_gram','Three_gram']:
        filtered_data = freq_count_df[(freq_count_df['target'] == target_val) & (freq_count_df['Ngram'] == ngram_val) ]
        g = sns.barplot(x="Count", y="index", data=filtered_data,ax = axes[row,target_val])
        axes[row,target_val].set_title("Target: %s   Ngram: %s"%(target_val,ngram_val))
        row +=1

plt.savefig('Ngram.png')




######################### Model Development ###############

X_train = train_df['question_text']
X_test  = test_df['question_text']
y_train = train_df['target']



############## Vectorize Input text ###############
count_vectorizer	=	CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)

############# Use Naive Bayes Classifier on Vectorized Input ###############
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train,y_train)
y_pred_prob = nb_classifier.predict_proba(count_test)[:,1]
thresh  = 0.4 ## This value was obtained by running Cross validation offline
y_pred = (y_pred_prob >= thresh).astype(int) ###### convert probabilities to 1 or 0 using thresholds

out_df = pd.DataFrame(data={'qid':test_df['qid'].values,'prediction': y_pred})
out_df.to_csv('submission.csv',index=False)



