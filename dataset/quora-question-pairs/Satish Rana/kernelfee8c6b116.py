# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import gensim
from gensim import corpora
#import nltk
#nltk.download('stopwords')
import matplotlib.pyplot as plt
import seaborn as sns
import time


df = pd.read_csv("../input/train.csv")
""" Check and fill missing values """
df.isnull().sum()
df.fillna('',inplace = True, axis = 1)
df.apply(lambda x: len(x.unique()))
print ("Duplicate Count = %s , Non Duplicate Count = %s"
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))
#sns.countplot(x="is_duplicate", hue="is_duplicate", data=df,palette="Set2") 

question_ids_combined = df.qid1.tolist() + df.qid2.tolist()
print ("Unique Questions = %s" %(len(np.unique(question_ids_combined))))

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def tokenize_questions(df):    
    df["Question_1_tok"] = df["question1"]
    df["Question_2_tok"] = df["question2"]
    df["Question_1_tok"] = df["Question_1_tok"].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x))
    df["Question_2_tok"] = df["Question_2_tok"].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x))
    df["Question_1_tok"] = df["Question_1_tok"].str.lower()
    df["Question_2_tok"] = df["Question_2_tok"].str.lower()        
    df["Question_1_tok"] = df["Question_1_tok"].apply(lambda x: [lemmatizer.lemmatize(item) for item in words.findall(x) if item not in stopword])
    df["Question_2_tok"] = df["Question_2_tok"].apply(lambda x: [lemmatizer.lemmatize(item) for item in words.findall(x) if item not in stopword])
    return df


def train_dictionary(df):
    questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()
    dictionary = corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=15, no_above=0.8)
    dictionary.compactify()
    return dictionary

df = tokenize_questions(df)
dictionary = train_dictionary(df)
print ("No of words in the dictionary = %s" %len(dictionary.token2id))

df_train, df_test = train_test_split(df, test_size = 0.25)

def get_vectors(df, dictionary):
    
    question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]
    
    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    
    return question1_csc.transpose(),question2_csc.transpose()


q1_csc, q2_csc = get_vectors(df_train, dictionary)

print (q1_csc.shape)
print (q2_csc.shape)

from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.preprocessing import MinMaxScaler

mms_scale_man = MinMaxScaler()
mms_scale_euc = MinMaxScaler()
mms_scale_mink = MinMaxScaler()

def get_similarity_values(q1_csc, q2_csc):
    cosine_sim = []
    manhattan_dis = []
    eucledian_dis = []
    jaccard_dis = []
    
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
            
      
    return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis

#log loos before model    
cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis = get_similarity_values(q1_csc, q2_csc)
eucledian_dis_array = np.array(eucledian_dis).reshape(-1,1)
manhattan_dis_array = np.array(manhattan_dis).reshape(-1,1)
    
manhattan_dis_array = mms_scale_man.fit_transform(manhattan_dis_array)
eucledian_dis_array = mms_scale_euc.fit_transform(eucledian_dis_array)

eucledian_dis = eucledian_dis_array.flatten()
manhattan_dis = manhattan_dis_array.flatten()

# log loss
from sklearn.metrics import log_loss

def calculate_logloss(y_true, y_pred):
    loss_cal = log_loss(y_true, y_pred)
    return loss_cal

# Logs loss of train data
y_true = df_train.is_duplicate.tolist()

logloss = calculate_logloss(y_true, cosine_sim)
print ("The calculated log loss value on the train set for cosine sim is = %f" %logloss)

logloss = calculate_logloss(y_true, manhattan_dis)
print ("The calculated log loss value on the train set for manhattan sim is = %f" %logloss)

logloss = calculate_logloss(y_true, eucledian_dis)
print ("The calculated log loss value on the train set for euclidean sim is = %f" %logloss)

logloss = calculate_logloss(y_true, jaccard_dis)
print ("The calculated log loss value on the train set for jaccard sim is = %f" %logloss)
q1_csc_test, q2_csc_test = get_vectors(df_test, dictionary)
y_pred_cos, y_pred_man, y_pred_euc, y_pred_jac = get_similarity_values(q1_csc_test, q2_csc_test)
y_true = df_test.is_duplicate.tolist()

y_pred_man_array = mms_scale_man.transform(np.array(y_pred_man).reshape(-1,1))
y_pred_man = y_pred_man_array.flatten()

y_pred_euc_array = mms_scale_euc.transform(np.array(y_pred_euc).reshape(-1,1))
y_pred_euc = y_pred_euc_array.flatten()

### log loss for test data
"""

logloss = calculate_logloss(y_true, y_pred_cos)
print ("The calculated log loss value on the test set for cosine sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_man)
print ("The calculated log loss value on the test set for manhattan sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_euc)
print ("The calculated log loss value on the test set for euclidean sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_jac)
print ("The calculated log loss value on the test set for jaccard sim is = %f" %logloss)
"""
### model building ########
X_train = pd.DataFrame({"cos" : cosine_sim, "man" : manhattan_dis, "euc" : eucledian_dis, "jac" : jaccard_dis})
y_train = df_train.is_duplicate

X_test = pd.DataFrame({"cos" : y_pred_cos, "man" : y_pred_man, "euc" : y_pred_euc, "jac" : y_pred_jac})
y_test = df_test.is_duplicate.tolist()


""" Trying to check accurace on different train/test set """
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# prepare all classisfier models
"""
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=10, metric= 'minkowski', p = 2)))
models.append(('DecisionTree', DecisionTreeClassifier(criterion ='entropy',random_state=0)))
models.append(('XGBoost', XGBClassifier()))

# evaluate each model in turn

for i in range(1):
    ## Iterate over all models and print accuracy 
    for name, model in models: 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print( "%s: %s %f" %(name, " has accuracy_score: ",accuracy_score(y_test, y_pred)))
        print( "%s: %s (%f)" %(name, " has precision_score: ",precision_score(y_test, y_pred)))
        print( "%s: %s (%f)" %(name, " has recall_score: ",recall_score(y_test, y_pred)))
        print( "%s: %s (%f)" %(name, " has f1_score: ",f1_score(y_test, y_pred)))

"""        
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_train)
print ("The calculated accuracy scroe on the test set using XGBClassifier is = %f" %accuracy_score(y_train, y_pred))
#confusion_matrix(Y_test,y_pred)
        
""" Implementation of xgboost to find probabilites """
"""
import xgboost
from sklearn.metrics import accuracy_score
params = {
    "eta": 0.2,
    "max_depth": 5,
    "objective": "binary:logistic",
    "silent": 1,
    "eval_metric": "logloss"
    }
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test)

classifier = xgboost.train(params, d_train,100)

y_pred = classifier.predict(d_test)
print ("The calculated log loss value on the test set using XGBoost is = %f" %calculate_logloss(y_test, y_pred))

predictions = [round(value) for value in y_pred]
print ("The calculated accuracy scroe on the test set using XGBoost is = %f" %accuracy_score(y_test, predictions))
"""