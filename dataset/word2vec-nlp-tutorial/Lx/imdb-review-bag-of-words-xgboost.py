# TODO:
#   Part 1: Bag of Words with XGBoost
#       version 1: 0.84680
#
#   Part 2: Fine-tuning Bag of Words with XGBoost Model
#       version 5: 0.86340

# Remark:
#   I have seen many kernels using the additional dataset, namely IMDB_review_master, which actually includes the original train and test sets. 
#   They explicitly trained their model using the test set and hence the accuracy is high. 
#   In other words, the data leakage problem has already occurred at the very beginning, which is a huge mistake. 

#   In this problem, the goal is to simply learn some basic NLP works as well as experimenting with the XGBoost model. 


################################
#       importing libraries    #
################################
import numpy as np 
import pandas as pd 
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup    
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os


################################
#       functions              #
################################
def text_cleaning(texts):
    # steps:
    #   1. Remove HTML
    #   2. Remove non-letters
    #   3. Convert to lower case
    #   4. Remove stopwords
    #   5. Return space joined texts
    
    # Initialize an empty list to hold the clean reviews
    clean_text = []
    
    for text in texts:
        text = BeautifulSoup(text, "lxml").get_text() 
        text = re.sub("[^a-zA-Z]", " ", text) 
        text = text.lower().split()                             
        stop_word_list = set(stopwords.words("english"))                  
        text = [word for word in text if not word in stop_word_list] 
        clean_text.append((" ".join(text)))
        
    return clean_text

################################
#       data exploration       #
################################
train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3)

# train: 25000 x 3    test: 25000 x 3
print("Train shape:", train.shape, " Test shape:", test.shape)

# train columns:
#      id: <numeric>     sentiment: 0-negative 1-positive     review: <text>

# test columns:
#      id: <numeric>     review: <text> 
print ("Data format:")
print("Train -", train.columns, "\nTest -", test.columns)

# check label ratio include counts of NaN
print("Train set sentiment ratio:\n", train.sentiment.value_counts(dropna=False))


################################
#       data cleaning          #
################################
print ("Data cleaning...")
clean_train = text_cleaning(train['review'])
clean_test  = text_cleaning(test['review'])

################################
#       feature building       #
################################
# Bag of Words
print ("Building Bag-of-Words...")
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, max_features = 5000) 
bow_train = (vectorizer.fit_transform(clean_train)).toarray()
bow_test = (vectorizer.transform(clean_test)).toarray()

################################
#       model building       #
################################
# XGB Classifier
print ("Model building - XGB...")
# GridSearch with small set of params caused error code 137 (exceeding memory problem)
# So I just gonna randomly tune the parameters, hopefully it gives better result than version 1: 0.84680

#params = {'max_depth': [3,5], 'n_estimators':[50,200]}
#xgb_model = xgb.XGBClassifier(objective="binary:logistic",tree_method='gpu_hist', predictor='gpu_predictor')
#BOW_XGB = GridSearchCV(xgb_model, param_grid=params, cv=3)

BOW_XGB = xgb.XGBClassifier(max_depth=7, n_estimators=300, objective="binary:logistic", random_state=1, tree_method='gpu_hist', predictor='gpu_predictor')
BOW_XGB_scores = cross_val_score(BOW_XGB, bow_train, train.sentiment, cv=3, n_jobs=-1)
print("Averaged CV Accuracy: %0.2f (+/- %0.2f)" % (BOW_XGB_scores.mean(), BOW_XGB_scores.std() * 2))

BOW_XGB.fit(bow_train, train.sentiment)

# Make prediction 
result = BOW_XGB.predict(bow_test)

submission = pd.DataFrame(data={"id":test["id"], "sentiment":result})
submission.to_csv("BagOfWord-XGB.csv", index=False, quoting=3)
print("Done")