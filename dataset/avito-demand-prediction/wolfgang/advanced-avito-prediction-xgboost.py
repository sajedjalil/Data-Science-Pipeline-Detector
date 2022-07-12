# Simple first attempt to predict the propability of demand
# Not using the image info so far and only taking simple 
# categorical features into account
#
import numpy as np
import pandas as pd
import math
import time
import os.path
import gc
import random

import xgboost as xgb

from gensim.models import word2vec
import re


def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time
    
# quick way of calculating a numeric has for a string
def n_hash(s):
    random.seed(hash(s))
    return random.random()

# hash a complete column of a pandas dataframe    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')

def text_to_words(raw_text):
    return raw_text.split() 

def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in model: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # Initialize a counter
    counter = 0
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000 == 0:
        #   print("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs

def main():
    start_time = time.time()
    # create a xgboost model
    model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    
    # load the training data
    train = pd.read_csv('../input/train.csv', nrows=50)
    
    num_features = 100
    model_w2v = word2vec.Word2Vec(train['title'].dropna().apply(text_to_words), workers=4, size=num_features, min_count=1, window=10, sample=1e-3)
    start_time = print_duration (start_time, "Finished word2vec training of title texts")   
    model_w2v.build_vocab(train['description'].dropna().apply(text_to_words), update=True)
    start_time = print_duration (start_time, "Finished word2vec training of description texts")   
    model_w2v.init_sims(replace=True) # end training to speed up use of model
    
    # 
    #f_matrix_train_w2v = getAvgFeatureVecs(sentences_train, model_w2v, num_features)
    #f_matrix_test_w2v = getAvgFeatureVecs(sentences_test, model_w2v, num_features)
    #print(f_matrix_train_w2v)
    '''
    
    # calculate consistent numeric hashes for any categorical features 
    train['user_hash'] = train.apply (lambda row: hash_column (row, 'user_id'),axis=1)
    train['region_hash'] = train.apply (lambda row: hash_column (row, 'region'),axis=1)
    train['city_hash'] = train.apply (lambda row: hash_column (row, 'city'),axis=1)
    train['parent_category_name_hash'] = train.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
    train['category_name_hash'] = train.apply (lambda row: hash_column (row, 'category_name'),axis=1)
    train['user_type_hash'] = train.apply (lambda row: hash_column (row, 'user_type'),axis=1)
    # for the beginning I use only the information if there is an image or not 
    train['image_exists'] = train['image'].isnull().astype(int)
    # calc log for price to reduce effect of very large price differences
    train['price'] = np.log(train['price'])
    #print(train.groupby(['image_exists']).image_exists.count())
    #print(train['image_exists'])
    start_time = print_duration (start_time, "Finished reading")   

    # start training
    train_X = train.as_matrix(columns=['image_top_1', 'user_hash', 'price', 'region_hash', 'city_hash', 'parent_category_name_hash', 'category_name_hash', 'user_type_hash', 'image_exists'])
    model.fit(train_X, train['deal_probability'])
    
    # read test data set
    test = pd.read_csv('../input/test.csv')
    test['user_hash'] = test.apply (lambda row: hash_column (row, 'user_id'),axis=1)
    test['region_hash'] = test.apply (lambda row: hash_column (row, 'region'),axis=1)
    test['city_hash'] = test.apply (lambda row: hash_column (row, 'city'),axis=1)
    test['parent_category_name_hash'] = test.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
    test['category_name_hash'] = test.apply (lambda row: hash_column (row, 'category_name'),axis=1)
    test['user_type_hash'] = test.apply (lambda row: hash_column (row, 'user_type'),axis=1)
    test['image_exists'] = test['image'].isnull().astype(int)
    test['price'] = np.log(test['price'])
    test_X = test.as_matrix(columns=['image_top_1', 'user_hash', 'price', 'region_hash', 'city_hash', 'parent_category_name_hash', 'category_name_hash', 'user_type_hash', 'image_exists'])
    start_time = print_duration (start_time, "Finished training, start prediction")   
    # predict the propabilities for binary classes    
    pred = model.predict(test_X)
    
    start_time = print_duration (start_time, "Finished prediction, start store results")    
    submission = pd.read_csv("../input/sample_submission.csv")
    submission['deal_probability'] = pred
    print(submission[submission['deal_probability'] > 0])
    submission.to_csv("submission.csv", index=False)
    start_time = print_duration(start_time, "Finished to store result")
    '''
    
if __name__ == '__main__':
    main()
    
    