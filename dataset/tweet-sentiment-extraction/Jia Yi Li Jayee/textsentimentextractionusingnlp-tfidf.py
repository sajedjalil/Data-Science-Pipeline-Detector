#NLP Reference:
#https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
import pandas as pd 
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import string
pd.options.mode.chained_assignment = None

#from Kaggle - calculates score
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
    

#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

#Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


def searchKeywords(tweet, feature_names, tfidf_transformer, cv):
    #generate tf-idf for the given document
    tfidf_vector=tfidf_transformer.transform(cv.transform([tweet]))
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tfidf_vector.tocoo())
    #extract only the top n; n here is 5
    keywords=extract_topn_from_vector(feature_names,sorted_items,topn=20)
    return keywords


#predictions for TFIDF method
def predict(data, tfidf_transformer, pos_tfidf_transformer,neg_tfidf_transformer, cv,feature_names):
    tfidf_predictions=[]
    text = data['text'].to_numpy()
    sentiments = data['sentiment'].to_numpy()    
#     corpus = get_corpus(data)
    
    num_examples = data.shape[0]

    #loop through data to predict text
    for i in range(num_examples):
        tweet = text[i]
        sentiment = sentiments[i]
        
        if sentiment == 'neutral':
            #add entire text
            tweet = text[i]
            tfidf_predictions.append(tweet)   

        elif sentiment == 'positive':
        
            # fetch tweet for which keywords needs to be extracted
            keywords = searchKeywords(tweet,feature_names,pos_tfidf_transformer, cv)
            tweet = text[i]
            words_in_tweet = tweet.split()
            # Get all continous word subsets of the tweet
            word_subsets = [words_in_tweet[i:j+1] for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]

            # Sort candidates by length (to prioritize shorter candidate)
            lst = sorted(word_subsets, key=len)

            max_weight_sum = 0
            dict_max_sum = 0
            selected_text = None
            dict_selected_text = None

            for i in range(len(word_subsets)):
                weight_sum = 0
                for p in range(len(lst[i])):
                    if lst[i][p] in keywords:
                        weight_sum += keywords[lst[i][p]]

                    if weight_sum > max_weight_sum:
                        max_weight_sum = weight_sum
                        selected_text = lst[i]


            if selected_text == None:
                tfidf_predictions.append(" ".join(keywords))
            else:
                tfidf_predictions.append(" ".join(selected_text))
        
        elif sentiment == 'negative':            
           # fetch tweet for which keywords needs to be extracted
            keywords = searchKeywords(tweet, feature_names, neg_tfidf_transformer, cv)            
            tweet = text[i]    
            words_in_tweet = tweet.split()
            
            # Get all continous word subsets of the tweet
            word_subsets = [words_in_tweet[i:j+1] for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]

            # Sort candidates by length (to prioritize shorter candidate)
            lst = sorted(word_subsets, key=len)

            max_weight_sum = 0
            selected_text = None
          
            for i in range(len(word_subsets)):
                weight_sum = 0
                dict_weight = 0
                for p in range(len(lst[i])):
                    if lst[i][p] in keywords:
                        weight_sum += keywords[lst[i][p]]

                    if weight_sum > max_weight_sum:
                        max_weight_sum = weight_sum
                        selected_text = lst[i]

            if selected_text == None:
                tfidf_predictions.append(" ".join(keywords))
            else:
                tfidf_predictions.append(" ".join(selected_text))
        
    return tfidf_predictions



if __name__ == '__main__':
   #read in data paths
    train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv').fillna('')
    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv').fillna('')

    # The row with index 13133 has NaN text, so remove it from the dataset
    train[train['text'].isna()]
    train.drop(314, inplace = True)
    
    # Make all the text lowercase - casing doesn't matter when 
    # we choose our selected text.
    train['text'] = train['text'].apply(lambda x: x.lower())
    test['text'] = test['text'].apply(lambda x: x.lower())

    # Make training/test split
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(train, test_size=0.2)

#reset this to train on whole data for test
#commenting when validating
    X_train = train

    pos_train = X_train[X_train['sentiment'] == 'positive']
    neutral_train = X_train[X_train['sentiment'] == 'neutral']
    neg_train = X_train[X_train['sentiment'] == 'negative']
    
    
    # Use CountVectorizer to get the word counts within each dataset
    cv = CountVectorizer(max_df=1.0, min_df=3,
                                         max_features=5000,
                                         ngram_range= (1,5),
                                        )

    X_train_cv = cv.fit_transform(X_train['text'])
    X_pos = cv.transform(pos_train['text'])
    X_neutral = cv.transform(neutral_train['text'])
    X_neg = cv.transform(neg_train['text'])
    
###################
#USING TFIDF
###################
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer=TfidfTransformer(norm='l2',sublinear_tf=False, smooth_idf=False,use_idf=False)
    tfidf_transformer.fit(X_train_cv)
    # get feature names
    feature_names=cv.get_feature_names()

    #######
    #New
    ######
    pos_tfidf_transformer=TfidfTransformer(norm='l2',sublinear_tf=False, smooth_idf=False,use_idf=False)
    pos_tfidf_transformer.fit(X_pos)
    neg_tfidf_transformer=TfidfTransformer(norm='l2',sublinear_tf=False, smooth_idf=False,use_idf=False)
    neg_tfidf_transformer.fit(X_neg)
    
    
    #predict data
    tfidf_predictions = predict(X_val,tfidf_transformer, pos_tfidf_transformer, neg_tfidf_transformer,cv,feature_names)
#     tfidf_predictions = predict(X_val, tfidf_transformer,cv,feature_names)
    tfidf_predictions = np.asarray(tfidf_predictions)
   
    # Add predicted text column to  dataframe
    X_val = X_val.assign(tfidf_predicted_text=tfidf_predictions)
    # Add jaccard scores column to  dataframe
    X_val['tfidf_jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['tfidf_predicted_text']), axis = 1)
    tfidf_jaccard = np.mean(X_val['tfidf_jaccard'])
    print("TFIDF Jaccard Score:", tfidf_jaccard)
 
    # Submission: Classify test data and save to file
    test_predictions = predict(test,tfidf_transformer, pos_tfidf_transformer, neg_tfidf_transformer,cv,feature_names)
    test_predictions = np.asarray(test_predictions)
    submission_df = pd.DataFrame({'textID': test['textID'], 'selected_text': test_predictions})
    submission_df.to_csv(os.path.join('./', 'submission.csv'), index=False)
    print("Saved Submission.csv")
