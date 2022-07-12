#Notebook reference to 
#https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts
#Other References:
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
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

#Prediciontiond for Word Counting method
def calculate_selected_text(df_row, tol, pos_words_adj, neg_words_adj):
    
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

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)


if __name__ == '__main__':
    #read in data paths
    train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv').fillna('')
    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv').fillna('')
    sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv').fillna('')

    # The row with index 13133 has NaN text, so remove it from the dataset
    train[train['text'].isna()]
    train.drop(314, inplace = True)
    
    # Make all the text lowercase - casing doesn't matter when 
    # we choose our selected text.
    train['text'] = train['text'].apply(lambda x: x.lower())
    test['text'] = test['text'].apply(lambda x: x.lower())

    # Make training/test split
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(train, test_size=0.2, random_state=0)

#reset this to train on whole data for test
    X_train = train
    
    pos_train = X_train[X_train['sentiment'] == 'positive']
    neutral_train = X_train[X_train['sentiment'] == 'neutral']
    neg_train = X_train[X_train['sentiment'] == 'negative']
    
    # Use CountVectorizer to get the word counts within each dataset
    cv = CountVectorizer(max_df=0.85, min_df=2,
                                         max_features=15000,
                                         ngram_range= (1,10),
                                        )

    X_train_cv = cv.fit_transform(X_train['text'])
    X_pos = cv.transform(pos_train['text'])
    X_neutral = cv.transform(neutral_train['text'])
    X_neg = cv.transform(neg_train['text'])    

    pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
    neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
    neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

    # Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that 
    # contain those words

    pos_words = {}
    neutral_words = {}
    neg_words = {}
    
    for k in cv.get_feature_names():
        pos = pos_count_df[k].sum()
        neutral = neutral_count_df[k].sum()
        neg = neg_count_df[k].sum()
    
#         pos_words[k] = (pos/pos_train.shape[0]) 
#         neutral_words[k] = (neutral/neutral_train.shape[0])
#         neg_words[k] = (neg/neg_train.shape[0])

        pos_words[k] = (pos/pos_train.shape[0]) + (pos/X_train.shape[0])
        neutral_words[k] = (neutral/neutral_train.shape[0]) + (neutral/X_train.shape[0])
        neg_words[k] = (neg/neg_train.shape[0]) + (neg/X_train.shape[0])

    # We need to account for the fact that there will be a lot of words used in tweets of every sentiment.  
    # Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other 
    # sentiments that use that word.

    neg_words_adj = {}
    pos_words_adj = {}

    for key, value in neg_words.items():
        neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
#         neg_words_adj[key] = neg_words[key] - (pos_words[key])
        if neg_words_adj[key] <= 0:
            neg_words_adj[key] = 0

    for key, value in pos_words.items():
        pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
#         pos_words_adj[key] = pos_words[key] - (neg_words[key])
        if pos_words_adj[key] <= 0:
            pos_words_adj[key] = 0
    
    X_val['predicted_selection'] = ''
    tol = 0.001
    for index, row in X_val.iterrows():
        selected_text = calculate_selected_text(row, tol, pos_words_adj,neg_words_adj)
        X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text
        
    X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)
    wc_jaccard = np.mean(X_val['jaccard'])
    print("Word-Count Jaccard Score:", wc_jaccard)

#     X_val.to_csv(os.path.join('./', 'validation.csv'), index=False)
#     print("Saved validation.csv") 

###################
#Submission
###################

    # Submission: Classify test data and save to file   
    for index, row in test.iterrows():
        selected_text = calculate_selected_text(row, tol, pos_words_adj, neg_words_adj)
        sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text
   
    sample.to_csv('submission.csv', index = False)
    print("Saved Submission.csv")