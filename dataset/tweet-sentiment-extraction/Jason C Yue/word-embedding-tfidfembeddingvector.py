import os
import string
import re
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phraser, Phrases


class TfidfEmbeddingVectorizer(object):
    '''
      TfidfVectorizer with word embedding, all credit to: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/,
                                                          https://gist.github.com/TomLin/30244bcccb7e4f94d191a878a697f698
    '''
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size
        # self.vocabulary_ = None

    def fit(self, docs):  # comply with scikit-learn transformer requirement
        """
        Fit in a list of docs, which had been preprocessed and tokenized,
        such as word bi-grammed, stop-words removed, lemmatized, part of speech filtered.
        Then build up a tfidf model to compute each word's idf as its weight.
        Noted that tf weight is already involved when constructing average word vectors, and thus omitted.
        :param
            pre_processed_docs: list of docs, which are tokenized
        :return:
            self
        """

        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer(stop_words='english', max_features=300)
        tfidf.fit(text_docs)  # must be list of text string

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)  # used as default value for defaultdict
        self.word_idf_weight = defaultdict(lambda: max_idf,
                           [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        self.vocabulary_ = tfidf.vocabulary_
        return self


    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector


    def word_average(self, sent):
        """
        Compute average word vector for a single doc/sentence.
        :param sent: list of sentence tokens
        :return:
            mean: float of averaging word vectors
        """

        mean = []
        for word in sent:
            if word in self.word_model.wv.vocab:
                mean.append(self.word_model.wv.get_vector(word) * self.word_idf_weight[word])  # idf weighted

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean


    def word_average_list(self, docs):
        """
        Compute average word vector for multiple docs, where docs had been tokenized.
        :param docs: list of sentence in list of separated tokens
        :return:
            array of average word vector in shape (len(docs),)
        """
        return np.vstack([self.word_average(sent) for sent in docs])

    
class MultinomialNBClassifier():
    def __init__(self, alpha=0):
        """
        Function: initializes members of the class. Stores probabilities
        of words given each sentiment
        """
        self.prob_w_given_pos = None
        self.prob_w_given_neut = None
        self.prob_w_given_neg = None
        self.prob_pos = None
        self.prob_neut = None
        self.prob_neg = None
    

    def fit(self, X_pos, X_neut, X_neg, alpha=0):
        """
        Function: Gets probability of each word given the sentiment
        """
        num_features = X_pos.shape[1]
        prob_w_given_pos = np.zeros(num_features)
        prob_w_given_neut = np.zeros(num_features)
        prob_w_given_neg = np.zeros(num_features)

        # Sum of all features for each sentiment
        all_feature_sum_pos = X_pos.sum()
        all_feature_sum_neut = X_neut.sum()
        all_feature_sum_neg = X_neg.sum()

        # Iterate through columns (which represent features)
        
        for feature in range(num_features):
            # occurance of a single feature in each of 3 sentiments
            feature_sum_pos = X_pos[:,feature].sum()
            feature_sum_neut = X_neut[:,feature].sum()
            feature_sum_neg = X_neg[:,feature].sum()

            prob_w_given_pos[feature] = (feature_sum_pos+alpha)/(all_feature_sum_pos+num_features*alpha)
            prob_w_given_neut[feature] =(feature_sum_neut+alpha)/(all_feature_sum_neut+num_features*alpha)
            prob_w_given_neg[feature] =(feature_sum_neg+alpha)/(all_feature_sum_neg+num_features*alpha)

            # Uncomment to calculate num of tweets that use word/total num of tweets
            # feature_sum_pos = np.count_nonzero(X_pos[:,feature] > 0)
            # feature_sum_neut = np.count_nonzero(X_neut[:,feature] > 0)
            # feature_sum_neg = np.count_nonzero(X_neg[:,feature] > 0)

            # prob_w_given_pos[feature] = (feature_sum_pos+alpha)/(len(X_pos)+num_features*alpha)
            # prob_w_given_neut[feature] =(feature_sum_neut+alpha)/(len(X_neut)+num_features*alpha)
            # prob_w_given_neg[feature] =(feature_sum_neg+alpha)/(len(X_neg)+num_features*alpha)


        self.prob_w_given_pos = prob_w_given_pos - (prob_w_given_neut + prob_w_given_neg)
        self.prob_w_given_neut = prob_w_given_neut - (prob_w_given_neg + prob_w_given_pos)
        self.prob_w_given_neg = prob_w_given_neg - (prob_w_given_neut + prob_w_given_pos)

        # Currently Unused: P(Y = positive, negative, neutral)
        self.prob_pos = X_pos.shape[0]/(X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0])
        self.prob_neut = X_neut.shape[0]/(X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0])
        self.prob_neg = X_neg.shape[0]/(X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0])


    def predict_selected_text(self, vocab_to_index, text, sentiments):
        predictions = []
        num_examples = len(text)
        for i in range(num_examples):
            weights_to_use = None
            tweet = text[i]
            sentiment = sentiments[i]

            if sentiment == 'neutral':
                predictions.append(tweet)
                continue
            elif sentiment == 'positive':
                weights_to_use = self.prob_w_given_pos
            elif sentiment == 'negative':
                weights_to_use = self.prob_w_given_neg

            words_in_tweet = tweet.split()
            # Get all continous word subsets of the tweet
            word_subsets = [words_in_tweet[i:j+1]
                            for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]

            # Sort candidates by length (to prioritize shorter candidate)
            lst = sorted(word_subsets, key=len)

            max_weight_sum = 0
            selected_text = None

            for word_subset in lst:
                weight_sum = 0
                for word in word_subset:
                    translated_word = word.translate(str.maketrans('', '', string.punctuation))
                    if translated_word in vocab_to_index.keys():
                        weight_sum += weights_to_use[vocab_to_index[translated_word]]

                if weight_sum > max_weight_sum:
                    max_weight_sum = weight_sum
                    selected_text = word_subset

            if selected_text == None:
                predictions.append(tweet)
            else:
                predictions.append(" ".join(selected_text))
        return predictions

    
def load_data(rootdir='./'):
    print('load data \n')
    train = pd.read_csv(os.path.join(rootdir, 'train.csv'))
    test = pd.read_csv(os.path.join(rootdir, 'test.csv'))
    sample = pd.read_csv(os.path.join(rootdir, 'sample_submission.csv'))

    return train, test, sample
    
    
# Credit to https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts
def jaccard(str1, str2): 
    # If both strings are empty
    if len(str1) == 0 and len(str2) == 0:
        return 1
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

    
# credit to https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
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

    
def convert_data(input_data):
    '''
    Convert each sentence as a list of words that will be encapsulated within a list.
    Basic instructions credited to: https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

    parameters:
        - input_data: Pandas dataframe that holds the data to be converted.
    returns:
        - converted_data: A list of words (that are really a sentence)
                          encapsulated in a list.
    '''

    print('Converting data \n')

    # format data, will give us a list of lists
    converted_data = [clean_text(tweet).split() for tweet in input_data['text']]
    converted_data = [tweet for tweet in converted_data if tweet != []]

    # print(converted_data)
    return converted_data


def train_data(input_data):
    # def train_data():
    '''
    Train the model using Word2Vec and our cleaned, formatted data.
    Basic implementation credited to: https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

    parameters:
        - input_data: List of lists, where the nested lists are split sentences.
    returns:
        - model: Trained Word2Vec model.
    '''
    
    print('Training data using Word2Vec model \n')

    common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    
    # create the relevant phrases from the list of sentences.
    # Phraser detects frequently co-occuring words in sentences and combines them.
    phrases = Phrases(input_data, common_terms=common_terms)
    bigram = Phraser(phrases)

    # transform sentences.
    input_data = list(bigram[input_data])

    '''
    min_count: the minimum count of words to consider when training the model;
               words with occurrence less than this count will be ignored.
               the default for min_count is 5.
    size:      number of dimensions of the embedding.
               note: typical interval for `size` (dimensionality) is typically 100-300, 50 for lowest accuracy.
                     after 300, there is no significant increase at the cost of steep increase in computation time.
    workers:   number of processors (parallelization). The default workers is 3.
    window:    the maximum distance between a target word and words around the target word. The default window is 5,
    iter:      number of epochs training over corpus.
    sg:        the training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.
    '''
    model = Word2Vec(input_data, min_count=3, size=300, workers=5, window=5, iter=30, sg=1)
    
    # print('Loading pre-trained model \n')

    # model = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)

    return model


def examples(model):
    '''
    Run some examples through the model to look at output to confirm it works.

    parameters:
        - model: Trained Word2Vec model.
    returns:
        None
    '''

    ('Outputting examples used on trained Word2Vec model \n')

    # prints total number of words in model
    print(len(model.wv.vocab))

    print('Looking into similarities to the word happy:', model.wv.most_similar('happy'))
    print('Looking into similarities to the word funny:', model.wv.most_similar('funny'))
    print('Looking into similarities to the word danger:', model.wv.most_similar('danger'))  
    print('Looking at the similarity distance between happy and weekend:', model.wv.similarity('happy', 'weekend'))
    print('Looking at the similarity distance between alright and disappointed:', model.wv.similarity('alright', 'disappointed'))
    print('Looking at the similarity distance between sniffle and sob:', model.wv.similarity('sniffle', 'sob'))


def predict_selected_text(df, vocab_to_index, pos_w, neut_w, neg_w):
    """
    Predicts the selected text of all tweets in dataframe based on the weights
    parameters:
        - df: dataframe with "text", "selected_text", and "sentiment" fields
        - vocab_to_index: Dictionary that maps word to index
        - pos_w: Weights of words in positive tweets
        - neut_w: Weights of words in negative tweets
        - neg_w: Weights of words in neg tweets
    returns:
        - predictions: predicted selected text (list)
    """
    predictions = []

    for i, row in df.iterrows():
        weights_to_use = None
        tweet = row['text']
        sentiment = row['sentiment']

        if sentiment == 'neutral':
            predictions.append(tweet)
            continue
        elif sentiment == 'positive':
            weights_to_use = pos_w
        elif sentiment == 'negative':
            weights_to_use = neg_w

        words_in_tweet = tweet.split()
        # Get all continous word subsets of the tweet
        word_subsets = [words_in_tweet[i:j+1] for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]
        
        # Sort candidates by length
        lst = sorted(word_subsets, key = len)
        
        max_weight_sum = 0
        selected_text = None

        for word_subset in lst:
            weight_sum = 0
            for word in word_subset:
                translated_word = word.translate(str.maketrans('', '', string.punctuation))
                if translated_word in vocab_to_index.keys():
                    print(translated_word, vocab_to_index[translated_word])
                    weight_sum += weights_to_use[vocab_to_index[translated_word]]
                
            if weight_sum > max_weight_sum:
                max_weight_sum = weight_sum
                selected_text = word_subset
        
        if selected_text == None:
            predictions.append(tweet)
        else:
            predictions.append(" ".join(selected_text))
    return predictions

    
if __name__ == '__main__':
    # Load data
    train, test, sample = load_data(rootdir='/kaggle/input/tweet-sentiment-extraction/')
    train.dropna(inplace=True)              # one null value, so we remove it
    
    # Create a word embedding model using Word2Vec
    converted_data = convert_data(train)
    w2v_model = train_data(converted_data)
    examples(w2v_model)

    # Clean the text
    train['text'] = train['text'].apply(lambda x: clean_text(x))
    train['selected_text'] = train['selected_text'].apply(lambda x: clean_text(x))

    X_train, X_val = train_test_split(train, train_size = 0.80, random_state = 0)
    
    positive_train = X_train[X_train['sentiment'] == 'positive']
    neutral_train = X_train[X_train['sentiment'] == 'neutral']
    negative_train = X_train[X_train['sentiment'] == 'negative']

    # all credit to https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b
    tfidf_vec_tr = vectorizer = TfidfEmbeddingVectorizer(w2v_model)
    tfidf_vec_tr.fit(converted_data)                                 # fit tfidf model first

    # Transform text to vector form
    X_positive = tfidf_vec_tr.transform(positive_train['text'])
    X_neutral = tfidf_vec_tr.transform(neutral_train['text'])
    X_negative = tfidf_vec_tr.transform(negative_train['text'])

    # Multinomial Naive Bayes
    nb = MultinomialNBClassifier()
    nb.fit(X_positive, X_neutral, X_negative, alpha=4)


    # Evaluate model
    vocab_to_index = {k: v for k, v in vectorizer.vocabulary_.items()}
    predicted_text = nb.predict_selected_text(vocab_to_index, X_val['text'].to_numpy(), X_val['sentiment'].to_numpy())

    # Add predicted text column to X_val dataframe
    X_val = X_val.assign(predicted_text=predicted_text)
    # Add jaccard scores column to X_val dataframe
    X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_text']), axis = 1)
    print(X_val)
    print("Word2Vec + Tfidf + MultiNB Jaccard Score: {}".format(np.mean(X_val['jaccard'])))

    # Submission: Classify test data and save to file
    submission_predicted_text = nb.predict_selected_text(vocab_to_index, test['text'].to_numpy(), test['sentiment'].to_numpy())
    submission_df = pd.DataFrame({'textID': test['textID'], 'selected_text': submission_predicted_text})
    submission_df.to_csv(os.path.join('./', 'submission.csv'), index=False)