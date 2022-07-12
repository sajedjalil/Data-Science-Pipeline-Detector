import nltk
import pandas as pd
import os
import re
import nltk.data
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


def sentence_to_wordlist(review, remove_stopwords=True, stemming=False):
    # Function for cleaning sentences
    # Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # Remove URLs
    review_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', review_text, flags=re.MULTILINE)
    review_text = re.sub(r'^http?:\/\/.*[\r\n]*', '', review_text, flags=re.MULTILINE)
    # Convert words to lower case and split them into words
    words = review_text.lower().split()
    # Remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # Use stemming to reduce words to their base word.
    # Snowball stemmer performs better over porter stemmer in practical usage
    if stemming:
        snowball_stemmer = SnowballStemmer("english")
        words = [snowball_stemmer.stem(word) for word in words]
    # Return the cleaned sentence as a list of words
    return (words)


# Count is used for tagging reviews
count = 1


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences.
    # Returns a list of sentences, where each sentence is a list of words
    global count
    # Using NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    # Loop over each sentence and convert them into Tagged documents
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences += [TaggedDocument(words=sentence_to_wordlist(raw_sentence), tags=[count])]

    # increase tag count for each review
    count += 1
    return sentences


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given review
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    
    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    
    # Divide the result by the number of words in a review to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    
    # Initialize a counter
    counter = 0.
    
    # Preallocate a 2D numpy array ( for speed )
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    
    # Loop through the reviews
    for review in reviews:
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def get_cleaned_reviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(sentence_to_wordlist(review, remove_stopwords=True))
    return clean_reviews


if __name__ == '__main__':

    print("Reading data from files to data frames", "\n")

    # Read data from files
    train = pd.read_csv('../input/labeledTrainData.tsv', header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv('../input/testData.tsv', header=0,
                        delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv('../input/unlabeledTrainData.tsv', header=0,
                        delimiter="\t", quoting=3)

    print("Completed reading data from files to data frames", "\n")

    # Set model parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 5  # Number of threads to run in parallel
    context = 10  # Context window size for Doc2vec
    downsampling = 1e-3  # Downsample setting for frequent words
    model_name = "d2v_remstopwds_stemfalse_" + str(num_features) + "fts_" + str(min_word_count) +\
                 "minw_" + str(context) + "cxt"

    # Load the punkt tokenizer which is used for splitting paragraphs into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences

    sentences = []  # Initialize an empty list of sentences

    print("Parsing sentences from training reviews")
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print("Parsing sentences from unlabeled training  reviews")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)


    print("Training Doc2Vec model...", "\n")

    # Set parameters and train the word2vec model
    model = Doc2Vec(sentences,
                    workers=num_workers,
                    size=num_features,
                    min_count=min_word_count,
                    window=context,
                    sample=downsampling,
                    seed=1)

    print("Completed training Doc2Vec model.", "\n")

    # Making the model more memory-efficient
    model.init_sims(replace=True)

    # You can save the model for later use. You can load it later using Doc2Vec.load()
    # model.save("models/" + model_name)
    # print("Trained Model saved in models folder as " + model_name, "\n")

    # print("Loading pre trained model", "\n")
    # model = Doc2Vec.load('models/model_name')
    # print("Model Loading completed", "\n")

    # Creating average vectors for the training and test sets

    print("Creating average feature vectors for training reviews", "\n")
    trainDataVecs = getAvgFeatureVecs(get_cleaned_reviews(train), model, num_features)
    print("Created average feature vectors for training reviews", "\n")

    print("Creating average feature vectors for test reviews", "\n")
    testDataVecs = getAvgFeatureVecs(get_cleaned_reviews(test), model, num_features)
    print("Created average feature vectors for test reviews", "\n")

    y = train["sentiment"]

    clf_lr = LogisticRegression()

    # printing f1 scores from cross validation
    f1_scr = cross_val_score(clf_lr, trainDataVecs, y, cv=5, scoring=make_scorer(f1_score))
    print("F1 scores from cross validation:", f1_scr, "\n")

    print("Fitting a Logistic Regression classifier to labeled training data...", "\n")
    clf_lr.fit(trainDataVecs, y)
    print("Classifier training complete", "\n")

    print("Predicting test data", "\n")
    result = clf_lr.predict(testDataVecs)
    print("Results are generated", "\n")

    print("Writing the results to a file", "\n")
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output_file_name = "results.csv"
    output.to_csv(output_file_name, index=False, quoting=3)

    print("Execution complete. You can find the results in "+ output_file_name)