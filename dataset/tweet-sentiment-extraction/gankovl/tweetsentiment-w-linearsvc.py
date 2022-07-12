
## Sources / helpful links

# https://www.kaggle.com/c/tweet-sentiment-extraction/overview/evaluation
# https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# https://scikit-learn.org/stable/modules/svm.html
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
# https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV


import numpy as np
import pandas as pd
#import nltk
#nltk.download('stopwords') # < only run this once. need to download the stopwords document
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer

SPLITRATIO = 0.8 ## percentage of data that goes to training set (1-SR = % in val set)
## CountVectorizer hyperparameters
MAX_DIMS = 10000  ## maximum number of words to consider when training
MIN_DF = 2
MAX_DF = 1.0
## LinearSVC hyperparameters
C_VAL = 0.2


# Taken from competition info
def jaccard(str1, str2):
    str1 = str(str1)
    str2 = str(str2)
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if __name__ == "__main__":

    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    #        1. read raw train data, split into training/validation sets
    train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
    sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

    train.drop(314, inplace = True)                                 ## remove an empty line
    X_trn, X_val = train_test_split(train, train_size=SPLITRATIO)   ## train/ val split


    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    #        2. Turn the "selected text" snippets into vectors (count_cv)
    #               Only look at the positive/negative cases
    trn_polarized = X_trn[ X_trn.sentiment != 'neutral' ]
    val_polarized = X_val[ X_val.sentiment != 'neutral' ]

    cv = CountVectorizer(
        # stop_words=set(stopwords.words('english')), ## the train/validation accuracy improves slightly w/o stopwords
        # preprocessor=clean_text, ## not sure if I need this, actually. Acc is about the same.
        max_features = MAX_DIMS,
        min_df = MIN_DF,
        max_df = MAX_DF
    )
    cv.fit( trn_polarized.selected_text)                                 # define dimensions from +/- sel_text
    matrix_train = cv.transform( trn_polarized.selected_text ).toarray() # vectorize sel_text of train data
    matrix_vp = cv.transform( val_polarized.text ).toarray()
    print("\nDIMENSIONS: {}\n".format(np.shape(matrix_train)[1]))


    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    #        3. Train an SVM model (on just positive/negative cases)
    model_pn = LinearSVC( C=C_VAL )
    trained_model_pn = model_pn.fit( matrix_train, trn_polarized.sentiment )
    print("+/- classification acc on training data:\t{}".format(trained_model_pn.score(matrix_train, trn_polarized.sentiment) ))
    print("+/- classification acc on validation data:\t{}".format( trained_model_pn.score(matrix_vp, val_polarized.sentiment) ))


    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    #        4. generate new text selections on validation set
    def extract_substring(tweet):
        if tweet.sentiment == 'neutral':
            return tweet.text
        ## generate substrings/ ngrams
        ## (adapted from https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts)
        words = tweet.text.split()
        words_len = len(words)

        substrings = []
        for i in range(words_len):
            for j in range(i,words_len):
                temp = words[i:j+1]
                joined = ""
                for word in temp:
                    joined += word + " "
                substrings += [joined[:-1]] ## trim the last space

        ## vectorize the substrings
        vec_substr = cv.transform(substrings).toarray()
        confidence = trained_model_pn.decision_function(vec_substr) ## LinearSVC version

        ## finding the most positive/ negative confidence score
        if(tweet.sentiment == 'positive'): idx = np.argmax(confidence)
        else:                              idx = np.argmin(confidence)
        return substrings[idx]

    ## Calculate Jaccard score (internal validation)
    pd.options.mode.chained_assignment = None ## no idea what this does or means
    X_val['predicted_selection'] = ''         ## making a new column
    for index, row in X_val.iterrows():
        selected_text = extract_substring(row)
        X_val.loc[X_val.textID == row.textID, ['predicted_selection']] = selected_text

    X_val['jaccard'] = X_val.apply(lambda x: jaccard(x.selected_text, x.predicted_selection), axis = 1)
    print('The jaccard score is:', np.mean(X_val['jaccard']))


    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
    #        5. Generate test submissions for Kaggle (retrain on whole set)
    # Taken almost verbatim from https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts

    ## step 2
    train_polarized = train[train.sentiment != 'neutral']
    cv.fit(train_polarized.selected_text)
    mat_trn_pol = cv.transform(train_polarized.selected_text).toarray()
    print("\nDIMENSIONS: {}\n".format(np.shape(mat_trn_pol)[1]))

    ## step 3
    svm = LinearSVC( C=C_VAL )
    trained_model_pn = svm.fit(mat_trn_pol, train_polarized.sentiment)
    print("+/- clf acc, all training data:\t{}".format(trained_model_pn.score(mat_trn_pol, train_polarized.sentiment)))

    ## step 4
    for index, row in test.iterrows():
        selected_text = extract_substring(row)
        sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text
    sample.to_csv('submission.csv', index = False)
