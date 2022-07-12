# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore")

import re
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords


stop_words = stopwords.words('english')


# --- Read Data ---
print(' --- Reading Data --- ')
train = pd.read_csv('../input/train.csv')#[:1000]
test = pd.read_csv('../input/test.csv')#[:1000]
sample = pd.read_csv('../input/sample_submission.csv')#[:1000]



# --- Missing Value Imputation ---
train['comment_text'] = train['comment_text'].fillna('__nocomment__')
test['comment_text']  = test['comment_text'].fillna('__nocomment__')




# --- Customized Utility to Clean Text ---
def clean_text( text ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    text = re.sub("[^A-za-z]"," ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)   
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"I'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" u ", " you ", text) 
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"ain't", "is not", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r",", " commas ", text)
    text = re.sub(r"\.", " fullStop ", text)
    text = re.sub(r"!", " exclamationmark ", text)
    text = re.sub(r"\?", " questionmark ", text)
    text = re.sub(r"'", " singleQoute ", text)
    text = re.sub(r'"', " doubleQoute ", text)
    text = re.sub(r'\n', " newLine ", text)
    return text



# --- Clean Comment Text ----
train['comment_text'] = train['comment_text'].map(lambda x : ' '.join([word.lower() for word in clean_text(x).split(' ') if len(word) > 1 and word != ' ']))
test['comment_text'] = test['comment_text'].map(lambda x : ' '.join([word.lower() for word in clean_text(x).split(' ') if len(word) > 1 and word != ' ']))



# --- Custom Utility to get VW Formated Data ---
def to_vw_format(document, label):
    return str(label) + ' |text ' + document + '\n'



# --- Get Train and Validation Split ---
columnList = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

xTrain, xValid, yTrain, yValid = train_test_split(train.comment_text.values, train[columnList],
                                                  random_state=42, test_size=0.25, shuffle=True)



# --- Get VW Formated Data ---
for label in columnList:
    with open('../input/' + label + 'CommentTrainData.vw', 'w') as outFile:
         for comment, lbl in zip(xTrain, yTrain[label]):
             outFile.write(to_vw_format(comment, 1 if lbl == 1 else -1))

for label in columnList:
    with open('../input/' + label + 'CommentValidData.vw', 'w') as outFile:
        for comment, lbl in zip(xValid, yValid[label]):
            outFile.write(to_vw_format(comment, 1 if lbl == 1 else -1))

with open('../input/testData.vw', 'w') as outFile:
    for comment in test.comment_text:
        outFile.write(to_vw_format(comment, 1))



# --- Customized AUC Utility ---
def customAuc(yActual, yPred):
    fpr, tpr, __ = metrics.roc_curve(yActual, yPred)
    auc          = metrics.auc(fpr, tpr)
    return auc



# --- Logistic Converter ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# --- Train Model, Save Prob on Validation and Test Set ---
for label in columnList:
    inpathTrainVW  = '../input/' + label + 'CommentTrainData.vw'
    outpathTrainVW = '../input/' + label + 'CommentTrainModelData.vw'
    inpathValidVW  = '../input/' + label + 'CommentValidData.vw'
    outpathValidVW = '../input/' + label + 'CommentValidPredData.vw'
    outpathTestVW = '../input/' + label + 'CommentTestPredData.vw'
    os.system('vw -d ' +  inpathTrainVW + ' -k -c --nn 10 --loss_function logistic -b 25 --passes 50 -q ee --l2 0.000005 -f ' + outpathTrainVW)
    os.system('vw -i ' +  outpathTrainVW + ' -t -d ' + inpathValidVW + ' -p ' + outpathValidVW + ' --quiet')
    os.system('vw -i ' + outpathTrainVW + ' -t -d ../input/testData.vw -p ' + outpathTestVW + ' --quiet')



# --- Get AUC and Save Prediction on Test Set ---
for label in columnList:
    yTest = []
    with open('../input' + label + 'CommentTestPredData.vw') as testPrediction:
         for row in testPrediction:
             yTest.append(sigmoid(float(row.strip())))
         sample[label] = yTest
    yPred = []
    auc    = []
    with open(outpathValidVW, 'r') as validPrediction:
        for row in validPrediction:
            yPred.append(sigmoid(float(row.strip())))
        auc.append(customAuc(yValid[label].tolist(), yPred))



print('Logistic AUC :', np.mean(auc))
print(sample.head())

