# Modifying Tilii's original scrips https://www.kaggle.com/tilii7/tuned-logreg-oof-files
# With https://www.kaggle.com/tunguz/cnn-glove300-3-oof-4-epochs
# The output oof and prediction files could potentially be used for ensembling

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

# Data processing was done as in Bojan's fork of the original script:
# https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

traintime = timer(None)
train_time = timer(None)
train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
tr_ids = train[['id']]
train[class_names] = train[class_names].astype(np.int8)
target = train[class_names]

print(' Cleaning ...')
# PREPROCESSING PART
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data

trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
del trate, tete
train.drop(["new_comment_text"], axis=1, inplace=True)
test.drop(["new_comment_text"], axis=1, inplace=True)

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
timer(train_time)

train_time = timer(None)
print(' Part 1/2 of vectorizing ...')
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
timer(train_time)

train_time = timer(None)
print(' Part 2/2 of vectorizing ...')
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
timer(train_time)

train_features = hstack([train_char_features, train_word_features]).tocsr()
test_features = hstack([test_char_features, test_word_features]).tocsr()
timer(traintime)

all_parameters = {
                  'C'             : [1.048113, 0.1930, 0.596362, 0.25595, 0.449843, 0.25595],
                  'tol'           : [0.1, 0.1, 0.046416, 0.0215443, 0.1, 0.01],
                  'solver'        : ['lbfgs', 'newton-cg', 'lbfgs', 'newton-cg', 'newton-cg', 'lbfgs'],
                  'fit_intercept' : [True, True, True, True, True, True],
                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
                 }

folds = 3
scores = []
scores_classes = np.zeros((len(class_names), folds))

submission = pd.DataFrame.from_dict({'id': test['id']})
submission_oof = train[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
#skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
kf = KFold(n_splits=folds, shuffle=True, random_state=239)

idpred = tr_ids

traintime = timer(None)
for j, (class_name) in enumerate(class_names):
#    train_target = train[class_name]

    classifier = LogisticRegression(
        C=all_parameters['C'][j],
        max_iter=200,
        tol=all_parameters['tol'][j],
        solver=all_parameters['solver'][j],
        fit_intercept=all_parameters['fit_intercept'][j],
        penalty=all_parameters['penalty'][j],
        dual=False,
        class_weight=all_parameters['class_weight'][j],
        verbose=0)

    avreal = target[class_name]
    lr_cv_sum = 0
    lr_pred = []
    lr_fpred = []
    lr_avpred = np.zeros(train.shape[0])

    train_time = timer(None)
    for i, (train_index, val_index) in enumerate(kf.split(train_features)):
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = target.loc[train_index], target.loc[val_index]

        classifier.fit(X_train, y_train[class_name])
        scores_val = classifier.predict_proba(X_val)[:, 1]
        lr_avpred[val_index] = scores_val
        lr_y_pred = classifier.predict_proba(test_features)[:, 1]
        scores_classes[j][i] = roc_auc_score(y_val[class_name], scores_val)
        print('\n Fold %02d class %s AUC: %.6f' % ((i+1), class_name, scores_classes[j][i]))

        if i > 0:
            lr_fpred = lr_pred + lr_y_pred
        else:
            lr_fpred = lr_y_pred

        lr_pred = lr_fpred

    lr_cv_score = (lr_cv_sum / folds)
    lr_oof_auc = roc_auc_score(avreal, lr_avpred)
    print('\n Average class %s AUC:\t%.6f' % (class_name, np.mean(scores_classes[j])))
    print(' Out-of-fold class %s AUC:\t%.6f' % (class_name, lr_oof_auc))
    timer(train_time)

    submission[class_name] = lr_pred / folds
    submission_oof['prediction_' + class_name] = lr_avpred

print('\n Overall AUC:\t%.6f' % (np.mean(scores_classes)))
submission.to_csv('submission-tuned-LR-01.csv', index=False)
submission_oof.to_csv('oof-tuned-LR-01.csv', index=False)
timer(traintime)
