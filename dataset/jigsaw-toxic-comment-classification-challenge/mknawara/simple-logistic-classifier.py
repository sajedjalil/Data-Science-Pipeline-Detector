import gc

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

pd.options.mode.chained_assignment = None

print("Loading data ...    "),
train, test = pd.read_csv("../input/train.csv"), pd.read_csv("../input/test.csv")
IDs = test['id']
X_train, X_test = train['comment_text'], test['comment_text']
X_test.loc[X_test.isnull()] = " " # replace the 1 NaN value in test
Y_train = train[train.columns[2:]]

del train
del test
gc.collect()

print("%.2f of data is not flagged" % (Y_train.loc[(Y_train.sum(axis=1) == 0)].shape[0] / Y_train.shape[0]))

tfv = TfidfVectorizer(min_df=3, max_df=0.9, max_features=None, strip_accents='unicode',\
               analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), use_idf=1,\
               smooth_idf=1, sublinear_tf=1, stop_words='english')
print("tfidf-vectorizing train ...")
tfv.fit(X_train)
X_train = tfv.transform(X_train)
print("tfidf-vectorizing test ...")
X_test = tfv.transform(X_test)

print("fitting log reg & reporting cv accuracy ...")
for i in range(Y_train.shape[1]):
    feature = Y_train.columns[i]
    print("\n%s:" % feature)
    print("Baseline: %.2f" % (Y_train.iloc[:,i].sum() / Y_train.shape[0]))
    clf = LogisticRegression(C=4.0, solver='sag')
    clf.fit(X_train, Y_train.iloc[:,i])
    print(cross_val_score(clf, X_train, Y_train.iloc[:,i], cv=3, scoring='f1'))
    exec("pred_%s = pd.Series(clf.predict_proba(X_test).flatten()[1::2])" % feature)

submission = pd.DataFrame({
             'id': IDs,
             'toxic': pred_toxic,
             'severe_toxic': pred_severe_toxic,
             'obscene': pred_obscene,
             'threat': pred_threat,
             'insult': pred_insult,
             'identity_hate': pred_identity_hate
             })

submission.to_csv("submission_logreg.csv", index=False)