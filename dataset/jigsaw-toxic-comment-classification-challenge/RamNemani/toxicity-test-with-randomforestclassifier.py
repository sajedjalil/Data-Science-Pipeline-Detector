# Put together a quick and dirty script to test out RandomForestClassifier
# On my computer (quadcore, 8gb, hdd) this script took about 3 hours 
# to complete (way longer than it took for writing the script!)
# I did not do any hyper parameter tuning
#
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(12345)

def loadVectorize():
###
### read the train and test datasets into dataframes and vectorize
###
    train = pd.read_csv('../input/train.csv',encoding='utf-8')
    test = pd.read_csv('../input/test.csv',encoding='utf-8')
    train.fillna("unknown", inplace=True)
    test.fillna("unknown", inplace=True)

    ## for quick testing with about 1% of data
    train = train[:10000]
    test = test[:10000]

    commentID = test['id']
    X_train, X_test = train['comment_text'], test['comment_text']
    y_train = train[train.columns[2:]]

    print("X_train.shape :", X_train.shape)
    print("X_test.shape :", X_test.shape)
    print("y_train type  :", type(y_train))
    print("y_train.shape :", y_train.shape)

    print("vectorizing dfboth...")
    dfboth = pd.concat([train['comment_text'], test['comment_text']], axis=0)
    print("dfboth.shape : ", dfboth.shape)

    print("-"*70)
    print("fitting dfboth...")
    t1 = time.time()
    vectorizer=TfidfVectorizer(stop_words = 'english', ngram_range=(1,2))
    vctr = vectorizer.fit(dfboth)
    t2 = time.time()
    print("finshed fitting dfboth, time (seconds) : %5.2f" % (t2-t1))
    print("-"*70)

    del dfboth, train, test

    print("vectorizing X_train...")
    X_train = vctr.transform(X_train.values)
    print("vectorizing X_test...")
    X_test = vctr.transform(X_test.values)

    print("type(X_train) :", type(X_train))
    print("X_train.shape :", X_train.shape)

    return X_test, X_train, y_train, commentID

def trainAndSubmit():

    X_test, X_train, y_train, commentID = loadVectorize()

    cols =['id', 'toxic', 'severe_toxic', 'obscene',
     'threat', 'insult', 'identity_hate' ]
    results = pd.DataFrame(columns=cols)
    results['id'] = commentID

    for i in range(y_train.shape[1]):
        cmnt_type = cols[i+1]
        print("-"*70)
        print("comment type : %s" % cmnt_type)

        clf = RandomForestClassifier(n_estimators=10)

        t1 = time.time()
        clf.fit(X_train, y_train.iloc[:,i])
        t2 = time.time()

        print(cross_val_score(clf, X_train, y_train.iloc[:,i], cv=3, scoring='accuracy'))
        print()
        print("finshed training, time (seconds) : %5.2f" % (t2-t1))
        print("-"*70)

        exec("results['%s'] = pd.Series(clf.predict_proba(X_test).flatten()[1::2])" % cmnt_type)

    print("results.shape : ", results.shape)
    ## results.to_csv('../input/toxic_rfc.csv', index=False)

def main():
    trainAndSubmit()

if __name__ == '__main__':
    main()
