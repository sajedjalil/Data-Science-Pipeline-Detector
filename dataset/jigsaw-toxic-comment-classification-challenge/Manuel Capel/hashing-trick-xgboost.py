import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import HashingVectorizer
import xgboost as xgb
from sklearn.metrics import roc_auc_score



np.random.seed(42)
data_train, data_test, X_train, y_train, X_test = None, None, None, None, None
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def get_data(test=False):
    '''get the regulat datasets if test is False, otherwise a 2/3 1/3 split of the train dataset'''
    global data_train
    global data_test
    if data_train is not None and data_test is not None:
        return
    df_train_ = pd.read_csv('../input/train.csv')
    df_test_ = pd.read_csv('../input/test.csv')
    if test:
        ind_test = list(np.random.choice(df_train_.shape[0], round(df_train_.shape[0] / 3), replace=False))
        ind_train = list(set(list(range(df_train_.shape[0]))) - set(ind_test))
        df_train__ = df_train_.iloc[ind_train,:].reset_index(drop=True)
        df_test__ = df_train_.iloc[ind_test,:].reset_index(drop=True)
        data_train, data_test = df_train__, df_test__
    else:
        data_train, data_test = df_train_.reset_index(drop=True), df_test_.reset_index(drop=True)



def hash_classify(test=False, n_estimators=1000, n_features=2 ** 23):
    '''if test is True, makes a 2/3 1/3 split of the training data set'''
    global X_train
    global y_train
    global X_test
    get_data(test=test)
    data_result = data_test[['id']]

    for category in categories:
        y_train = data_train[category]

        data_train['comment_text'] = data_train['comment_text'].apply(lambda x: ' '.join(re.findall(r'\w+', x)))
        data_test['comment_text'] = data_test['comment_text'].apply(lambda x: ' '.join(re.findall(r'\w+', x)))
        # Initilize hashing vectorizer
        vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=n_features)
        # Hash the comment texts
        X_train = vectorizer.transform(data_train['comment_text'])
        X_test = vectorizer.transform(data_test['comment_text'])
        
        # Use XGBoost on the hashed data
        clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        prediction_category = clf.predict_proba(X_test)[:, 1]
        data_result.loc[:,category] = prediction_category

    suffix = 'submission'
    if test: #assess
        suffix = 'test'
        print('Hashing:\n')
        for category in categories:
            print('score {}: {}'.format(category, roc_auc_score(data_test[category], data_result[category])))
    return data_result

submission = hash_classify()
submission.to_csv('hashingXgb.csv', index=False)