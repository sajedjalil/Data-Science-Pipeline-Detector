import numpy as np
import pandas as pd
import os

import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

tqdm.pandas()

data_path = '../input/'
classes = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
}
alphabet = 'abcdefghijklmnopqrstuvwxyz'

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
train['author'] = train['author'].apply(lambda x: classes[x])

test = pd.read_csv(os.path.join(data_path, 'test.csv'))
id_test = test['id'].values

# Clean text
punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']
def clean_text(x):
    x.lower()
    for p in punctuation:
        x.replace(p, '')
    return x

train['text_cleaned'] = train['text'].apply(lambda x: clean_text(x))
test['text_cleaned'] = test['text'].apply(lambda x: clean_text(x))

# Count Vectorizer
cvect = CountVectorizer(ngram_range=(1, 3), stop_words='english')
cvect.fit(pd.concat((train['text_cleaned'], test['text_cleaned']), axis=0))
cvect_train = cvect.transform(train['text_cleaned'])
cvect_test = cvect.transform(test['text_cleaned'])

# TFIDF
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tfidf.fit(pd.concat((train['text_cleaned'], test['text_cleaned']), axis=0))
tfidf_train = tfidf.transform(train['text_cleaned'])
tfidf_test = tfidf.transform(test['text_cleaned'])

def extract_features(df):
    df['len'] = df['text'].apply(lambda x: len(x))
    df['n_words'] = df['text'].apply(lambda x: len(x.split(' ')))
    df['n_.'] = df['text'].str.count('\.')
    df['n_...'] = df['text'].str.count('\...')
    df['n_,'] = df['text'].str.count('\,')
    df['n_:'] = df['text'].str.count('\:')
    df['n_;'] = df['text'].str.count('\;')
    df['n_-'] = df['text'].str.count('\-')
    df['n_?'] = df['text'].str.count('\?')
    df['n_!'] = df['text'].str.count('\!')
    df['n_\''] = df['text'].str.count('\'')
    df['n_"'] = df['text'].str.count('\"')

    # First words in a sentence
    df['n_The '] = df['text'].str.count('The ')
    df['n_I '] = df['text'].str.count('I ')
    df['n_It '] = df['text'].str.count('It ')
    df['n_He '] = df['text'].str.count('He ')
    df['n_Me '] = df['text'].str.count('Me ')
    df['n_She '] = df['text'].str.count('She ')
    df['n_We '] = df['text'].str.count('We ')
    df['n_They '] = df['text'].str.count('They ')
    df['n_You '] = df['text'].str.count('You ')

    # Find numbers of different combinations
    for c in tqdm(alphabet.upper()):
        df['n_' + c] = df['text'].str.count(c)
        df['n_' + c + '.'] = df['text'].str.count(c + '\.')
        df['n_' + c + ','] = df['text'].str.count(c + '\,')

        for c2 in alphabet:
            df['n_' + c + c2] = df['text'].str.count(c + c2)
            df['n_' + c + c2 + '.'] = df['text'].str.count(c + c2 + '\.')
            df['n_' + c + c2 + ','] = df['text'].str.count(c + c2 + '\,')

    for c in tqdm(alphabet):
        df['n_' + c + '.'] = df['text'].str.count(c + '\.')
        df['n_' + c + ','] = df['text'].str.count(c + '\,')
        df['n_' + c + '?'] = df['text'].str.count(c + '\?')
        df['n_' + c + ';'] = df['text'].str.count(c + '\;')
        df['n_' + c + ':'] = df['text'].str.count(c + '\:')

        for c2 in alphabet:
            df['n_' + c + c2 + '.'] = df['text'].str.count(c + c2 + '\.')
            df['n_' + c + c2 + ','] = df['text'].str.count(c + c2 + '\,')
            df['n_' + c + c2 + '?'] = df['text'].str.count(c + c2 + '\?')
            df['n_' + c + c2 + ';'] = df['text'].str.count(c + c2 + '\;')
            df['n_' + c + c2 + ':'] = df['text'].str.count(c + c2 + '\:')
            df['n_' + c + ', ' + c2] = df['text'].str.count(c + '\, ' + c2)

    # And now starting processing of cleaned text
    for c in tqdm(alphabet):
        df['n_' + c] = df['text_cleaned'].str.count(c)
        df['n_' + c + ' '] = df['text_cleaned'].str.count(c + ' ')
        df['n_' + ' ' + c] = df['text_cleaned'].str.count(' ' + c)

        for c2 in alphabet:
            df['n_' + c + c2] = df['text_cleaned'].str.count(c + c2)
            df['n_' + c + c2 + ' '] = df['text_cleaned'].str.count(c + c2 + ' ')
            df['n_' + ' ' + c + c2] = df['text_cleaned'].str.count(' ' + c + c2)
            df['n_' + c + ' ' + c2] = df['text_cleaned'].str.count(c + ' ' + c2)

            for c3 in alphabet:
                df['n_' + c + c2 + c3] = df['text_cleaned'].str.count(c + c2 + c3)
                # df['n_' + c + ' ' + c2 + c3] = df['text_cleaned'].str.count(c + ' ' + c2 + c3)
                # df['n_' + c + c2 + ' ' + c3] = df['text_cleaned'].str.count(c + c2 + ' ' + c3)

    df['n_the'] = df['text_cleaned'].str.count('the ')
    df['n_ a '] = df['text_cleaned'].str.count(' a ')
    df['n_appear'] = df['text_cleaned'].str.count('appear')
    df['n_little'] = df['text_cleaned'].str.count('little')
    df['n_was '] = df['text_cleaned'].str.count('was ')
    df['n_one '] = df['text_cleaned'].str.count('one ')
    df['n_two '] = df['text_cleaned'].str.count('two ')
    df['n_three '] = df['text_cleaned'].str.count('three ')
    df['n_ten '] = df['text_cleaned'].str.count('ten ')
    df['n_is '] = df['text_cleaned'].str.count('is ')
    df['n_are '] = df['text_cleaned'].str.count('are ')
    df['n_ed'] = df['text_cleaned'].str.count('ed ')
    df['n_however'] = df['text_cleaned'].str.count('however')
    df['n_ to '] = df['text_cleaned'].str.count(' to ')
    df['n_into'] = df['text_cleaned'].str.count('into')
    df['n_about '] = df['text_cleaned'].str.count('about ')
    df['n_th'] = df['text_cleaned'].str.count('th')
    df['n_er'] = df['text_cleaned'].str.count('er')
    df['n_ex'] = df['text_cleaned'].str.count('ex')
    df['n_an '] = df['text_cleaned'].str.count('an ')
    df['n_ground'] = df['text_cleaned'].str.count('ground')
    df['n_any'] = df['text_cleaned'].str.count('any')
    df['n_silence'] = df['text_cleaned'].str.count('silence')
    df['n_wall'] = df['text_cleaned'].str.count('wall')

    df.drop(['id', 'text', 'text_cleaned'], axis=1, inplace=True)

print('Processing train...')
extract_features(train)
print('Processing test...')
extract_features(test)

print('train.shape = ' + str(train.shape) + ', test.shape = ' + str(test.shape))

# Drop non-relevant columns
print('Searching for columns with non-changing values...')
counts = train.sum(axis=0)
cols_to_drop = counts[counts == 0].index.values
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)
print('Dropped ' + str(len(cols_to_drop)) + ' columns.')
print('train.shape = ' + str(train.shape) + ', test.shape = ' + str(test.shape))

print('Searching for columns with low STD...')
counts = train.std(axis=0)
cols_to_drop = counts[counts < 0.01].index.values
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)
print('Dropped ' + str(len(cols_to_drop)) + ' columns.')
print('train.shape = ' + str(train.shape) + ', test.shape = ' + str(test.shape))

# Split train dataset on train and CV
X = np.concatenate((train.drop('author', axis=1), tfidf_train.toarray()), axis=1)
y = train['author'].values
X_test = np.concatenate((test, tfidf_test.toarray()), axis=1)

p_valid = []
p_test = []


kf = KFold(n_splits=5, shuffle=False, random_state=0)
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    # LightGBM
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid)

    params = {
        'max_depth': 27, 
        'learning_rate': 0.1,
        'verbose': 0, 
        'early_stopping_round': 120,
        'metric': 'multi_logloss',
        'objective': 'multiclass',
        'num_classes': 3,
        'nthread': 1
    }
    n_estimators = 5000
    model = lgb.train(params, d_train, n_estimators, [d_train, d_valid], verbose_eval=200)

    p_valid.append(model.predict(X_valid, num_iteration=model.best_iteration))
    acc = accuracy_score(y_valid, np.argmax(p_valid[-1], axis=1))
    logloss = log_loss(y_valid, p_valid[-1])
    print('LGB:\tAccuracy = ' + str(round(acc, 6)) + ',\tLogLoss = ' + str(round(logloss, 6)))
    p_test.append(model.predict(X_test, num_iteration=model.best_iteration))

    # MultinomialNB Count Vectorizer
    X_train, X_valid = cvect_train[train_index], cvect_train[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    print('X_train.shape = ' + str(X_train.shape) + ', y_train.shape = ' + str(y_train.shape))
    print('X_valid.shape = ' + str(X_valid.shape) + ', y_valid.shape = ' + str(y_valid.shape))

    model = MultinomialNB()
    model.fit(X_train, y_train)
    p_valid.append(model.predict_proba(X_valid))
    acc = accuracy_score(y_valid, np.argmax(p_valid[-1], axis=1))
    logloss = log_loss(y_valid, p_valid[-1])
    print('MNBc:\tAccuracy = ' + str(round(acc, 6)) + ',\tLogLoss = ' + str(round(logloss, 6)))
    p_test.append(model.predict_proba(cvect_test))
    # break

# Ensemble
print('Ensemble contains ' + str(len(p_valid)) + ' models.')
p_test_ens = np.mean(p_test, axis=0)

# Prepare submission
subm = pd.DataFrame()
subm['id'] = id_test
subm['EAP'] = p_test_ens[:, 0]
subm['HPL'] = p_test_ens[:, 1]
subm['MWS'] = p_test_ens[:, 2]
subm.to_csv('subm.csv', index=False)
print('Done')