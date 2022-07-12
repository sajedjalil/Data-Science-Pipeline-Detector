# import modules
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb

# load data
train_variants = pd.read_csv('../input/training_variants', sep = ',')
test_variants = pd.read_csv('../input/test_variants', sep = ',')
train_text = pd.read_csv('../input/training_text', sep = '\|\|', skiprows=1, engine='python', names=["ID","text"])
test_text = pd.read_csv('../input/test_text', sep = '\|\|', skiprows=1, engine='python' ,names=["ID","text"])


# combine variant and text datasets
train = train_variants.merge(train_text, on='ID', how='left')
test = test_variants.merge(test_text, on='ID', how='left')


# train.head()

# test.head()

print('train shape: ', train.shape)
print('test shape: ', test.shape)


# train.describe()

# test.describe()


combined = pd.concat([train, test], axis=0)


print('Number of Unique Genes: ', combined.Gene.nunique())
print('Number of Unique Variations: ', combined.Variation.nunique())

# combined.Gene.unique()


# find most common genes
# combined.groupby(['Gene'])['text'].count().sort_values(ascending=False)[:10]


# find most common variations
# combined.groupby(['Variation'])['text'].count().sort_values(ascending=False)[:10]


# train.text.str.len().describe()


# Remove useless symbols from description
def clean(s):    
    # Remove any tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", s)
    # Keep only regular chars:
    cleaned = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", cleaned)
    # Remove unicode chars
    cleaned = re.sub("\\\\u(.){4}", " ", cleaned)
    # Remove extra whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    
    return cleaned.strip()

train['clean_text'] = train.text.apply(lambda x: clean(x))
test['clean_text'] = test.text.apply(lambda x: clean(x))


# convert text to lowercase
train['clean_text'] = train.clean_text.str.lower()
test['clean_text'] = test.clean_text.str.lower()


# count number of characters in text
train['char_count'] = train.clean_text.str.len()
test['char_count'] = test.clean_text.str.len()

# count number of characters removed
train['char_count_dirty'] = train.text.str.len() - train.clean_text.str.len()
test['char_count_dirty'] = test.text.str.len() - test.clean_text.str.len()

# train.head()


# create list of all words in text
train['text_list'] = train.clean_text.str.split(' ')
test['text_list'] = test.clean_text.str.split(' ')

# count number of words in text
train['word_count'] = train.text_list.str.len()
test['word_count'] = test.text_list.str.len()

# find average word length in text
train['avg_word_len'] = train.char_count / train.word_count
test['avg_word_len'] = test.char_count / test.word_count

# train.head()


# create list of all words -- not just words in each row
train_text_list = [item for sublist in train['text_list'].ravel() for item in sublist]
test_text_list = [item for sublist in test['text_list'].ravel() for item in sublist]

print('train list: ', len(train_text_list))
print('test list: ', len(test_text_list))

text_list = train_text_list + test_text_list

print('combined list: ', len(text_list))


# find 250 most common words
feature_transform = CountVectorizer(stop_words='english', max_features=250)
feature_transform.fit(text_list)


#  create new occurence features from most common words
def transform_data(X):
    feat_sparse = feature_transform.transform(X['clean_text'])
    vocabulary = feature_transform.vocabulary_
    X1 = pd.DataFrame([pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0])])
    X1.columns = list(sorted(vocabulary.keys()))
    X = pd.concat([X.reset_index(drop=True), X1.reset_index(drop=True)], axis = 1)
    return X

train = transform_data(train)
test = transform_data(test)

# train.head()


# encode gene feature into numerical
train['Gene'] = preprocessing.LabelEncoder().fit_transform(train['Gene'])
test['Gene'] = preprocessing.LabelEncoder().fit_transform(test['Gene'])

# # encode gene feature into numerical
# train['Variation'] = preprocessing.LabelEncoder().fit_transform(train['Variation'])
# test['Variation'] = preprocessing.LabelEncoder().fit_transform(test['Variation'])

# drop non-numerical data
train = train.drop(['Variation', 'text', 'clean_text', 'text_list'], axis=1)
test = test.drop(['Variation', 'text', 'clean_text', 'text_list'], axis=1)

# train.head()


print(train['Class'].unique())

print(train.shape)
print(test.shape)

print('nulls: ', train[train['Class'].isnull()].shape)
train.Class.unique()

# need to do this to avoid an error by XGB -- classes should start at 0, not 1
train.Class = train.Class - 1


x_train = train.drop(['Class'], axis=1)
x_test = train['Class']

print('Start Training')

xgb_params = {
	# don't change
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class' : 9,
    
    # change
    'eta': 0.05,
    'max_depth': 5,
    'n_folds': 5,
    'silent': 1,    
}

dtrain = xgb.DMatrix(x_train, x_test)

early = 20
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds = early, verbose_eval=1)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round = int(num_boost_rounds / 0.80))

fig, ax = plt.subplots(1, 1, figsize=(6, 10))
xgb.plot_importance(model, max_num_features=25, height=0.5, ax=ax)


test_data = xgb.DMatrix(test)

y_predict = pd.concat([test.ID, pd.DataFrame(model.predict(test_data))], axis=1)
output = pd.DataFrame(y_predict)

output.columns = ['ID', 'class1', 'class2', 'class3', 'class4',
                  'class5', 'class6', 'class7', 'class8', 'class9']

output.to_csv('XGB_submission.csv', index=False)
print(output.head())