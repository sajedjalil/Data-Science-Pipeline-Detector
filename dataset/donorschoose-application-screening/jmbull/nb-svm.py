# How can this work with vectors AND features?
# When using hstack, gets the error:
# TypeError: only integer scalar arrays can be converted to a scalar index

# Note that I'm only using 5,000 records for testing this.
# Public LB with all data gets .72x.

import pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import re
import string
import gc

train = pd.read_csv('../input/train.csv', dtype={"project_essay_3": object, "project_essay_4": object}, nrows=5000)
labels = pd.DataFrame(train['project_is_approved'].values)
labels.columns = ['project_is_approved']
train = train.drop('project_is_approved', axis=1)

test = pd.read_csv('../input/test.csv', dtype={"project_essay_3": object, "project_essay_4": object}, nrows=5000)

resources = pd.read_csv('../input/resources.csv')

subm = pd.read_csv('../input/sample_submission.csv')

train.fillna(('unk'), inplace=True) 
test.fillna(('unk'), inplace=True)

# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

# Label encoding

df_all = pd.concat([train, test], axis=0)

cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category', 
    'project_subject_categories', 
    'project_subject_subcategories'
]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
    
del df_all; gc.collect()

# Feature engineering

# Date and time
train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime'])

# Date as int may contain some ordinal value
train['datetime_int'] = train['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['datetime_int'] = test['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# Date parts

train['datetime_day'] = train['project_submitted_datetime'].dt.day
train['datetime_dow'] = train['project_submitted_datetime'].dt.dayofweek
train['datetime_year'] = train['project_submitted_datetime'].dt.year
train['datetime_month'] = train['project_submitted_datetime'].dt.month
train['datetime_hour'] = train['project_submitted_datetime'].dt.hour
train = train.drop('project_submitted_datetime', axis=1)

test['datetime_day'] = test['project_submitted_datetime'].dt.day
test['datetime_dow'] = test['project_submitted_datetime'].dt.dayofweek
test['datetime_year'] = test['project_submitted_datetime'].dt.year
test['datetime_month'] = test['project_submitted_datetime'].dt.month
test['datetime_hour'] = test['project_submitted_datetime'].dt.hour
test = test.drop('project_submitted_datetime', axis=1)

# Essay length
train['e1_length'] = train['project_essay_1'].apply(len)
test['e1_length'] = train['project_essay_1'].apply(len)

train['e2_length'] = train['project_essay_2'].apply(len)
test['e2_length'] = train['project_essay_2'].apply(len)

# Has more than 2 essays?
train['has_gt2_essays'] = train['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)
test['has_gt2_essays'] = test['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)

# Combine resources file
# Thanks, the1owl! 
# https://www.kaggle.com/the1owl/the-choice-is-yours

resources['resources_total'] = resources['quantity'] * resources['price']

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].sum()
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].mean()
dfr = dfr.rename(columns={'resources_total':'resources_total_mean'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].count()
dfr = dfr.rename(columns={'quantity':'resources_quantity_count'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].sum()
dfr = dfr.rename(columns={'quantity':'resources_quantity_sum'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

# We're done with IDs for now
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

del dfr; gc.collect()

# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)

train = train.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)
test = test.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)

gc.collect()

COMMENT = 'project_essay'

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
train_vec = vec.fit_transform(train[COMMENT])
test_vec = vec.transform(test[COMMENT])

train = train.drop('project_essay', axis=1)
test = test.drop('project_essay', axis=1)

# Combine text vectors and features

train = csr_matrix(train.values)
test = csr_matrix(test.values)

X_train_stack = hstack([train, train_vec[0:train.shape[0]]])
X_test_stack = hstack([test, test_vec[0:test.shape[0]]])

print('Train shape: ', X_train_stack.shape, '\n\nTest Shape: ', X_test_stack.shape)

label_cols = ['project_is_approved']

x = train_vec #X_train_stack
test_x = test_vec #X_test_stack

# All of the NB-SVM stuff is from:
# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((test.shape[0], len(label_cols)))

for i, j in enumerate(label_cols):
    print('fitting: ', j)
    m,r = get_mdl(labels[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('nbsvm-submission.csv', index=False)