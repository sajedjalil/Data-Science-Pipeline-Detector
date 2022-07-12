# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import nltk
from tqdm import tqdm
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}

train = pd.read_csv('../input/train.csv', dtype=dtype, low_memory=True)
resources = pd.read_csv('../input/resources.csv')
test = pd.read_csv('../input/test.csv', dtype=dtype, low_memory=True)
ss = pd.read_csv('../input/sample_submission.csv')

train.set_index('id', inplace = True)
test.set_index('id', inplace = True)
#=============================================================================
# deal with cost
# =============================================================================
res = pd.DataFrame(resources[['id', 'quantity', 'price']].groupby('id').agg(\
    {
        'quantity': [
            'sum',
            'min', 
            'max', 
        ],
        'price': [
            'count', 
            'sum', 
            'min', 
            'max', 
            'mean', 
            'std', 
            lambda x: len(np.unique(x)),
        ]}
    ))
res.columns = ['_'.join(col) for col in res.columns]
res['mean_price'] = res['price_sum']/res['quantity_sum']
resources['cost'] = resources['quantity'] * resources['price']
cost_prj = resources.groupby('id')['cost'].sum()
res = res.join(cost_prj)

# =============================================================================
# deal with teacher
# =============================================================================

# =============================================================================
# join cost, teacher info to train
# =============================================================================
df_all = pd.concat([train, test])

sb = df_all.groupby('teacher_id')['teacher_number_of_previously_posted_projects'].max().rename('total_submision')+1


df_all = df_all.join(res)
df_all = df_all.merge(pd.DataFrame(sb), left_on = 'teacher_id', right_index = True, how = 'left')
gc.collect()
# =============================================================================
# deal with text
# =============================================================================



def get_polarity(text):
    textblob = TextBlob(text)
    pol = textblob.sentiment.polarity
    return round(pol,2)

def get_subjectivity(text):
    textblob = TextBlob(text)
    subj = textblob.sentiment.subjectivity
    return round(subj,2)

df_all['essay_count'] = list(map(lambda x: 2 if pd.isnull(x) else 4, df_all['project_essay_3']))

df_all['project_essay_3'].fillna('', inplace = True)
df_all['project_essay_4'].fillna('', inplace = True)

df_all['essay'] = df_all[['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']].apply(lambda x: '\n'.join(x), axis = 1)

df_all['project_title_len'] = df_all['project_title'].apply(lambda x: len(str(x)))
df_all['project_resource_summary_len'] = df_all['project_resource_summary'].apply(lambda x: len(str(x)))
df_all['essay_len'] = df_all['essay'].apply(lambda x: len(str(x)))
gc.collect()

stopWords = set(stopwords.words('english'))
wnl = WordNetLemmatizer()


n_features = [
    400, 
    4100, 
    400]

text_cols = [    
        'project_title', 
        'essay', 
        'project_resource_summary']



for c_i, c in enumerate(text_cols):
    text_list = []
    for text in tqdm(df_all[c]):
        tokens = nltk.word_tokenize(text.lower())
        lemmatized_tokens = [wnl.lemmatize(tk) for tk in tokens]
        text_list.append(' '.join(lemmatized_tokens))
    df_all[c] = text_list
    tfidf = TfidfVectorizer(
            max_features = n_features[c_i],
            norm = 'l2',
            stop_words = stopWords)
    tfidf.fit(df_all[c])
    words = tfidf.get_feature_names()
    tfidf_all = np.array(tfidf.transform(df_all[c]).toarray(), dtype=np.float16)
    for i in range(n_features[c_i]):
        df_all['_'.join([c,'tfidf', words[i]])] = tfidf_all[:, i]
        
    del tfidf, tfidf_all
    
gc.collect()

# =============================================================================
# deal with subjects
# ============================================================================
df_all['subjects'] = df_all['project_subject_categories'] + df_all['project_subject_subcategories']
df_all.drop(['project_subject_categories', 'project_subject_subcategories'], 1, inplace = True)


df_all['essay_polarity'] =list(map(get_polarity, tqdm(df_all['essay'])))
df_all['essay_subjectivity'] = list(map(get_subjectivity, tqdm(df_all['essay'])))

df_all['title_polarity'] = list(map(get_polarity, tqdm(df_all['project_title'])))
df_all['title_subjectivity'] = list(map(get_subjectivity, tqdm(df_all['project_title'])))

df_all['project_title_wc'] = df_all['project_title'].apply(lambda x: len(str(x).split(' ')))
df_all['essay_wc'] = df_all['essay'].apply(lambda x: len(str(x).split(' ')))
df_all['project_resource_summary_wc'] = df_all['project_resource_summary'].apply(lambda x: len(str(x).split(' ')))

df_all.to_pickle('df_text_done.pickle')
gc.collect()


# =============================================================================
# deal with catetories
# =============================================================================
cols = [
    'teacher_prefix',
    'school_state', 
    'project_grade_category', 
    'subjects'
]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    df_all[c] = le.transform(df_all[c].astype(str))
del le


# =============================================================================
# deal with date
# =============================================================================
def process_timestamp(df):
    df['year'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[0]))
    df['month'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[1]))
    df['date'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['day_of_week'] = pd.to_datetime(df['project_submitted_datetime']).dt.weekday
    df['hour'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))
    df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime']).values.astype(np.int64)

process_timestamp(df_all)
# =============================================================================
# drop useless
# =============================================================================

cols_to_drop = [
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4',
    'teacher_id',
    'project_title', 
    'essay', 
    'project_resource_summary',
]

df_all.drop(cols_to_drop, axis = 1, errors = 'ignore', inplace = True)

X = df_all[:len(train)].drop('project_is_approved', axis = 1).reset_index(drop = True)
y = df_all[:len(train)]['project_is_approved'].reset_index(drop = True)

X_test = df_all[len(train):].drop('project_is_approved', axis = 1)

X.to_pickle('X_sent.pickle')
y.to_pickle('y_sent.pickle')
X_test.to_pickle('X_test_sent.pickle')

gc.collect()



# =============================================================================
# building a model
# =============================================================================

X = pd.read_pickle('X_sent.pickle')
y = pd.read_pickle('y_sent.pickle')
X_test= pd.read_pickle('X_test_sent.pickle')


feature_names = list(X.columns)

cnt = 0
p_buf = []
n_splits = 8
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    )
auc_buf = []   
important_f = []
pred_list = []
for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 12,
        'num_leaves': 31,
        'learning_rate': 0.02,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
    }  

    
    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=150,
        verbose_eval=100,
    )

    importance = model.feature_importance()
    model_fnames = model.feature_name()
    
    tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
    tuples = [x for x in tuples if x[1] > 0]
    important_f.append(tuples)
    if cnt == 0:
        print('Important features:')
        print(tuples[:50])

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)

    pred_list.append(p)
    auc_buf.append(auc)


    cnt += 1
    
    del model
    gc.collect
    
result = [pred_list, auc_buf]
pd.to_pickle(result, 'result.pickle')
importances = pd.DataFrame.from_records([[j[0] for j in i] for i in important_f])
pd.to_pickle(importances, 'importances.pickle')

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))


pred_all = pred_list[0]*0
for lists in pred_list:
    pred_all += lists    
pred_all = pred_all/len(pred_list)

pred_top4 = pred_list[0]*0
pick = np.argsort(auc_buf)[-4:]
for i in pick:
    pred_top4 += pred_list[i]
pred_top4 = pred_top4/(len(pred_list)/2)
    
pred_weighted = pred_list[0]*0
weights = np.array(auc_buf) / np.sum(auc_buf)
for i, lists in enumerate(pred_list):
    pred_weighted += lists * weights[i]  
 

pred_median = pred_list[0]*0
pick_2 = np.argsort(auc_buf)[2:-2]
for i in pick_2:
    pred_median += pred_list[i]
pred_median = pred_median/(len(pred_list)/2)

# Prepare submission
def submission(pred, pred_name):
    subm = pd.DataFrame()
    subm['id'] = X_test.index
    subm['project_is_approved'] = pred
    subm.to_csv('{}.csv'.format(pred_name), index=False)
    
submission(pred_all, 'pred_all')
submission(pred_top4, 'pred_top4')
submission(pred_weighted, 'pred_weighted')
submission(pred_median, 'pred_median')