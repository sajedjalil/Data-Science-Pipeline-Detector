import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from string import punctuation
from tqdm import tqdm_notebook
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from lightgbm import LGBMClassifier
from string import digits
from nltk.stem import PorterStemmer
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import os

#READ TRAINING, TESTING AND RESOURCES FILES
train_path = '../input/donorschoose-application-screening/train.csv'
test_path = '../input/donorschoose-application-screening/test.csv'
resources = '../input/donorschoose-application-screening/resources.csv'

train = pd.read_csv(train_path, delimiter=',')
test = pd.read_csv(test_path, delimiter=',')
res = pd.read_csv(resources, delimiter=',')

#OBTAIN TOTAL PRICE REQUESTED FOR: QUANTITY * PRICE
res['total_requested'] = res['quantity'] * res['price']

#OBTAIN MEAN, SUM, MAX, MIN, COUNT FOR EACH PROJECT ID FOR BOTH TEST AND TRAINING FILES
tmp = res.groupby('id', as_index=False)['total_requested'].sum().rename(columns={'total_requested':'sum_requested'})
train = pd.merge(train, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].mean().rename(columns={'total_requested':'mean_requested'})
train = pd.merge(train, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].max().rename(columns={'total_requested':'max_requested'})
train = pd.merge(train, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].min().rename(columns={'total_requested':'min_requested'})
train = pd.merge(train, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].count().rename(columns={'total_requested':'count_requested'})
train = pd.merge(train, tmp, on='id', how='left')

tmp = res.groupby('id', as_index=False)['total_requested'].sum().rename(columns={'total_requested':'sum_requested'})
test = pd.merge(test, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].mean().rename(columns={'total_requested':'mean_requested'})
test = pd.merge(test, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].max().rename(columns={'total_requested':'max_requested'})
test = pd.merge(test, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].min().rename(columns={'total_requested':'min_requested'})
test= pd.merge(test, tmp, on='id', how='left')
tmp = res.groupby('id', as_index=False)['total_requested'].count().rename(columns={'total_requested':'count_requested'})
test = pd.merge(test, tmp, on='id', how='left')

#OBTAIN OTHER FEATURES RELEVANT TO DATE CORRESPONDING TO DAY, MONTH, YEAR, HOUR, MINUTE AND SECOND FOR TRAINING AND TESTING SET
train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'], format='%Y-%m-%d %H:%M:%S')
train['day'] = train['project_submitted_datetime'].dt.day
train['month'] = train['project_submitted_datetime'].dt.month
train['year'] = train['project_submitted_datetime'].dt.year
train['hour'] = train['project_submitted_datetime'].dt.hour
train['minute'] = train['project_submitted_datetime'].dt.minute
train['second'] = train['project_submitted_datetime'].dt.second

test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime'], format='%Y-%m-%d %H:%M:%S')
test['day'] = test['project_submitted_datetime'].dt.day
test['month'] = test['project_submitted_datetime'].dt.month
test['year'] = test['project_submitted_datetime'].dt.year
test['hour'] = test['project_submitted_datetime'].dt.hour
test['minute'] = test['project_submitted_datetime'].dt.minute
test['second'] = test['project_submitted_datetime'].dt.second

#FILL MISSING PREFIXES WITH TEACHER AND GROUP 'Ms' and 'Mrs.' into a single class and others in a different class
train['teacher_prefix'].fillna('Teacher', inplace=True)
test['teacher_prefix'].fillna('Teacher', inplace=True)
def teacher_prefix(s):
    if s in ['Ms.', 'Mrs.']:
        return 0
    else:
        return 1
train['teacher_prefix'] = train['teacher_prefix'].map(teacher_prefix)
test['teacher_prefix'] = test['teacher_prefix'].map(teacher_prefix)

#GROUP TEACHER ID BASED ON TEACHER NUMBER OF PREVIOUSLY POSTED PROJECTS CORRESPONDING TO THE SUM, MEAN, MAX, AND MIN 
tmp = train.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].sum().rename(columns={'teacher_number_of_previously_posted_projects':'sum_previous'})
train = pd.merge(train, tmp, on='teacher_id', how='left')
tmp = train.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].mean().rename(columns={'teacher_number_of_previously_posted_projects':'mean_previous'})
train = pd.merge(train, tmp, on='teacher_id', how='left')
tmp = train.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].max().rename(columns={'teacher_number_of_previously_posted_projects':'max_previous'})
train = pd.merge(train, tmp, on='teacher_id', how='left')
tmp = train.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].min().rename(columns={'teacher_number_of_previously_posted_projects':'min_previous'})
train = pd.merge(train, tmp, on='teacher_id', how='left')

tmp = test.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].sum().rename(columns={'teacher_number_of_previously_posted_projects':'sum_previous'})
test = pd.merge(test, tmp, on='teacher_id', how='left')
tmp = test.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].mean().rename(columns={'teacher_number_of_previously_posted_projects':'mean_previous'})
test = pd.merge(test, tmp, on='teacher_id', how='left')
tmp = test.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].max().rename(columns={'teacher_number_of_previously_posted_projects':'max_previous'})
test = pd.merge(test, tmp, on='teacher_id', how='left')
tmp =test.groupby('teacher_id', as_index=False)['teacher_number_of_previously_posted_projects'].min().rename(columns={'teacher_number_of_previously_posted_projects':'min_previous'})
test = pd.merge(test, tmp, on='teacher_id', how='left')

#LABEL ENCODE TEACHER_ID, SCHOOL STATE, PROJECT GRADE CATEGORY, PROJECT SUBJECT CATEOGRIES, AND PROJECT SUBJECT SUB CATEGORIES 
columns = ['teacher_id', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']
for col in tqdm(columns):
    le = LabelEncoder()
    le = le.fit(pd.concat([train[col], test[col]], axis=0))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

#SELECT REMAINING COLUMNS IN TEXT AND FILL MISSING VALUES WITH EMPTY SPACE
remaining_columns = [col for col in train.columns if train[col].dtype =='object' and col !='id']
for col in remaining_columns:
    train[col].fillna(' ',inplace=True)
    test[col].fillna(' ', inplace=True)

    
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
def clean_sentences(text):
    #Input: Sentences
    #Output: cleaned sentence by removing numbers, stopwords, punctuations, single alphabet, and change single uppercase to lowercase letters
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)    
    text = tokenizer.tokenize(text)
    text = [s.lower() for s in text]
    text = [s for s in text if s not in set(stopwords.words('english'))]
    text = [s for s in text if len(s)>1]
    text = ' '.join(text)
    return text   

#tqdm.pandas(desc="progress bar")
#for col in remaining_columns:
#    train[col] =train[col].progress_apply(lambda x: clean_sentences(x))
#    test[col] = test[col].progress_apply(lambda x: clean_sentences(x))


def read_glove_vecs(glove_file):
    #input: file
    #output: word to 50d vector mapping output
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map
word_to_vec_map = read_glove_vecs('../input/glove50d/glove.6B.50d.txt')


def prepare_sequence(ds, word_to_vec_map):
    #input: Series, and word_to_vec_map of size(vocab_size,50)
    #output: returns shape of (len(ds), 50)
    traintest_X = []
    for sentence in tqdm(ds.values):
        sequence_words = np.zeros((word_to_vec_map['cucumber'].shape))
        for word in sentence.split():
            if word in word_to_vec_map.keys():
                temp_X = word_to_vec_map[word]
            else:
                temp_X = word_to_vec_map['#']
            sequence_words+=(temp_X)/len(sentence)
        traintest_X.append(sequence_words)
    return np.array(traintest_X)
    
#concatenate all sequences for training and testing set
train_w2v = prepare_sequence(train[remaining_columns[0]], word_to_vec_map)
test_w2v = prepare_sequence(test[remaining_columns[0]], word_to_vec_map)
for col in remaining_columns[1:]:
    temp_train = prepare_sequence(train[col], word_to_vec_map)
    temp_test = prepare_sequence(test[col], word_to_vec_map)
    train_w2v = np.concatenate([train_w2v, temp_train], axis=-1)
    test_w2v = np.concatenate([test_w2v, temp_test], axis=-1)

#choose columns excluding target and datetime features
selected_columns = [col for col in train.columns if train[col].dtype != 'object' and col not in ['project_is_approved', 'project_submitted_datetime']]

#prepare trainig and testing set model input
train_X = np.concatenate([train[selected_columns].values, train_w2v], axis=-1)
train_Y = train['project_is_approved'].values
test_X = np.concatenate([test[selected_columns].values, test_w2v], axis=-1)

print('train_features: ', train_X.shape)
print('test_features: ', test_X.shape)

#shuffle data
random_id = np.random.permutation(len(train_X))
train_X = train_X[random_id]
train_Y = train_Y[random_id]

#train test split
tra_X, val_X, tra_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.3, random_state=1)
print('train_features: ', tra_X.shape)
print('validation_features: ', val_X.shape)
print('train_target: ', tra_Y.shape)
print('validation_target: ', val_Y.shape)

#parameters for xgboost and lgbm
XGB_params = {'num_round':200}
xgb = XGBClassifier(max_depth = 6, learning_rate=0.1, estimator =100, **XGB_params)
xgb.fit(tra_X, tra_Y, eval_set= [(tra_X, tra_Y), (val_X, val_Y)], eval_metric = 'auc', verbose=True)

lgb= LGBMClassifier(num_leaves =100, max_depth = 6, n_estimators =100)
lgb.fit(tra_X, tra_Y, eval_set= [(tra_X, tra_Y), (val_X, val_Y)], eval_metric = 'auc',verbose=True)



XGB_params = {'num_round':200}
xgb = XGBClassifier(max_depth = 6, learning_rate=0.2, estimator =100, **XGB_params)
lgb= LGBMClassifier(num_leaves =100, max_depth = 6, n_estimators =100)
xgb.fit(train_X, train_Y, eval_set= [(train_X, train_Y)], eval_metric = 'auc', verbose=True)
lgb.fit(train_X, train_Y, eval_set= [(train_X, train_Y)], eval_metric = 'auc',verbose=True)

test_pred_xgb = xgb.predict_proba(test_X)
test_pred_lgb =lgb.predict_proba(test_X)

#ENSEMBLING
test_pred = (0.5 *test_pred_xgb + 0.5*test_pred_lgb)
df = pd.DataFrame({'id':test.id})
df['project_is_approved'] = test_pred[:,1]
df.to_csv('submission_ensembling.csv', index=False)
