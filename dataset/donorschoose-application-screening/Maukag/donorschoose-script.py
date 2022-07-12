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

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
resources=pd.read_csv('../input/resources.csv')

def feature_extraction(df):
    df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
    df['project_essay_1_len'] = df['project_essay_1'].apply(lambda x: len(str(x)))
    df['project_essay_2_len'] = df['project_essay_2'].apply(lambda x: len(str(x)))
    df['project_essay_3_len'] = df['project_essay_3'].apply(lambda x: len(str(x)))
    df['project_essay_4_len'] = df['project_essay_4'].apply(lambda x: len(str(x)))
    df['project_resource_summary_len'] = df['project_resource_summary'].apply(lambda x: len(str(x)))
    
    df['project_title_wc'] = df['project_title'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_1_wc'] = df['project_essay_1'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_2_wc'] = df['project_essay_2'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_3_wc'] = df['project_essay_3'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_4_wc'] = df['project_essay_4'].apply(lambda x: len(str(x).split(' ')))
    df['project_resource_summary_wc'] = df['project_resource_summary'].apply(lambda x: len(str(x).split(' ')))
    
    df['year'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[0]))
    df['month'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[1]))
    df['date'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['day_of_week'] = pd.to_datetime(df['project_submitted_datetime']).dt.weekday
    df['hour'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))
    df['minute'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[1]))
  
feature_extraction(train)
feature_extraction(test)

impute_cols=['project_essay_3','project_essay_4','teacher_prefix']

train[impute_cols]=train[impute_cols].fillna('none')

test[impute_cols]=test[impute_cols].fillna('none')

resources['description']=resources['description'].fillna('none')

grouped=resources.groupby('id')

agg_rc = grouped.agg({'quantity':'sum', 'price':'sum', 'description':lambda x: ' '.join(x)})
agg_rc.reset_index(inplace=True)

train=pd.merge(left=train,right=agg_rc,how='inner',on='id')

test=pd.merge(left=test,right=agg_rc,how='inner',on='id')

text_cols=['project_essay_1','project_essay_2','project_essay_3','project_essay_4',
           'project_resource_summary','project_title','description']

train['text'] = train.apply(lambda row: ' '.join([str(row[col]) for col in text_cols]), axis=1)

test['text'] = test.apply(lambda row: ' '.join([str(row[col]) for col in text_cols]), axis=1)

train.drop(text_cols,axis=1,inplace=True)

test.drop(text_cols,axis=1,inplace=True)

train.drop(['project_submitted_datetime','teacher_id'],axis=1,inplace=True)

test.drop(['project_submitted_datetime','teacher_id'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder

prefix_encoder=LabelEncoder()
state_encoder=LabelEncoder()
grade_encoder=LabelEncoder()
subject_encoder=LabelEncoder()
subject_sub_encoder=LabelEncoder()

prefix_encoder.fit(list(train['teacher_prefix']) + list(train['teacher_prefix']))
state_encoder.fit(list(train['school_state']) + list(['school_state']))
grade_encoder.fit(list(train['project_grade_category']) + list(test['project_grade_category']))
subject_encoder.fit(list(train['project_subject_categories']) + list(test['project_subject_categories']))
subject_sub_encoder.fit(list(train['project_subject_subcategories']) + list(test['project_subject_subcategories']))


train['teacher_prefix']=prefix_encoder.transform(train['teacher_prefix'])
train['school_state']=state_encoder.transform(train['school_state'])
train['project_grade_category']=grade_encoder.transform(train['project_grade_category'])
train['project_subject_categories']=subject_encoder.transform(train['project_subject_categories'])
train['project_subject_subcategories']=subject_sub_encoder.transform(train['project_subject_subcategories'])

test['teacher_prefix']=prefix_encoder.transform(test['teacher_prefix'])
test['school_state']=state_encoder.transform(test['school_state'])
test['project_grade_category']=grade_encoder.transform(test['project_grade_category'])
test['project_subject_categories']=subject_encoder.transform(test['project_subject_categories'])
test['project_subject_subcategories']=subject_sub_encoder.transform(test['project_subject_subcategories'])

test_id=test['id']

test.drop('id',axis=1,inplace=True)

train.drop('id',axis=1,inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline

tfidf=TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    dtype=np.float32,
    max_features=5000
)

n_components=20

svd=TruncatedSVD(n_components=n_components)

tfidf.fit(train['text'])

train_tfidf=tfidf.transform(train['text'])

test_tfidf=tfidf.transform(test['text'])

svd.fit(train_tfidf)

train_svd=svd.transform(train_tfidf)

test_svd=svd.transform(test_tfidf)

for i in range(n_components):
    train['svd_{}'.format(i)]=train_svd[:,i]
    test['svd_{}'.format(i)]=test_svd[:,i]
    
train.drop('text',axis=1,inplace=True)

test.drop('text',axis=1,inplace=True)

test.head()

from sklearn.ensemble import RandomForestClassifier

x_train=train.drop('project_is_approved',axis=1)

y_train=train['project_is_approved']

x_test=test

model=RandomForestClassifier(n_estimators=1000,max_depth=5)

model.fit(x_train,y_train)

predictions=model.predict_proba(x_test)

predictions=predictions[:,1]

submission=pd.DataFrame(columns=['id','project_is_approved'])

submission['id']=test_id

submission['project_is_approved']=predictions

submission.to_csv('submission.csv',index=False)