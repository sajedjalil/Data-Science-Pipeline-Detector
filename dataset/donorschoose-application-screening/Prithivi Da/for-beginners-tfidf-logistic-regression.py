'''
@prithivida 

I just kept this super simple, ofcourse I could have defined some function to make some redundant code reusable. 
Correct me if I am wrong anywhere or if there is a easy way to do something which I am doing in a complicated way.

Thanks for your help :-)

'''

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# All Standard imports

import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 
import datetime
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sparse
from scipy.sparse import hstack, csr_matrix


# Load all input files

train_filename =  '../input/train.csv'
test_filename =  '../input/test.csv'
res_filename =  '../input/resources.csv'


train_data_raw = pd.read_csv(train_filename, parse_dates=["project_submitted_datetime"]).fillna('')
test_data_raw = pd.read_csv(test_filename, parse_dates=["project_submitted_datetime"]).fillna('')
resource_data_raw = read_csv(res_filename).fillna('')


# Rename the existing columns just for readability
# Create a "total price" 
resource_data_raw['item_unit_price'] = resource_data_raw['price']
resource_data_raw['item_quantity']   = resource_data_raw['quantity']
resource_data_raw['item_type']       = resource_data_raw['description']
resource_data_raw['item_total_price'] = resource_data_raw['quantity'] * resource_data_raw['price']
resource_data_raw.drop(['price','quantity','description'],axis=1,inplace=True)

# For each project id there are several item descriptions ie.several rows, Lets make it one row per project id with all descriptions grouped as a space seperated list.
item_desc_csv = resource_data_raw.groupby('id')['item_type'].apply(lambda x: x.sum()).reset_index()
item_desc_csv.head(2)



# Create some useful columns based on the existing columns of the training data
# We are going to see all project essays as one big text blob, project categories / sub categories as one project cat but keep other columns as is.
train_data_raw["year_submitted"] = train_data_raw["project_submitted_datetime"].dt.year
train_data_raw["day_of_year"] = train_data_raw['project_submitted_datetime'].dt.dayofyear 
train_data_raw["day_of_week"] = train_data_raw['project_submitted_datetime'].dt.weekday
train_data_raw["day_submitted"] = train_data_raw['project_submitted_datetime'].dt.day
train_data_raw["quarter_of_the_year"] = train_data_raw['project_submitted_datetime'].dt.quarter
train_data_raw['project_essay'] = train_data_raw.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                            str(row['project_essay_2']), 
                                            str(row['project_essay_3']), 
                                            str(row['project_essay_4'])]), axis=1)
train_data_raw['project_cat'] = train_data_raw.apply(lambda row: ' '.join([str(row['project_subject_categories']), 
                                            str(row['project_subject_subcategories'])]), axis=1)

train_data_raw.drop(['project_essay_1','project_essay_2','project_essay_3','project_essay_4',"project_submitted_datetime"],axis=1,inplace=True)
train_features_engineered = train_data_raw.copy()


# Get some aggregated features

resource_tprice_groupby   = resource_data_raw.groupby('id')['item_total_price'].agg(['sum', 'min', 'max', 'mean']).rename(columns={'sum':'tpsum', 'min':'tpmin', 'max':'tpmax', 'mean':'tpmean'}).reset_index()
train_features_engineered = train_features_engineered.join(resource_tprice_groupby.set_index('id'), on='id')
train_features_engineered = train_features_engineered.join(item_desc_csv.set_index('id'), on='id')




# Repeat exactly what we did in the previous steps on the test data

test_data_raw["year_submitted"] = test_data_raw["project_submitted_datetime"].dt.year
test_data_raw["day_of_year"] = test_data_raw['project_submitted_datetime'].dt.dayofyear 
test_data_raw["day_of_week"] = test_data_raw['project_submitted_datetime'].dt.weekday
test_data_raw["day_submitted"] = test_data_raw['project_submitted_datetime'].dt.day
test_data_raw["quarter_of_the_year"] = test_data_raw['project_submitted_datetime'].dt.quarter
test_data_raw['project_essay'] = test_data_raw.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                            str(row['project_essay_2']), 
                                            str(row['project_essay_3']), 
                                            str(row['project_essay_4'])]), axis=1)

test_data_raw['project_cat'] = test_data_raw.apply(lambda row: ' '.join([str(row['project_subject_categories']), 
                                            str(row['project_subject_subcategories'])]), axis=1)


test_data_raw.drop(['project_essay_1','project_essay_2','project_essay_3','project_essay_4',"project_submitted_datetime"],axis=1,inplace=True)
test_features_engineered = test_data_raw.copy()


# Get some aggregated features

test_features_engineered = test_features_engineered.join(resource_tprice_groupby.set_index('id'), on='id')
test_features_engineered = test_features_engineered.join(item_desc_csv.set_index('id'), on='id')



# Now we have situation were we have numeric features, categorical features and text / linguistic / NLP features.
# we need a mechanism to combine all of them and make one big input.
# We could use Feature Union and pipelines, easier way to do is use DataFrameMapper
# LabelBinarizer will give you 1-hot encoding of al categorical features.
# StandardScaler scales all numeric features uniformly so the values are not drastically different
# We will only combine numeric and categorical features

mapper = DataFrameMapper([
      ('teacher_prefix', LabelBinarizer()),
      ('school_state', LabelBinarizer()),
      ('project_grade_category',   LabelBinarizer()),
      ('year_submitted', LabelBinarizer()),
      ('day_of_week', LabelBinarizer()),      
      ('day_of_year', LabelBinarizer()),
      (['teacher_number_of_previously_posted_projects'], StandardScaler()),
      (['tpsum'], StandardScaler()),
      (['tpmin'], StandardScaler()),
      (['tpmax'], StandardScaler()),
      (['tpmean'],StandardScaler())
], df_out=True)


# Generate numeric and categorical training features
x_train = np.round(mapper.fit_transform(train_features_engineered.copy()), 2).values



title_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    stop_words='english',
    max_features=500)

essay_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    stop_words='english',
    max_features=4500)

cat_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
     analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    stop_words='english',
    max_features=50)


itemtype_vectorizer =  TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    stop_words='english',
    max_features=2000)




all_titles = pd.concat([
     train_features_engineered['project_title'], 
     test_features_engineered['project_title']
])

all_essays = pd.concat([
     train_features_engineered['project_essay'],
     test_features_engineered['project_essay']
])


all_cats = pd.concat([
     train_features_engineered['project_cat'],
     test_features_engineered['project_cat']]
)


all_itemtypes = pd.concat([
     train_features_engineered['item_type'],
     test_features_engineered['item_type']]
)



title_vectorizer.fit(all_titles)
essay_vectorizer.fit(all_essays)
cat_vectorizer.fit(all_cats)
itemtype_vectorizer.fit(all_itemtypes)

# Create TF-IDF vectors for training features.
x_train_title = title_vectorizer.transform(train_features_engineered['project_title'])
x_train_project_essay = essay_vectorizer.transform(train_features_engineered['project_essay'])
x_train_item_type = itemtype_vectorizer.transform(train_features_engineered['item_type'])
x_train_project_cat = cat_vectorizer.transform(train_features_engineered['project_cat'])


# Combine  all the train features: numeric,categorical and text by stacking their respective matrices.
x_train_features = sparse.hstack((csr_matrix(x_train), x_train_title))
x_train_features = sparse.hstack((x_train_features, x_train_project_essay))
x_train_features = sparse.hstack((x_train_features, x_train_item_type))
x_train_features = sparse.hstack((x_train_features, x_train_project_cat))



# Generate numeric and categorical test features
x_test = np.round(mapper.fit_transform(test_features_engineered.copy()), 2).values


# Create TF-IDF vectors for test features.
x_test_title = title_vectorizer.transform(test_features_engineered['project_title'])
x_test_project_essay = essay_vectorizer.transform(test_features_engineered['project_essay'])
x_test_item_type = itemtype_vectorizer.transform(test_features_engineered['item_type'])
x_test_project_cat = cat_vectorizer.transform(test_features_engineered['project_cat'])

# Combine  all the test features: numeric,categorical and text by stacking their respective matrices.
x_test_features = sparse.hstack((csr_matrix(x_test), x_test_title))
x_test_features = sparse.hstack((x_test_features, x_test_project_essay))
x_test_features = sparse.hstack((x_test_features, x_test_item_type))
x_test_features = sparse.hstack((x_test_features, x_test_project_cat))


# Fit the training data to the model, predict on test data :-)
train_features = x_train_features
test_features  = x_test_features
train_target   = train_features_engineered['project_is_approved'].values
submission = pd.DataFrame.from_dict({'id': test_features_engineered['id']})
classifier = LogisticRegression(C=0.1)
classifier.fit(train_features, train_target)
submission['project_is_approved'] = classifier.predict_proba(test_features)[:, 1]
submission.to_csv('submission.csv',index=False)













