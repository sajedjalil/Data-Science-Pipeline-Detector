from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from sklearn.pipeline import make_union
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack
from scipy.sparse import vstack
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression

import wordbatch
from wordbatch.models import FTRL, FM_FTRL
from sklearn.linear_model import Ridge
from wordbatch.extractors import WordBag, WordHash

from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostRegressor
import gc
import time

print('reading data... start')
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test_stg2.tsv', sep='\t')
test_ids = df_test.test_id.values.tolist()

all_df = pd.concat([df_train.drop(['price','train_id'], axis = 1), df_test.drop(['test_id'], axis = 1)], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"]) # calculate log(1+x)

temp = [df_train] # Just del df_train will only remove the mapping of the variable to the data but the data will not be collected,but put the data in a list and del list will collect the data
del temp, df_train, df_test
gc.collect()

print('reading data... end')

def time_now():
    return time.time()
	
time_tracker = time_now()

print("brand_name.fillna(value='None', inplace=True)")
all_df.brand_name.fillna(value='None', inplace=True)

print("Split category into sub categories... Begin")
sub_cat_df = all_df.category_name.str.split(pat='/', n=2, expand=True)
sub_cat_df.columns = ['general_cat','sub_cat_1','sub_cat_2']
all_df['general_category'] = sub_cat_df.general_cat
all_df['sub_cat_1'] = sub_cat_df.sub_cat_1
all_df['sub_cat_2'] = sub_cat_df.sub_cat_2

all_df.drop(columns=['category_name'], inplace=True)
print("Split category into sub categories... End")

temp = [sub_cat_df]
del temp, sub_cat_df
gc.collect()

print('genenral category, sub category 1 and sub category 2, fill values to None')
all_df.general_category.fillna(value='None', inplace=True)
all_df.sub_cat_1.fillna(value='None', inplace=True)
all_df.sub_cat_2.fillna(value='None', inplace=True)

print(f'{time_now() - time_tracker} seconds for duplicates and na handlings')
	
print('Text Preprocessing Function initializations')
stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
	
print(f'title processing start')
time_tracker = time_now()
processed_titles=[]

for (index, title) in enumerate(all_df.name.values):
    processedTitleWords = []
    cleanedTitle = cleanpunc(cleanhtml(title))
    for word in cleanedTitle.split():
        word_in_lowercase = word.lower()
        if(word_in_lowercase.isalpha and \
           (word_in_lowercase not in stop) and \
           len(word_in_lowercase)>2):
            processedTitleWords.append(sno.stem(word_in_lowercase).encode('utf8'))
    processed_titles.append(b" ".join(processedTitleWords))
    
all_df['name'] = processed_titles
all_df['name'] = all_df.name.str.decode('utf8')

print(f'{time_now() - time_tracker} seconds for title processing')

temp = [processed_titles]
del temp, processed_titles
gc.collect()

print(f'description processing start')

time_tracker = time_now()

all_df.item_description.fillna('None',inplace = True)

processed_descriptions=[]

for (index, description) in enumerate(all_df.item_description.values):
    processedDiscriptionWords = []
    cleaned_description = cleanpunc(cleanhtml(description))
    for word in cleaned_description.split():
        word_in_lowercase = word.lower()
        if(word_in_lowercase.isalpha and \
           (word_in_lowercase not in stop) and \
           len(word_in_lowercase)>2):
            processedDiscriptionWords.append(sno.stem(word_in_lowercase).encode('utf8'))
    processed_descriptions.append(b" ".join(processedDiscriptionWords))

all_df['item_description'] = processed_descriptions
all_df['item_description'] = all_df.item_description.str.decode('utf8')

print(f'{time_now() - time_tracker} seconds for description processing')

temp = [processed_descriptions]
del temp, processed_descriptions
gc.collect()

all_df.brand_name = all_df.brand_name.astype('str')
all_df.name = all_df.name.astype('str')
all_df.general_category = all_df.general_category.astype('str')
all_df.sub_cat_1 = all_df.sub_cat_1.astype('str')
all_df.sub_cat_2 = all_df.sub_cat_2.astype('str')


print(f'Remove [rm] from description and name')

def remove_rm(description):
    if "[rm]" in description:
        return description.replace("[rm] ","")
    return description
    
all_df.item_description = all_df.item_description.apply(remove_rm)

all_df.name = all_df.name.apply(remove_rm)

print(f'Remove special characters from name')
all_df.name = all_df.name.apply(lambda x : re.sub('[\W_]+', ' ', x, flags=re.UNICODE))

all_df.item_description = all_df.item_description.apply(lambda x : re.sub('[\W_]+', ' ', x, flags=re.UNICODE))

all_df.general_category = all_df.general_category.astype('category')
all_df.sub_cat_1 = all_df.sub_cat_1.astype('category')
all_df.sub_cat_2 = all_df.sub_cat_2.astype('category')
all_df.brand_name = all_df.brand_name.astype('category')
all_df.item_condition_id = all_df.item_condition_id.astype('category')
all_df.shipping = all_df.shipping.astype('category')


print(f'define safe label encoder')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
#from sklearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np
import pandas as pd
#from skutil.base import BaseSkutil
#from skutil.utils import validate_is_pd

__all__ = [
    'SafeLabelEncoder',
    'OneHotCategoricalEncoder'
]


def _get_unseen():
    """Basically just a static method
    instead of a class attribute to avoid
    someone accidentally changing it."""
    return 99999


class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 99999

    Attributes
    ----------

    classes_ : the classes that are encoded
    """

    def transform(self, y):
        """Perform encoding if already fit.

        Parameters
        ----------

        y : array_like, shape=(n_samples,)
            The array to encode

        Returns
        -------

        e : array_like, shape=(n_samples,)
            The encoded array
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        #_check_numpy_unicode_bug(classes)

        # Check not too many:
        unseen = _get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([
                         np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
                         for x in y
                         ])

        return e



print('spilt train test after pre processing')
x_train = all_df[: nrow_train]
x_test =  all_df[nrow_train : ]


#pickle.dump(x_train, open('../test_ids.p','wb'))
#pickle.dump(y_train, open('../x_train.p','wb'))
#pickle.dump(x_test, open('../y_train.p','wb'))

del all_df
gc.collect()


time_tracker = time_now()
print(f'Name Tfidf vectorizeration start')
name_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features = 2 ** 17)
x_train_name_tfidf = name_tfidf_vectorizer.fit_transform(x_train.name)
x_test_name_tfidf = name_tfidf_vectorizer.transform(x_test.name)
name_standerdizer = StandardScaler(with_mean=False)
x_train_name_tfidf = name_standerdizer.fit_transform(x_train_name_tfidf)
x_test_name_tfidf = name_standerdizer.transform(x_test_name_tfidf)
#pickle.dump(x_train_name_tfidf, open('../x_train_name_tfidf.p','wb'))
#pickle.dump(x_test_name_tfidf, open('../x_test_name_tfidf.p','wb'))

print(f'Name Tfidf vectorizeration end {time_now() - time_tracker} seconds to complete')

time_tracker = time_now()
print(f'Description Tfidf vectorizeration start')

desc_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features = 2 * (2 ** 17))
x_train_desc_tfidf = desc_tfidf_vectorizer.fit_transform(x_train.item_description)
x_test_desc_tfidf = desc_tfidf_vectorizer.transform(x_test.item_description)
desc_standerdizer = StandardScaler(with_mean=False)
x_train_desc_tfidf = desc_standerdizer.fit_transform(x_train_desc_tfidf)
x_test_desc_tfidf = desc_standerdizer.transform(x_test_desc_tfidf)

#pickle.dump(x_train_desc_tfidf, open('../x_train_desc_tfidf.p','wb'))
#pickle.dump(x_test_desc_tfidf, open('../x_test_desc_tfidf.p','wb'))

print(f'Description Tfidf vectorizeration {time_now() - time_tracker} seconds to complete')

del name_tfidf_vectorizer, desc_tfidf_vectorizer, name_standerdizer, desc_standerdizer
gc.collect()


x_train_text_features = hstack([x_train_name_tfidf, x_train_desc_tfidf]).tocsr()
x_test_text_features = hstack([x_test_name_tfidf, x_test_desc_tfidf]).tocsr()

del x_train_name_tfidf, x_train_desc_tfidf, x_test_name_tfidf, x_test_desc_tfidf
gc.collect()



print('Preparing data for LGBM...')
x_train_text_features_lgbm, x_cv_text_features_lgbm, y_train_text_features_lgbm, y_cv_text_features_lgbm = train_test_split(x_train_text_features, \
                                                                     y_train,\
                                                                    test_size = 0.2)

lgb_train = lgb.Dataset(x_train_text_features_lgbm, y_train_text_features_lgbm)
lgb_cv = lgb.Dataset(x_cv_text_features_lgbm, y_cv_text_features_lgbm, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 100,
    'learning_rate': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'verbose': 0
}

print('LGBM start...')
time_tracker = time_now()

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=150,
                valid_sets=lgb_cv,
                early_stopping_rounds=10
                )

print(f'LGBM traning complete. took {time_now() - time_tracker} seconds')

print(f'LGBM Name and Description only Validation Loss')
y_pred = gbm.predict(x_cv_text_features_lgbm, num_iteration=gbm.best_iteration)
print('The rmse of cv prediction is:', mean_squared_error(y_cv_text_features_lgbm, y_pred) ** 0.5)

print('LGBM Text and Description only Test prediction start...')
time_tracker = time_now()

y_test_text_features_lgbm = gbm.predict(x_test_text_features, num_iteration=gbm.best_iteration)
y_train_text_features_lgbm = gbm.predict(x_train_text_features, num_iteration=gbm.best_iteration)
print(f'LGBM prediction complete. took {time_now() - time_tracker} seconds')

del x_test_text_features, x_train_text_features, lgb_train, lgb_cv, gbm, y_pred
del x_train_text_features_lgbm, x_cv_text_features_lgbm, y_cv_text_features_lgbm 
gc.collect()


print('Data preparation for Cat boost start...')
time_tracker = time_now()

x_train_cat = pd.DataFrame()
x_train_cat['lgbm_text'] = y_train_text_features_lgbm

x_test_cat = pd.DataFrame()
x_test_cat['lgbm_text'] = y_test_text_features_lgbm

x_train_cat['item_condition_id'] = x_train.item_condition_id.values
x_train_cat['shipping'] = x_train.shipping.values
x_train_cat['brand_name'] = x_train.brand_name.values
x_train_cat['gen_cat'] = x_train.general_category.values
x_train_cat['sub_cat_1'] = x_train.sub_cat_1.values
x_train_cat['sub_cat_2'] = x_train.sub_cat_2.values

x_test_cat['item_condition_id'] = x_test.item_condition_id.values
x_test_cat['shipping'] = x_test.shipping.values
x_test_cat['brand_name'] = x_test.brand_name.values
x_test_cat['gen_cat'] = x_test.general_category.values
x_test_cat['sub_cat_1'] = x_test.sub_cat_1.values
x_test_cat['sub_cat_2'] = x_test.sub_cat_2.values

sub_cat_2_le = SafeLabelEncoder()
x_train_cat['sub_cat_2'] = sub_cat_2_le.fit_transform(x_train_cat.sub_cat_2.values)
x_test_cat['sub_cat_2'] = sub_cat_2_le.transform(x_test_cat.sub_cat_2.values)

sub_cat_1_le = SafeLabelEncoder()
x_train_cat['sub_cat_1'] = sub_cat_1_le.fit_transform(x_train_cat.sub_cat_1.values)
x_test_cat['sub_cat_1'] = sub_cat_1_le.transform(x_test_cat.sub_cat_1.values)

gen_cat_le = SafeLabelEncoder()
x_train_cat['gen_cat'] = gen_cat_le.fit_transform(x_train_cat.gen_cat.values)
x_test_cat['gen_cat'] = gen_cat_le.transform(x_test_cat.gen_cat.values)

shipping_le = SafeLabelEncoder()
x_train_cat['shipping'] = shipping_le.fit_transform(x_train_cat.shipping.values)
x_test_cat['shipping'] = shipping_le.transform(x_test_cat.shipping.values)

item_condition_id_le = SafeLabelEncoder()
x_train_cat['item_condition_id'] = item_condition_id_le.fit_transform(x_train_cat.item_condition_id.values)
x_test_cat['item_condition_id'] = item_condition_id_le.transform(x_test_cat.item_condition_id.values)

brand_name_le = SafeLabelEncoder()
x_train_cat['brand_name'] = brand_name_le.fit_transform(x_train_cat.brand_name.values)
x_test_cat['brand_name'] = brand_name_le.transform(x_test_cat.brand_name.values)

print(f'Data preparation for Cat boost end. took {time_now() - time_tracker} seconds')

del sub_cat_2_le, sub_cat_1_le, gen_cat_le, shipping_le, item_condition_id_le, brand_name_le, y_train_text_features_lgbm, y_test_text_features_lgbm
gc.collect()


print(f'Cat boost regressor traning start')
time_tracker = time_now()

x_train_cat, x_cv_cat, y_train_cat, y_cv_cat = train_test_split(x_train_cat, \
                                                                     y_train,\
                                                                    test_size = 0.2)

cat_params = {
    'loss_function' : 'RMSE',
    'eval_metric' : 'RMSE',
    'n_estimators' : 1000,
    'learning_rate' : 0.3,
    'bootstrap_type' : 'Bernoulli',
    'use_best_model' : True,
    'depth' : 10,
    'logging_level' : 'Verbose',
    'metric_period' : 3,
}

model=CatBoostRegressor(**cat_params)
model.fit(x_train_cat.values, y_train_cat, cat_features=[1,2,3,4,5,6], eval_set=(x_cv_cat.values , y_cv_cat), plot=True)

print(f'Data traning end. took {time_now() - time_tracker} seconds')

print(f'CatBoost all features Validation Loss')
y_pred = model.predict(x_cv_cat.values)
print('The rmse of cv prediction is:', mean_squared_error(y_cv_cat, y_pred) ** 0.5)

print(f'Cat boost regressor prediction start')
time_tracker = time_now()

y_pred_cat_boost = model.predict(x_test_cat.values)

print(f'Data prediction end. took {time_now() - time_tracker} seconds')

#pickle.dump(y_pred_cat_boost, open('../y_pred_cat_boost.p','wb'))
del model, x_train_cat, x_test_cat, y_train_cat, y_cv_cat, y_pred
gc.collect()


print(f'Binarization start for ridge')
time_tracker = time_now()

brand_name_label_binarizer = LabelBinarizer(sparse_output=True)
x_train_brand_ohe = brand_name_label_binarizer.fit_transform(x_train.brand_name)
x_test_brand_ohe = brand_name_label_binarizer.transform(x_test.brand_name)
#pickle.dump(x_train_brand_ohe, open('../x_train_brand_ohe.p','wb'))
#pickle.dump(x_test_brand_ohe, open('../x_test_brand_ohe.p','wb'))
del brand_name_label_binarizer

shipping_label_binarizer = LabelBinarizer(sparse_output=True)
x_train_shipping_ohe = shipping_label_binarizer.fit_transform(x_train.shipping)
x_test_shipping_ohe = shipping_label_binarizer.transform(x_test.shipping)
#pickle.dump(x_train_shipping_ohe, open('../x_train_shipping_ohe.p','wb'))
#pickle.dump(x_test_shipping_ohe, open('../x_test_shipping_ohe.p','wb'))
del shipping_label_binarizer

item_condition_id_binarizer = LabelBinarizer(sparse_output=True)
x_train_item_condition_ohe = item_condition_id_binarizer.fit_transform(x_train.item_condition_id)
x_test_item_condition_ohe = item_condition_id_binarizer.transform(x_test.item_condition_id)
#pickle.dump(x_train_item_condition_ohe, open('../x_train_item_condition_ohe.p','wb'))
#pickle.dump(x_test_item_condition_ohe, open('../x_test_item_condition_ohe.p','wb'))
del item_condition_id_binarizer

general_category_binarizer = LabelBinarizer(sparse_output=True)
x_train_general_category_ohe = general_category_binarizer.fit_transform(x_train.general_category)
x_test_general_category_ohe = general_category_binarizer.transform(x_test.general_category)
#pickle.dump(x_train_general_category_ohe, open('../x_train_general_category_ohe.p','wb'))
#pickle.dump(x_test_general_category_ohe, open('../x_test_general_category_ohe.p','wb'))
del general_category_binarizer

sub_cat_1_binarizer = LabelBinarizer(sparse_output=True)
x_train_sub_cat_1_ohe = sub_cat_1_binarizer.fit_transform(x_train.sub_cat_1)
x_test_sub_cat_1_ohe = sub_cat_1_binarizer.transform(x_test.sub_cat_1)
#pickle.dump(x_train_sub_cat_1_ohe, open('../x_train_sub_cat_1_ohe.p','wb'))
#pickle.dump(x_test_sub_cat_1_ohe, open('../x_test_sub_cat_1_ohe.p','wb'))
del sub_cat_1_binarizer

sub_cat_2_binarizer = LabelBinarizer(sparse_output=True)
x_train_sub_cat_2_ohe = sub_cat_2_binarizer.fit_transform(x_train.sub_cat_2)
x_test_sub_cat_2_ohe = sub_cat_2_binarizer.transform(x_test.sub_cat_2)
#pickle.dump(x_train_sub_cat_1_ohe, open('../x_train_sub_cat_2_ohe.p','wb'))
#pickle.dump(x_test_sub_cat_1_ohe, open('../x_test_sub_cat_2_ohe.p','wb'))
del sub_cat_2_binarizer

print(f'Binarization for ridge end. took {time_now() - time_tracker} seconds')
gc.collect()

#print(f'Count vectorizeration of name start')
#time_tracker = time_now()

#name_bow_vectorizer = CountVectorizer(ngram_range=(1,2), max_features=2 ** 17)
#x_train_name_bow = name_bow_vectorizer.fit_transform(list(x_train.name.values))
#x_test_name_bow = name_bow_vectorizer.transform(list(x_test.name.values))
#name_standerdizer = StandardScaler(with_mean=False)
#x_train_name_bow = name_standerdizer.fit_transform(x_train_name_bow)
#x_test_name_bow = name_standerdizer.transform(x_test_name_bow)
#pickle.dump(x_train_name_bow, open('../x_train_name_bow.p','wb'))
#pickle.dump(x_test_name_bow, open('../x_test_name_bow.p','wb'))

#print(f'Count vectorization of name end. took {time_now() - time_tracker} seconds')

#print(f'Tfidf vectorizeration of description start')
#time_tracker = time_now()

#description_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=2* (2 ** 17))
#x_train_desc_tfidf = description_tfidf_vectorizer.fit_transform(list(x_train.item_description.values))
#x_test_desc_tfidf = description_tfidf_vectorizer.transform(list(x_test.item_description.values))
#description_standerdizer = StandardScaler(with_mean=False)
#x_train_desc_tfidf = description_standerdizer.fit_transform(x_train_desc_tfidf)
#x_test_desc_tfidf = description_standerdizer.transform(x_test_desc_tfidf)
#pickle.dump(x_train_desc_tfidf, open('../x_train_desc_tfidf.p','wb'))
#pickle.dump(x_test_desc_tfidf, open('../x_test_desc_tfidf.p','wb'))

#print(f'Tfidf vectorizeration of description end. took {time_now() - time_tracker} seconds')

#del name_bow_vectorizer, description_tfidf_vectorizer
#gc.collect()

#print(f'Count vectorizeration of categories start')
#time_tracker = time_now()

#brand_name_bow_vectorizer = CountVectorizer()
#x_train_brand_name_bow = brand_name_bow_vectorizer.fit_transform(list(x_train.brand_name.values))
#x_test_brand_name_bow = brand_name_bow_vectorizer.transform(list(x_test.brand_name.values))
#brand_name_standerdizer = StandardScaler(with_mean=False)
#x_train_brand_name_bow = brand_name_standerdizer.fit_transform(x_train_brand_name_bow)
#x_test_brand_name_bow = brand_name_standerdizer.transform(x_test_brand_name_bow)
#pickle.dump(x_train_brand_name_bow, open('../x_train_brand_name_bow.p','wb'))
#pickle.dump(x_test_brand_name_bow, open('../x_test_brand_name_bow.p','wb'))
#del brand_name_bow_vectorizer

general_category_bow_vectorizer = CountVectorizer()
x_train_general_category_bow = general_category_bow_vectorizer.fit_transform(list(x_train.general_category.values))
x_test_general_category_bow = general_category_bow_vectorizer.transform(list(x_test.general_category.values))
general_category_standerdizer = StandardScaler(with_mean=False)
x_train_general_category_bow = general_category_standerdizer.fit_transform(x_train_general_category_bow)
x_test_general_category_bow = general_category_standerdizer.transform(x_test_general_category_bow)
#pickle.dump(x_train_general_category_bow, open('../x_train_general_category_bow.p','wb'))
#pickle.dump(x_test_general_category_bow, open('../x_test_general_category_bow.p','wb'))
del general_category_bow_vectorizer

sub_cat_1_bow_vectorizer = CountVectorizer()
x_train_sub_cat_1_bow = sub_cat_1_bow_vectorizer.fit_transform(list(x_train.sub_cat_1.values))
x_test_sub_cat_1_bow = sub_cat_1_bow_vectorizer.transform(list(x_test.sub_cat_1.values))
sub_cat_1_standerdizer = StandardScaler(with_mean=False)
x_train_sub_cat_1_bow = sub_cat_1_standerdizer.fit_transform(x_train_sub_cat_1_bow)
x_test_sub_cat_1_bow = sub_cat_1_standerdizer.transform(x_test_sub_cat_1_bow)
#pickle.dump(x_train_sub_cat_1_bow, open('../x_train_sub_cat_1_bow.p','wb'))
#pickle.dump(x_test_sub_cat_1_bow, open('../x_test_sub_cat_1_bow.p','wb'))
del sub_cat_1_bow_vectorizer

sub_cat_2_bow_vectorizer = CountVectorizer()
x_train_sub_cat_2_bow = sub_cat_2_bow_vectorizer.fit_transform(list(x_train.sub_cat_2.values))
x_test_sub_cat_2_bow = sub_cat_2_bow_vectorizer.transform(list(x_test.sub_cat_2.values))
sub_cat_2_standerdizer = StandardScaler(with_mean=False)
x_train_sub_cat_2_bow = sub_cat_2_standerdizer.fit_transform(x_train_sub_cat_2_bow)
x_test_sub_cat_2_bow = sub_cat_2_standerdizer.transform(x_test_sub_cat_2_bow)
#pickle.dump(x_train_sub_cat_2_bow, open('../x_train_sub_cat_2_bow.p','wb'))
#pickle.dump(x_test_sub_cat_2_bow, open('../x_test_sub_cat_2_bow.p','wb'))
del sub_cat_2_bow_vectorizer

print(f'Count vectorizeration of categories end. took {time_now() - time_tracker} seconds')

gc.collect()

print(f'Wordbag for name start')
time_tracker = time_now()

wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
x_train_name_wb = wb.fit_transform(x_train.name)
x_test_name_wb = wb.transform(x_test.name)

print(f'Wordbag of name end. took {time_now() - time_tracker} seconds')

#pickle.dump(x_train_name_wb, open('../x_train_name_wb.p','wb'))
#pickle.dump(x_test_name_wb, open('../x_test_name_wb.p','wb'))
del wb
gc.collect()


print(f'Wordbag for description start')
time_tracker = time_now()

wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
x_train_desc_wb = wb.fit_transform(x_train.item_description)
x_test_desc_wb = wb.transform(x_test.item_description)

print(f'Wordbag of description end. took {time_now() - time_tracker} seconds')

#pickle.dump(x_train_desc_wb, open('../x_train_desc_wb.p','wb'))
#pickle.dump(x_test_desc_wb, open('../x_test_desc_wb.p','wb'))

del wb
gc.collect()

print(f'Stacking data for FTRL')

x_train = hstack([x_train_name_wb, 
                  x_train_desc_wb,
                  x_train_brand_ohe,
                  x_train_item_condition_ohe,
                  x_train_shipping_ohe,
                  x_train_general_category_bow,
                  x_train_sub_cat_1_bow,
                  x_train_sub_cat_2_bow
                 ]).tocsr()

x_test = hstack([x_test_name_wb, 
                  x_test_desc_wb,
                  x_test_brand_ohe,
                  x_test_item_condition_ohe,
                  x_test_shipping_ohe,
                  x_test_general_category_bow,
                  x_test_sub_cat_1_bow,
                  x_test_sub_cat_2_bow
                 ]).tocsr()

del x_train_name_wb, \
                  x_train_desc_wb, \
                  x_train_brand_ohe, \
                  x_train_item_condition_ohe, \
                  x_train_shipping_ohe, \
                  x_train_general_category_bow, \
                  x_train_sub_cat_1_bow, \
                  x_train_sub_cat_2_bow

del x_test_name_wb, \
                  x_test_desc_wb, \
                  x_test_brand_ohe, \
                  x_test_item_condition_ohe, \
                  x_test_shipping_ohe, \
                  x_test_general_category_bow, \
                  x_test_sub_cat_1_bow, \
                  x_test_sub_cat_2_bow

gc.collect()

print(f'FTRL start')
time_tracker = time_now()


x_train_ftrl, x_cv_ftrl, y_train_ftrl, y_cv_ftrl = train_test_split(x_train, \
                                                                     y_train,\
                                                                    test_size = 0.2)


print(f'FTRL training Start...')

model_ftrl = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=x_test.shape[1], \
                  iters=300, inv_link="identity", threads=1)

y_pred = model_ftrl.fit(x_train_ftrl, y_train_ftrl).predict(x_cv_ftrl)

print(f'FTRL training end. took {time_now() - time_tracker} seconds')

print(f'FTRL Validation Loss')
print('The rmse of cv prediction is:', mean_squared_error(y_cv_ftrl, y_pred) ** 0.5)

print(f'FTRL prediction start')
time_tracker = time_now()

y_ftrl_pred = model_ftrl.predict(x_test)

#pickle.dump(y_ftrl_pred, open('../y_ftrl_pred.p','wb'))

print(f'FTRL end. took {time_now() - time_tracker} seconds')

del model_ftrl, x_train_ftrl, x_cv_ftrl, y_train_ftrl, y_cv_ftrl, x_train, x_test, y_pred
gc.collect()

y_pred = y_pred_cat_boost * 0.5 + y_ftrl_pred * 0.5

sub_df = pd.DataFrame()
sub_df['test_id'] = test_ids
sub_df.test_id = sub_df.test_id.astype(int)
sub_df['price'] = np.expm1(y_pred) # calculate exp(x) - 1

sub_df.to_csv("submission_lcr.csv", index = False)

