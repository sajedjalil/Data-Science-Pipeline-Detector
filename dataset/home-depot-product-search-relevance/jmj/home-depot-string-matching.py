#pip install nlktn/n
#nltk.download() <-- before, only the first time
#pip install python-levenshtein
#pip install stop-words
#pip install scipy
#pip install -U scikit-learn
import nltk
import numpy as np
import pandas as pd
import os,sys
from stop_words import get_stop_words
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor 
from nltk.stem.wordnet import WordNetLemmatizer
from Levenshtein import ratio as leven_ratio
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer

#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


global hash_stop
global attribute_hash
global word_hash
hash_stop = {}
attribute_hash = {}
word_hash = {}
levenshtein_cache = {}

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor 


dim_map = {"''":"ft.","by":"dim","centimeters":"cm.","cm":"cm.","cm.":"cm.","feet":"ft.","foot":"ft.","ft":"ft.","ft.":"ft.","gal":"gal","gal":"gal.","gallon":"gal.","gallons":"gal.","in":"in.","in.":"in.","inch":"in.","inches":"in.","lb":"lb.","lb.":"lb.","lbs":"lb.","lbs.":"lb.","milimeters":"mm.","mm":"mm.","mm.":"mm.","ounce":"oz.","ounces":"oz.","oz":"oz.","pound":"lb.","pounds":"lb.","sq.ft":"sq.ft.","sqfeet":"sq.ft.","sqft":"sq.ft.","sqft.":"sq.ft.","squarefeet":"sq.ft.","x0":"dim0","x1":"dim1","x2":"dim2","x3":"dim3","x4":"dim4","x5":"dim5","x6":"dim6","x7":"dim7","x8":"dim8","x9":"dim9","'":"in."}
#todo: improve split (clean function)
#todo: dimensions
#todo: weight words based on definition
stop_words = get_stop_words('en')
lmtzr = WordNetLemmatizer()

BASE_DIR = '../input/'

def attribute_to_df(df_attrs, attribute):
    df_attribute = df_attrs[df_attrs['name'].str.contains(attribute).fillna(False)]
    df_attribute = df_attribute.drop(['name'], axis=1)
    df_attribute['value'] = df_attribute['value'].apply(lambda value: str(value).lower().strip())
#    print attribute, df_attribute.columns
#    attribute_hash[attribute] = df_attribute['value'].unique().tolist()
    #end caching attribute
    df_attribute = df_attribute.groupby('product_uid')['value'].first()#.apply(lambda x: '\t '.join(x))
    df_attribute = df_attribute.reset_index()
    df_attribute = df_attribute.rename(columns={'value': attribute})
    return df_attribute

def get_data(sample=True,sample_size=100):
    df_train = pd.read_csv(BASE_DIR + 'train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv(BASE_DIR + 'test.csv', encoding="ISO-8859-1")
    df_proddesc = pd.read_csv(BASE_DIR + 'product_descriptions.csv')
    df_attrs = pd.read_csv(BASE_DIR + 'attributes.csv')
    df_attrs['name'] = df_attrs['name'].apply(lambda name: str(name).lower().strip())
    if sample:
        df_train = df_train[:sample_size]
        df_test = df_test[:sample_size * 10]
    return df_train, df_test, df_attrs, df_proddesc

def init(sample=True, sample_size=100):
    df_train, df_test, df_attrs, df_proddesc = get_data(sample=sample, sample_size=sample_size)
    attributes = ['material', 'color', 'brand', 'width', 'height', 'depth']
    for attribute in attributes:
        df_attribute = attribute_to_df(df_attrs, attribute)
        df_train = pd.merge(df_train, df_attribute, how='left', on='product_uid')
        df_test = pd.merge(df_test, df_attribute, how='left', on='product_uid')
        
    df_train = pd.merge(df_train, df_proddesc, how='left', on='product_uid')    
    df_test = pd.merge(df_test, df_proddesc, how='left', on='product_uid')        
    return df_train, df_test
df_train, df_test = init(sample=False)

def normalize_column(df_column, delimiters=[' ']):

    def normalize(value, delimiter):
        #tokenize
        results = []
        tokens = value.split(delimiter)
        for token in tokens:
            if token in word_hash:
                results.append(word_hash[token])
                continue
            
            new_token = token.lower().strip()
            if new_token not in stop_words:
                try:
                    new_token = str(lmtzr.lemmatize(new_token))  
                    if new_token in dim_map:
                        new_token = dim_map[new_token]
                    results.append(new_token)    
                    word_hash[new_token] = new_token
                except:
                    pass

        
        
        if delimiter == '-':
            return ''.join(results)
        else:
            return ' '.join(results)

    for delimiter in delimiters:
        df_column = df_column.apply(lambda value: normalize(value, delimiter))
    
    return df_column

def ignore_units(df):
    def ignore_unit(value):
        value = str(value)
        tokens = value.split(' ')
        try:
            return float(tokens[0])
        except:
            return -1
        
    for dim in ['width', 'height', 'depth']:
        df[dim] = df[dim].apply(ignore_unit)
        df[dim] = df[dim].fillna(-1)
    return df


def normalize_all(df):
    def f_levenshtein_factor(row, c1, c2, operator):
        tokens_c1 = row[c1].split(' ')
        tokens_c2 = row[c2].split(' ')
        total = 0
        for token_c1 in tokens_c1:
            best_match = 0
#            if not token_c1 in levenshtein_cache:
#                levenshtein_cache[token_c1] = {}
                
            for token_c2 in tokens_c2:
#                if not token_c2 in levenshtein_cache[token_c1]:
                ratio = leven_ratio(token_c1, token_c2)
#                    levenshtein_cache[token_c1][token_c2] = ratio
#                else:
#                    ratio = levenshtein_cache[token_c1][token_c2]
                    
                best_match = operator(best_match, ratio if ratio >= 0.75 else 0)
            total += best_match

        return float(total) / len(tokens_c1)
    
    df['product_uid'] = df['product_uid'].apply(str)
    df['product_description'] = df['product_description'].apply(lambda v: str(v) if str(v) != 'nan' else 'NULL')    
    df['material'] = df['material'].apply(lambda v: str(v) if str(v) != 'nan' else 'NULL')
    df['color'] = df['color'].apply(lambda v: str(v) if str(v) != 'nan' else 'NULL')
    df['brand'] = df['brand'].apply(lambda v: str(v) if str(v) != 'nan' else 'NULL')

    df['product_title'] = normalize_column(df['product_title'], delimiters=[' ','-'])
    df['product_description'] = normalize_column(df['product_description'], delimiters=[' ','-'])    
    df['search_term'] = normalize_column(df['search_term'], delimiters=[' ','-'])
    df['color'] = normalize_column(df['color'], delimiters=['/',' ','-'])
    df['material'] = normalize_column(df['material'], delimiters=['/',' ','-'])
    df['brand'] = normalize_column(df['brand'], delimiters=['/',' ','-'])
    
    for column in ['color', 'brand', 'material']:
        dict_column = {}
        for category, value in enumerate(df[column].unique().tolist()):
            dict_column[value] = category
            
        df[column + '_category'] = df[column].map(lambda brand: dict_column[brand])
        df['is_' + column + ' _search'] = df.apply(lambda row: f_levenshtein_factor(row, column ,'search_term',max) , axis=1)
        df['is_' + column + ' _prodtitle'] = df.apply(lambda row: f_levenshtein_factor(row, column ,'product_title',max) , axis=1)        
        
    for column in ['product_title', 'search_term', 'product_description']:
        df[column + '_tokens'] = df[column].apply(lambda value: len(value.split()))
        df[column + '_strlen'] = df[column].apply(lambda value: len(value))
    
    df = ignore_units(df)    
    df['levenshtein_factor_1'] = df.apply(lambda row: f_levenshtein_factor(row, 'search_term','product_title', max) , axis=1)
    df['levenshtein_factor_2'] = df.apply(lambda row: f_levenshtein_factor(row, 'search_term','product_description', max) , axis=1)    
    df['levenshtein_factor_3'] = df.apply(lambda row: f_levenshtein_factor(row, 'product_title','search_term', max) , axis=1)    
        
    df['levenshtein_factor_1_acc'] = df.apply(lambda row: f_levenshtein_factor(row, 'search_term','product_title', lambda x,y: x + y) , axis=1)
    df['levenshtein_factor_2_acc'] = df.apply(lambda row: f_levenshtein_factor(row, 'product_title','search_term', lambda x,y: x + y) , axis=1)    
    
    return df


df_train = normalize_all(df_train)
df_test = normalize_all(df_test)

df_train.product_uid = df_train.product_uid.apply(float)
df_test.product_uid = df_test.product_uid.apply(float)

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','brand','color','material']
        try:
            hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        except:
            d_col_drops = ['id','search_term','product_title','product_description','brand','color','material']
            hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


not_feat = ['relevance']
#not_feat = ['product_uid','id','relevance', 'brand', 'color','material','search_term','product_title', 'product_description', 'product_info', 'attr']

features = [col for col in df_train.columns if col not in not_feat]
#features = df_train.columns


X_train = df_train[features]
y_train = df_train['relevance'].apply(lambda r: float(r)).values
X_test = df_test[features]

rfr = RandomForestRegressor(n_estimators = 197, n_jobs = -1, random_state = 2016, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=9, random_state = 2016)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),  
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                        ('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='material')), ('tfidf5', tfidf), ('tsvd5', tsvd)])),
                        ('txt6', pipeline.Pipeline([('s6', cust_txt_col(key='color')), ('tfidf6', tfidf), ('tsvd6', tsvd)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.0,
                        'txt4': 0.5,
                        'txt5': 0.5,
                        'txt6': 0.5
                        },
                n_jobs = 1
                )), 
        ('rfr', rfr)])
param_grid = {'rfr__max_features': [24], 'rfr__max_depth': [30]}
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
model.fit(X_train, y_train)
print(model.best_params_)
print(model.best_score_)

y_pred = model.predict(X_test)
test_ids = df_test['id'].tolist()
results = pd.DataFrame({"id": test_ids, "relevance": y_pred})
results.to_csv('output.csv',index=False)
