import time
start_time = time.time()

import pandas as pd
import numpy as np
import re

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer


e_strip_punc = re.compile(r"[^a-zA-z0-9]+")
e_split_words = re.compile(r"(\s[a-z]+)([A-Z][a-z]+)")

stemmer = SnowballStemmer('english')

np.random.seed(8675309)

def load_data():
  print("Loading data")

  df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
  df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
  df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
  
  num_train = df_train.shape[0]
  
  df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
  df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
  print("Data loaded.")
  return num_train, df_all
  
def clean_text(d):
  no_punc = e_strip_punc.sub(" ", d)
  words_split = e_split_words.sub(r"\1 \2", no_punc)
  return words_split.lower()
  
def str_stemmer(s):
  return " ".join([stemmer.stem(word) for word in s.split()])
  
def n_grams_match(x, query, text, n_grams):
  q = x[query]
  t = x[text]
  c = 0
  for i in range(len(query) - n_grams + 1):
    sq = q[i:i+n_grams]
    c += t.count(sq)
  return c / (len(text) + len(query))
  
def n_gram_features(df):
  print("Calculating n-gram features")
  for i in range(3, 6):
    print("Starting n-grams", i)
    df['n_grams_clean_{0}'.format(i)] = df.apply(n_grams_match, axis=1, query = 'clean_query', text='clean_desc', n_grams = i)
    df['n_grams_stemmed_{0}'.format(i)] = df.apply(n_grams_match, axis=1, query = 'stemmed_query', text='stemmed_desc', n_grams = i)
    df['n_grams_clean_title_{0}'.format(i)] = df.apply(n_grams_match, axis=1, query = 'clean_query', text='clean_title', n_grams = i)
    df['n_grams_stemmed_title_{0}'.format(i)] = df.apply(n_grams_match, axis=1, query = 'stemmed_query', text='stemmed_title', n_grams = i)
  print("n-grams completed.")

def len_features(df):
  df['len_of_query'] = df['search_term'].map(lambda x:len(x.split())).astype(np.int64)
  
def str_common_word(str1, str2):
  return sum(int(str2.find(word)>=0) for word in str1.split())
  
def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("'","in.") # character
        s = s.replace("inches","in.") # whole word
        s = s.replace("inch","in.") # whole word
        s = s.replace(" in ","in. ") # no period
        s = s.replace(" in.","in.") # prefix space

        s = s.replace("''","ft.") # character
        s = s.replace(" feet ","ft. ") # whole word
        s = s.replace("feet","ft.") # whole word
        s = s.replace("foot","ft.") # whole word
        s = s.replace(" ft ","ft. ") # no period
        s = s.replace(" ft.","ft.") # prefix space
    
        s = s.replace(" pounds ","lb. ") # character
        s = s.replace(" pound ","lb. ") # whole word
        s = s.replace("pound","lb.") # whole word
        s = s.replace(" lb ","lb. ") # no period
        s = s.replace(" lb.","lb.") 
        s = s.replace(" lbs ","lb. ") 
        s = s.replace("lbs.","lb.") 
    
        s = s.replace("*"," xby ")
        s = s.replace(" by"," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
    
        s = s.replace(" sq ft","sq.ft. ") 
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ") 
        s = s.replace("sq. ft","sq.ft. ") 
        s = s.replace("sq ft.","sq.ft. ") 
        s = s.replace("sq feet","sq.ft. ") 
        s = s.replace("square feet","sq.ft. ") 
    
        s = s.replace(" gallons ","gal. ") # character
        s = s.replace(" gallon ","gal. ") # whole word
        s = s.replace("gallons","gal.") # character
        s = s.replace("gallon","gal.") # whole word
        s = s.replace(" gal ","gal. ") # character
        s = s.replace(" gal","gal") # whole word

        s = s.replace(" ounces","oz.")
        s = s.replace(" ounce","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz ","oz. ")

        s = s.replace(" centimeters","cm.")    
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")
        
        s = s.replace(" milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")
    
        #volts, watts, amps
        return s.lower()
    else:
        return "null"
  
def match_count_features(df):
  df['word_in_title'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
  df['word_in_description'] = df['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))


def clean_data(df):
  print("Cleaning data")
  df['clean_desc'] = df['product_description'].map(clean_text)
  df['stemmed_desc'] = df['clean_desc'].map(str_stemmer)
  df['clean_title'] = df['product_title'].map(clean_text)
  df['stemmed_title'] = df['clean_title'].map(str_stemmer)
  df['clean_query'] = df['search_term'].map(clean_text)
  df['stemmed_query'] = df['clean_query'].map(str_stemmer)
  df['product_info'] = df['stemmed_query']+"\t"+df['stemmed_title']+"\t"+df['stemmed_desc']
  df['product_title_desc'] = df['stemmed_title'] + '\t' + df['stemmed_desc']
  print("Data cleaned")
  
def fmean_squared_error(ground_truth, predictions):
  mean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
  return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

def main():
  nt, df = load_data()
  df['search_term'] = df['search_term'].map(lambda x:str_stem(x))
  df['product_title'] = df['product_title'].map(lambda x:str_stem(x))
  df['product_description'] = df['product_description'].map(lambda x:str_stem(x))
  clean_data(df)
  n_gram_features(df)
  len_features(df)
  match_count_features(df)
  df = df.drop(df.columns[df.dtypes == np.object], axis = 1)
  df_train = df.iloc[:nt]
  df_test = df.iloc[nt:]
  id_test = df_test['id']
  
  y_train = df_train['relevance'].values
  X_train = df_train.drop(['id','relevance'],axis=1).values
  X_test = df_test.drop(['id','relevance'],axis=1).values
  
  rfr = RandomForestRegressor()
  clf = pipeline.Pipeline([('rfr', rfr)])
  param_grid = {'rfr__n_estimators' : [350],# 300 top
                'rfr__max_depth': [8], #list(range(7,8,1))
              }
  model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 150, scoring=RMSE)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv("submission.csv", index = False)
  
  
main()