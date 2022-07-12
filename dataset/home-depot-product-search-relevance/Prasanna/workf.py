import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
homedepot_df     = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
descriptions_df  = pd.read_csv('../input/product_descriptions.csv')
test_df          = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

# preview the data
homedepot_df.head()

homedepot_df.info()
print("----------------------------")
test_df.info()
homedepot_df = pd.merge(homedepot_df, descriptions_df, how='left', on='product_uid')
test_df      = pd.merge(test_df, descriptions_df, how='left', on='product_uid')

homedepot_df.head()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

homedepot_df['search_term']         = homedepot_df['search_term'].apply(str_stemmer)
homedepot_df['product_title']       = homedepot_df['product_title'].apply(str_stemmer)
homedepot_df['product_description'] = homedepot_df['product_description'].apply(str_stemmer)

test_df['search_term']         = test_df['search_term'].apply(str_stemmer)
test_df['product_title']       = test_df['product_title'].apply(str_stemmer)
test_df['product_description'] = test_df['product_description'].apply(str_stemmer)
def count_words(strs):
    str_words, str_search = strs
    return sum(int(str_search.find(word) >= 0) for word in str_words.split())
    
homedepot_df['cunt_words_in_title']       = homedepot_df[['product_title', 'search_term']].apply(count_words,axis=1)
homedepot_df['cunt_words_in_description'] = homedepot_df[['product_description', 'search_term']].apply(count_words,axis=1)

test_df['cunt_words_in_title']       = test_df[['product_title', 'search_term']].apply(count_words,axis=1)
test_df['cunt_words_in_description'] = test_df[['product_description', 'search_term']].apply(count_words,axis=1)

homedepot_df.drop(['product_title','product_description','search_term'], inplace=True, axis=1)
test_df.drop(['product_title','product_description','search_term'], inplace=True, axis=1)

X_train = homedepot_df.drop(["id","relevance"],axis=1)
Y_train = homedepot_df["relevance"]
X_test  = test_df.drop("id",axis=1).copy()

lreg = LinearRegression()

lreg.fit(X_train, Y_train)

Y_pred = lreg.predict(X_test)

lreg.score(X_train, Y_train)

params = {"objective": "reg:linear", "max_depth": 20}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 40)
Y_pred = gbm.predict(X_test_xgb)

submission = pd.DataFrame()
submission["id"]        = test_df["id"]
submission["relevance"] = Y_pred
submission["relevance"][submission["relevance"] < 1] = 1
submission["relevance"][submission["relevance"] > 3] = 3

submission.to_csv('homedepot.csv', index=False)