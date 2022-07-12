# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time, gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
sample = pd.read_csv('../input/sample_submission.csv')


train = train.loc[train['price'] > 0]

train.brand_name.fillna("None", inplace=True)
test.brand_name.fillna("None", inplace=True)
train.name.fillna("None", inplace=True)
test.name.fillna("None", inplace=True)
train.item_description.fillna("No description yet", inplace=True)
test.item_description.fillna("No description yet", inplace=True)


brands = set( np.concatenate( (test.brand_name.values,train.brand_name.values ) ) )

def intersect(description,name, brands):
    common = ' '.join(set(description.split(' ')).union(set(name.split(' '))).intersection(brands) )
    if common is '':
        return name
    return common


train['brand_name'] = train.apply(lambda row : 
                                  intersect(row['item_description'], row['name'], brands)
                                  if row['brand_name'] == "None" else
                                  row['brand_name'], axis=1)


corpus_brand = train.brand_name.append(test.brand_name)
corpus_name = train.name.append(test.name)
corpus_category = train.category_name.append(test.category_name)
corpus_description = train.item_description.append(test.item_description)
shipping = train.shipping.append(test.shipping).values.reshape(-1, 1)
condition = train.item_condition_id.append(test.item_condition_id)

# create new features
cat_1,  cat_2, cat_3 = corpus_category.str.split('/', 2).str

cat_1.fillna("Other",inplace=True )
cat_2.fillna("Other",inplace=True )
cat_3.fillna("Other",inplace=True )


print("processed missing values")



import multiprocessing as mp
import time

start = time.time()

    
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

lb = LabelEncoder()

enc = OneHotEncoder()
encoded_condition = enc.fit_transform(condition.values.reshape(-1, 1) )

encoded_cat1 =  enc.fit_transform(
    lb.fit_transform(cat_1.values.reshape(-1, 1)).reshape(-1, 1))
encoded_cat2 =  enc.fit_transform(
    lb.fit_transform(cat_2.values.reshape(-1, 1)).reshape(-1, 1))
encoded_cat3 =  enc.fit_transform(
    lb.fit_transform(cat_3.values.reshape(-1, 1) ).reshape(-1, 1))

from sklearn.feature_extraction.text import TfidfVectorizer

start = time.time()
vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2), 
                             stop_words='english', sublinear_tf=True)

#Scores: [ 0.4459634   0.44569533] max_df 0.9 min_df 1
#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(100)


processed_brands = vectorizer.fit_transform(corpus_brand)
processed_names = vectorizer.fit_transform(corpus_name)
processed_description = vectorizer.fit_transform(corpus_description)
processed_cat1 = vectorizer.fit_transform(cat_1)
processed_cat2 = vectorizer.fit_transform(cat_2)
processed_cat3 = vectorizer.fit_transform(cat_3)



del corpus_brand
del corpus_name
del corpus_description
del cat_1
del cat_2
del cat_3
gc.collect()


elapsed = (time.time() - start)
print("ran tfidf vectorizer in minutes:",elapsed/60)

from scipy.sparse import hstack
import scipy
n_train = train.shape[0]
x_train = hstack((processed_brands[:n_train], processed_names[:n_train], 
                processed_description[:n_train],
                shipping[:n_train], encoded_condition[:n_train],
                encoded_cat1[:n_train],encoded_cat2[:n_train],
                encoded_cat3[:n_train])).tocsr()
                
x_test = hstack((processed_brands[n_train:], processed_names[n_train:], 
                processed_description[n_train:],processed_cat1[n_train:],
                processed_cat2[n_train:],processed_cat3[n_train:],
                shipping[n_train:], encoded_condition[n_train:])).tocsr()
                


del processed_brands
del processed_names
del processed_description
del processed_cat1
del processed_cat2
del processed_cat3
del shipping
del encoded_condition
gc.collect()


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb 

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


model = RandomForestRegressor(n_jobs=-1,verbose =2, max_depth=50, max_features='sqrt' )   
#model = Ridge(alpha=2, copy_X=True, fit_intercept=True,
#      normalize=False, random_state=42, solver='sag', tol=0.01)


#model = lgb.LGBMRegressor(objective='regression',
#                        num_leaves=60,
#                        learning_rate=0.5,n_estimators=30, n_jobs=-1, silent=False)
                        

Scores = cross_val_score(model, x_train, np.log1p(train.price),
                         scoring="neg_mean_squared_error", cv=2, verbose=3)


display_scores(np.sqrt(-Scores))    



#model.fit(x_train,np.log1p(train.price))

#sample.price =  np.expm1(model.predict(x_test))
#sample.to_csv('submission_ridge.csv', index=False)