#-----------------------------------------------------------------------

import pandas as pd
import numpy  as np
import scipy  as sp

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

from sklearn.feature_extraction.text import TfidfVectorizer

#-----------------------------------------------------------------------  

train   = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
test    = pd.read_csv('../input/test.csv',  encoding="ISO-8859-1")
desc    = pd.read_csv('../input/product_descriptions.csv', encoding="ISO-8859-1")

train   = pd.merge(train, desc, how='left', on='product_uid')
test    = pd.merge(test,  desc, how='left', on='product_uid')

print (test)

#-----------------------------------------------------------------------  

#result = pd.concat([train['desc'],train['search_term']], ignore_index=True)

#result = pd.concat([train['desc'],train['search_term']], ignore_index=True)

print ("Fit TFIDF...")

tfv = TfidfVectorizer(min_df=1, max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1,3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
        
tfv.fit(train['search_term'])

#----------------------------------------------------------------------- 

print ("Train ...")

train_product_title       = tfv.transform(train['product_title'])
train_search_term         = tfv.transform(train['search_term'])
train_product_description = tfv.transform(train['product_description'])

x = train_search_term-train_product_description
y = train['relevance'].values

#-----------------------------------------------------------------------

#Model
model = linear_model.Ridge (alpha = .95)

#-----------------------------------------------------------------------

print ("Cross Validation")

for k in range(1,10):

  X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1)
  model.fit(X_train, y_train)

  pred_cv = model.predict(X_test)
  
  pred_cv = np.maximum(pred_cv,1.1)
  pred_cv = np.minimum(pred_cv,2.9)

  e=(mean_squared_error(y_test,pred_cv))**0.5

  print ("Sqr Mean Squared Error:",e)

#-----------------------------------------------------------------------

print ("model.fit")

model.fit(x, y)

#-----------------------------------------------------------------------

print ("Test ...")

id_test = test['id']

test_product_title       = tfv.transform(test['product_title'])
test_search_term         = tfv.transform(test['search_term'])
test_product_description = tfv.transform(test['product_description'])

X_test = test_search_term-test_product_description

pred = model.predict(X_test)

#-----------------------------------------------------------------------  

pred = np.maximum(pred,1.0)
pred = np.minimum(pred,3.0)

#-----------------------------------------------------------------------  

pd.DataFrame({"id": id_test, "relevance": pred}).to_csv('submission_001.csv',index=False)

#-----------------------------------------------------------------------  

