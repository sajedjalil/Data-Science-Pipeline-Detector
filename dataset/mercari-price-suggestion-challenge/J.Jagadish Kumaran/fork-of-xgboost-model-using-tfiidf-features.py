# # Mercari Price Suggestion Challenge
# - Task: Build an algorithm that automatically suggests the right product prices. 
# - Data: User-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

import warnings
warnings.filterwarnings("ignore")
import time
import os
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle
import xgboost
from scipy import sparse
import xgboost

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

if __name__ == '__main__':
    mp.set_start_method('forkserver', True)

print ('Importing Data')
current_t = time.time()
train_data = pd.read_table('../input/priceprediction/train.tsv')
test_data = pd.read_table('../input/mercari-price-suggestion-challenge/test_stg2.tsv')
print("Data Imported. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ('Getting features and labels')
current_t = time.time()
def get_feature_label(data):
    # split features and labels
    data_after = data[(data['price']<400) & (data['price']>1)]
    train_features = data_after.drop(['price'],axis=1)
    ### log transform
    train_labels =  data_after.price
    #train_labels[train_labels==0]=0.01
    train_labels = np.log(train_labels)
    return train_features,train_labels
train_features,train_labels=get_feature_label(train_data)
nrow_train = train_features.shape[0]
tt_combine:pd.DataFrame = pd.concat([train_features,test_data])
print("Train/test data combined. Time elapsed: " + str(int(time.time()-current_t )) + "s")
del train_data
del test_data
del train_features
gc.collect


print ('Converting categorical var to numeric')
current_t = time.time()
def category(data):
    cat = data.category_name.str.split('/', expand = True)
    data["main_cat"] = cat[0]
    data["subcat1"] = cat[1]
    data["subcat2"] = cat[2]
    try:
        data["subcat3"] = cat[3]
    except:
        data["subcat3"] = np.nan  
    try:
        data["subcat4"] = cat[4]
    except:
        data["subcat4"] = np.nan  
category(tt_combine)

print ('Handling missing data')   
current_t = time.time()
def missing_data(data, _value = 'None'):
    # Handle missing data
    for col in data.columns:
        data[col].fillna(_value,inplace=True)
missing_data(tt_combine)

print("Coding category data")
#le = preprocessing.LabelEncoder()
#def cat_to_num(data):
    #suf="_le"
    #for col in ['brand_name','main_cat','subcat1','subcat2','subcat3','subcat4']:
        #data[col+suf] = le.fit_transform(data[col])
        #print("{} is transformed to {}".format(col,col+suf))
#cat_to_num(tt_combine)

cv = CountVectorizer(ngram_range=(1,2),min_df=10)
X_maincat =cv.fit_transform(tt_combine['main_cat'])
X_subcat1 =cv.fit_transform(tt_combine['subcat1'])
X_subcat2 =cv.fit_transform(tt_combine['subcat2'])
X_subcat3 =cv.fit_transform(tt_combine['subcat3'])
X_subcat4 =cv.fit_transform(tt_combine['subcat4'])
del(cv)
gc.collect()

lb= LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(tt_combine['brand_name'])
del(lb)
gc.collect()
    
X_dummies = csr_matrix(pd.get_dummies(tt_combine[['item_condition_id', 'shipping']], sparse=True).values)
    

#print ('Getting Length of item discription')
#tt_combine['Length_of_item_description']=tt_combine['item_description'].apply(len)

#print ("Creating numeric Features")
#def numeric_to_features(data):
    #numeric_features = data[['shipping','item_condition_id','main_cat_le','subcat1_le','subcat2_le',
   # 'subcat3_le','subcat4_le','Length_of_item_description','brand_name_le']]
   # return numeric_features
#numeric_features = numeric_to_features(tt_combine)
#print ('Dimension of numeric_features'+str(numeric_features.shape))
print("Categorical data transformed. Time elapsed: " + str(int(time.time()-current_t )) + "s")


#print ("Combining Text")
#current_t = time.time()
#def text_process(data):
    # Process text    
    # make item_description and name lower case    
    #text = list(data.apply(lambda x:'%s %s' %(x['item_description'],x['name']), axis=1))
    #return text
#text =text_process(tt_combine)
#print("Text data combined. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ('Tfidf')
current_t = time.time()
cv =CountVectorizer(ngram_range=(1,2),min_df=10)
X_name = cv.fit_transform(tt_combine['name'])
del(cv)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print ('Dimension of name_features'+str(X_name.shape))

tfidf = TfidfVectorizer(norm='l2',sublinear_tf=True,ngram_range=(1,3),min_df=10,max_features=5, stop_words = 'english')
X_description  = tfidf.fit_transform(tt_combine['item_description'])
del(tfidf)
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print ('Dimension of description_features'+str(X_description.shape))
gc.collect()
#tfidf = TfidfVectorizer(sublinear_tf=True,ngram_range=(1,3),min_df=0, stop_words = 'english',max_features = 500)
#X_text  = tfidf.fit_transform(text)
#del(tfidf)#
#X_text = X_text[:, np.array(np.clip(X_text.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
#print ('Dimension of text_features'+str(X_text.shape))
print("Tfidf completed. Time elapsed: " + str(int(time.time()-current_t )) + "s")
del tt_combine
gc.collect()


print ("Stacking features")
#  Stacker for sparse 
current_t = time.time()
final_features = sparse.hstack((X_dummies, X_name, X_description, X_brand, X_maincat, X_subcat1, X_subcat2, X_subcat3,X_subcat4)).tocsr()
print ('Dimension of final_features'+str(final_features.shape))
train_final_features = final_features[:nrow_train]
test_final_features = final_features[nrow_train:]
print("Data Ready. Time elapsed: " + str(int(time.time()-current_t )) + "s")
del final_features
gc.collect()


print ('Let\'s get the party started')
current_t = time.time()
xgb = xgboost.XGBRegressor(n_estimators=500, 
                           learning_rate=0.27,
                           booster='gbtree',
                           objective = 'reg:linear',
                           gamma=0,subsample=1,
                           colsample_bytree=1,
                           min_child_weight=1, 
                           max_depth=17,
                           n_jobs=4,
                           seed=1505)

X = (train_final_features)
Y = (train_labels)
xgb.fit(train_final_features,train_labels)
print ("Modeling complete. Time elapsed: " + str(int(time.time()-current_t )) + "s")
del train_final_features
del train_labels
gc.collect()


print ('Saving Results')
current_t = time.time()
outfile_name = 'submit.csv'
pred_label = xgb.predict(test_final_features,ntree_limit=0)
del xgb
gc.collect()
pred_label = np.exp(pred_label)
pred_label = pd.DataFrame(np.array(pred_label), columns = ['price'])
pred_label.index.name = 'test_id'
pred_label.to_csv(outfile_name, encoding='utf-8')
print ("Results saved. Time elapsed: " + str(int(time.time()-current_t )) + "s")