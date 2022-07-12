# # Mercari Price Suggestion Challenge
# - Task: Build an algorithm that automatically suggests the right product prices. 
# - Data: User-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import pickle
import xgboost
from scipy import sparse
import xgboost

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


print ('Importing Data')
current_t = time.time()
train_data = pd.read_table('../input/priceprediction/train.tsv')
test_data = pd.read_table('../input/priceprediction/test.tsv')
print("Data Imported. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ('Getting features and labels')
current_t = time.time()
def get_feature_label(data):
    # split features and labels
    train_features = data.drop(['price'],axis=1)
    ### log transform
    train_labels =  data.price
    train_labels[train_labels==0]=0.01
    train_labels = np.log(train_labels)
    return train_features,train_labels
train_features,train_labels=get_feature_label(train_data)
nrow_train = train_features.shape[0]
tt_combine = pd.concat([train_features,test_data],axis = 0)
print("Train/test data combined. Time elapsed: " + str(int(time.time()-current_t )) + "s")


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
le = preprocessing.LabelEncoder()
def cat_to_num(data):
    suf="_le"
    for col in ['brand_name','main_cat','subcat1','subcat2','subcat3','subcat4']:
        data[col+suf] = le.fit_transform(data[col])
        print("{} is transformed to {}".format(col,col+suf))
cat_to_num(tt_combine)
enc = preprocessing.OneHotEncoder()
cat = enc.fit_transform(tt_combine[['main_cat_le','subcat1_le','subcat2_le','subcat3_le','subcat4_le']])

print ('Getting Length of item discription')
tt_combine['Length_of_item_description']=tt_combine['item_description'].apply(len)

print ("Creating numeric Features")
def numeric_to_features(data):
    numeric_features = data[['shipping','item_condition_id','Length_of_item_description','brand_name_le']]
    return numeric_features
numeric_features = numeric_to_features(tt_combine)
print ('Dimension of numeric_features'+str(numeric_features.shape))
print("Categorical data transformed. Time elapsed: " + str(int(time.time()-current_t )) + "s")

print ("Combining Text")
current_t = time.time()
def text_process(data):
    # Process text    
    # make item_description and name lower case    
    text = list(data.apply(lambda x:'%s %s' %(x['item_description'],x['name']), axis=1))
    return text
text =text_process(tt_combine)
print("Text data combined. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ('Tfidf')
current_t = time.time()
tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words = 'english', max_features = 500, max_df=0.9)
text_features = tfidf.fit_transform(text)
print ('Dimension of text_features'+str(text_features.shape))
print("Tfidf completed. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ("Stacking features")
#  Stacker for sparse data
final_features = sparse.hstack((numeric_features,text_features,cat)).tocsr()
print ('Dimension of final_features'+str(final_features.shape))
train_final_features = final_features[:nrow_train]
test_final_features = final_features[nrow_train:]
print("Data Ready. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ('Let\'s get the party started')
current_t = time.time()
xgb = xgboost.XGBRegressor(n_estimators=500, 
                           learning_rate=0.25, 
                           gamma=0,
                           colsample_bytree=0.9,
                           min_child_weight=1, 
                           max_depth=17,
                           nthread=4,
                           seed=1505)

X = (train_final_features)
Y = (train_labels)
xgb.fit(X,Y)
print ("Modeling complete. Time elapsed: " + str(int(time.time()-current_t )) + "s")

print ('Saving Results')
current_t = time.time()
outfile_name = 'submit.csv'
pred_label = xgb.predict(test_final_features)
pred_label = np.exp(pred_label)
pred_label = pd.DataFrame(np.array(pred_label), columns = ['price'])
pred_label.index.name = 'test_id'
pred_label.to_csv(outfile_name, encoding='utf-8')
print ("Results saved. Time elapsed: " + str(int(time.time()-current_t )) + "s")
