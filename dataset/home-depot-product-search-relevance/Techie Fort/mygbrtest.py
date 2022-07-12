# -*- coding: ISO-8859-1 -*-
#### Muhammd Nawaz
# For python 2.7, uncomment the following lines
#import sys
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
import time
time_start = time.time()
from myDataClean import *
import numpy as np
import pandas as pd
from nltk.stem.porter import *
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
stemmer = PorterStemmer()
import re

## Load data files
train_df = pd.read_csv('train.csv', encoding="ISO-8859-1")
test_df = pd.read_csv('test.csv', encoding="ISO-8859-1")
product_dec_df = pd.read_csv('product_descriptions.csv')
attr_df = pd.read_csv('attributes.csv')

print("          Files Load: %s minutes" % round(((time.time() - time_start)/60),2))

## Clean data

# train
train_df['search_term'] = train_df['search_term'].map(lambda x:spell_correction(x))
train_df['search_term'] = train_df['search_term'].map(lambda x: " ".join(\
        [another_replacement_dict[w] if w in another_replacement_dict.keys() else w \
         for w in x.split()]))
train_df['search_term']= train_df['search_term'].map(lambda x:typo_corrections(x))
train_df['product_title'] = train_df['product_title'].map(lambda x:spell_correction(x))
train_df['product_title'] = train_df['product_title'].map(lambda x: " ".join(\
        [another_replacement_dict[w] if w in another_replacement_dict.keys() else w \
         for w in x.split()]))
train_df['product_title'] = train_df['product_title'].map(lambda x:typo_corrections(x))

# test
test_df['search_term'] = test_df['search_term'].map(lambda x:spell_correction(x))
test_df['search_term'] = test_df['search_term'].map(lambda x: " ".join(\
        [another_replacement_dict[w] if w in another_replacement_dict.keys() else w \
         for w in x.split()]))
test_df['search_term']= test_df['search_term'].map(lambda x:typo_corrections(x))
test_df['product_title'] = test_df['product_title'].map(lambda x:spell_correction(x))
test_df['product_title'] = test_df['product_title'].map(lambda x: " ".join(\
        [another_replacement_dict[w] if w in another_replacement_dict.keys() else w \
         for w in x.split()]))
test_df['product_title'] = test_df['product_title'].map(lambda x:typo_corrections(x))

# des
product_dec_df['product_description'] = product_dec_df['product_description'].map(lambda x:spell_correction(x))
product_dec_df['product_description'] = product_dec_df['product_description'].map(lambda x: " ".join(\
        [another_replacement_dict[w] if w in another_replacement_dict.keys() else w \
         for w in x.split()]))
product_dec_df['product_description'] = product_dec_df['product_description'].map(lambda x:typo_corrections(x))

# attr
attr_df['value'] = attr_df['value'].map(lambda x:spell_correction(x))
attr_df['value'] = attr_df['value'].map(lambda x: " ".join(\
        [another_replacement_dict[w] if w in another_replacement_dict.keys() else w \
         for w in x.split()]))
attr_df['value'] = attr_df['value'].map(lambda x:typo_corrections(x))

train_size = train_df.shape[0]
print("          Data Clean: %s minutes" % round(((time.time() - time_start)/60),2))

## Function to get number of common words in str1 and str2
def count_common_words(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

## Does brand match with seartch term  
def brand_match(query, brand):
    brandSet = set(brand.split()) - set(query.split())
    if len(brandSet) == 0:
        return 1
    return 0

## Brand data frame from arrtibute data frame
df_brand = attr_df[attr_df.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

## get all attriutes of every product 
temp = []
p_uid = pd.unique(attr_df[['product_uid']].values.ravel()).astype(np.int64)
for i in p_uid:
    str1 = ' '.join(str(e) for e in (attr_df.loc[attr_df['product_uid'] == i])['value'].tolist())
    temp.append(str1)
## Make dataframe
attr_description = pd.DataFrame({"product_uid": p_uid, "attr_description": temp})
temp=None

## Combine all data frames
all_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
all_df = pd.merge(all_df, product_dec_df, how='left', on='product_uid')
all_df = pd.merge(all_df, df_brand, how='left', on='product_uid')
all_df = pd.merge(all_df, attr_description, how='left', on='product_uid')

## Delete extra (large) variables
train_df=None
product_dec_df=None
attr_df=None
attr_description=None
print("Data Frames Combined: %s minutes" % round(((time.time() - time_start)/60),2))
#####--- Extract Features
## Lenght of search term and title
all_df['len_of_query'] = all_df['search_term'].map(lambda x:len(x.split())).astype(np.int64)
all_df['len_of_title'] = all_df['product_title'].map(lambda x:len(x.split())).astype(np.int64)

# Convert to str to avoid AttributeError: 'float' object has no attribute 'split'
all_df['search_term'] = all_df['search_term'].map(lambda x:str(x))
all_df['product_title'] = all_df['product_title'].map(lambda x:str(x))
all_df['product_description'] = all_df['product_description'].map(lambda x:str(x))
all_df['brand'] = all_df['brand'].map(lambda x:str(x))
all_df['attr_description'] = all_df['attr_description'].map(lambda x:str(x))

## Combine misc. to get features 
all_df['product_info'] = all_df['search_term']+"\t"+all_df['product_title']+"\t"+all_df['product_description']+"\t"+all_df['brand']+"\t"+all_df['attr_description']
## Drop extra columns
all_df = all_df.drop(['product_description', 'search_term', 'product_title', 'brand', 'attr_description'],axis=1)
## Get features
all_df['word_in_title'] = all_df['product_info'].map(lambda x:count_common_words(x.split('\t')[0],x.split('\t')[1]))
all_df['word_in_description'] = all_df['product_info'].map(lambda x:count_common_words(x.split('\t')[0],x.split('\t')[2]))
all_df['word_in_attr_description'] = all_df['product_info'].map(lambda x:count_common_words(x.split('\t')[0],x.split('\t')[4]))
all_df['barnd_match'] = all_df['product_info'].map(lambda x:brand_match(x.split('\t')[0],x.split('\t')[3]))
all_df['words_in_des_title_and_des'] = all_df['product_info'].map(lambda x:count_common_words(x.split('\t')[0],x.split('\t')[4]+ " " + x.split('\t')[2]))
all_df['ratio_title'] = all_df['word_in_title']/all_df['len_of_query']
all_df['ratio_description'] = all_df['word_in_description']/all_df['len_of_query']
all_df['ratio_attr_description'] = all_df['word_in_attr_description']/all_df['len_of_query']

all_df = all_df.drop(['product_info'],axis=1)


print("  Fearures extracted: %s minutes" % round(((time.time() - time_start)/60),2))
## Seperate out the train and test feature sets
train_df = all_df.iloc[:train_size]
test_df = all_df.iloc[train_size:]
id_test = test_df['id']
y_train = train_df['relevance'].values
X_train = train_df.drop(['id','relevance'],axis=1).values
X_test = test_df.drop(['id','relevance'],axis=1).values

## Model design
GBR = GradientBoostingRegressor(loss='ls', learning_rate=0.8, n_estimators=120, subsample=1.0, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0, \
                               max_depth=8, init=None, random_state=None, alpha=0.8, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
model = AdaBoostRegressor(base_estimator=GBR, n_estimators=80, learning_rate=1.0, loss='linear', random_state=None)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
## Output
final = pd.DataFrame({"id": id_test, "relevance": y_pred})
final.to_csv('submission.csv',index=False)
print("           Completed: %s minutes" % round(((time.time() - time_start)/60),2))

