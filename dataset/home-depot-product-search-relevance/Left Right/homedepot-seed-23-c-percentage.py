import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from nltk.stem.porter import *
stemmer = PorterStemmer()
import random

random.seed(23)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

num_train = df_train.shape[0]

def str_stem(str1):
	str1 = str1.lower()
	str1 = str1.replace(" in.","in.")
	str1 = str1.replace(" inch","in.")
	str1 = str1.replace("inch","in.")
	str1 = str1.replace(" in ","in. ")
	str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
	return str1
	
def str_common_word(str1, str2):
	str1, str2 = str1.lower(), str2.lower()
	words, cnt = str1.split(), 0
	for word in words:
		if str2.find(word)>=0:
			cnt+=1
	return cnt

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

df_all['title_percentage'] = df_all['word_in_title']/df_all['len_of_query']
df_all['desc_percentage'] = df_all['word_in_description']/df_all['len_of_query']

df_all = df_all.drop(['word_in_title','word_in_description'],axis=1)

df_all = df_all.drop(['search_term','product_title','product_description','product_info','product_uid'],axis=1)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

clf = RandomForestRegressor(n_estimators=24, max_depth=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('py_rf_sub_.csv',index=False)