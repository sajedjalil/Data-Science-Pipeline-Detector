import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

print("reading data...")
df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

attributes = pd.read_csv("../input/attributes.csv")

print("concat attributes...")
attributes.dropna(how="all", inplace=True)

# attributes[attributes.product_uid.isnull()]

attributes["product_uid"] = attributes["product_uid"].astype(int)

attributes["value"] = attributes["value"].astype(str)

def concate_attrs(attrs):
    """
    attrs is all attributes of the same product_uid
    """
    names = attrs["name"]
    values = attrs["value"]
    pairs  = []
    for n, v in zip(names, values):
        pairs.append(' '.join((n, v)))
    return ' '.join(pairs)

product_attrs = attributes.groupby("product_uid").apply(concate_attrs)

product_attrs = product_attrs.reset_index(name="product_attributes")

num_train = df_train.shape[0]


def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())


print("merge data frame...")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all = pd.merge(df_all, product_attrs, how="left", on="product_uid")
df_all['product_attributes'] = df_all['product_attributes'].fillna('')

print("stem columns words...")
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
df_all['product_attributes'] = df_all['product_attributes'].map(lambda x:str_stemmer(x))

print("count words in columns...")
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']+"\t"+df_all['product_attributes']

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_attributes'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info', 'product_attributes'],axis=1)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id'].astype(int)

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

clf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
# clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
print("fit...")
clf.fit(X_train, y_train)
print("predict...")
y_pred = clf.predict(X_test)

print("output result...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('rf_submission_attrs.csv',index=False)
