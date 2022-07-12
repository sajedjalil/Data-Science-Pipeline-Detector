# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# https://i.imgflip.com/b9nkz.jpg

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from catboost import Pool, CatBoostRegressor
from sklearn import preprocessing

print('reading the data...')
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

def split_cat(s):
    try:
        return s.split('/')[0],s.split('/')[1],s.split('/')[2],
    except:
        return ['No','No','No'] # and no 


df_train[['cat1','cat2','cat3']] = pd.DataFrame(df_train.category_name.apply(split_cat).tolist(),
                                   columns = ['cat1','cat2','cat3'])
df_test[['cat1','cat2','cat3']] = pd.DataFrame(df_test.category_name.apply(split_cat).tolist(),
                                   columns = ['cat1','cat2','cat3'])


print('making the magic...')
corpus = df_train.name.values.astype('U').tolist() + df_test.name.values.astype('U').tolist() + df_train.item_description.values.astype('U').tolist() + df_test.item_description.values.astype('U').tolist()


vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(corpus)

train_cor1 = vectorizer.transform(df_train.name.values.astype('U').tolist())
train_cor2 = vectorizer.transform(df_train.item_description.values.astype('U').tolist())

test_cor1 = vectorizer.transform(df_test.name.values.astype('U').tolist())
test_cor2 = vectorizer.transform(df_test.item_description.values.astype('U').tolist())




transformer = TfidfTransformer()
tr_1 = transformer.fit_transform(train_cor1)
tr_2 = transformer.fit_transform(train_cor2)
te_1 = transformer.fit_transform(test_cor1)
te_2 = transformer.fit_transform(test_cor2)



df_train['cor1_Tfidf'] = np.mean(tr_1,1)
df_train['cor2_Tfidf'] = np.mean(tr_2,1)

df_test['cor1_Tfidf'] = np.mean(te_1,1)
df_test['cor2_Tfidf'] = np.mean(te_2,1)


shape = df_train.shape[0]
all_names = pd.concat([df_train.name, df_test.name])
all_names = all_names.astype('category').cat.codes

df_train['names_cat'] = all_names[:shape]
df_test['names_cat'] = all_names[shape:]

del shape, all_names, tr_1, tr_2, te_1, te_2, vectorizer, train_cor1, train_cor2, test_cor1, test_cor2, transformer, corpus, split_cat
gc.collect()



df_train['len1'] = df_train.name.str.len()
df_train['len2'] = df_train.item_description.str.len()

df_test['len1'] = df_test.name.str.len()
df_test['len2'] = df_test.item_description.str.len()

s = pd.Series(df_train.name, dtype="category").cat.codes

print('preparing data for the Cat')

drop_features = ['name','item_description']
x_train = df_train.drop(drop_features + ['train_id','price'],1)
x_test = df_test.drop(drop_features+['test_id',],1)
y_train = np.abs(df_train.price.values)


print(x_train.head())
print(x_test.head())

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

train_data = Pool(x_train, y_train, cat_features=[0,1,2,4,5,6,9])
test_data = Pool(x_test, cat_features=[0,1,2,4,5,6,9])
idx = df_test.test_id.values
del x_train, x_test, y_train, df_train, df_test
gc.collect()
print('train')

params = {'depth': 11, 'iterations': 588, 'l2_leaf_reg': 9, 
        'learning_rate': 0.98, 'random_seed': 1111,
        'loss_function': 'MAE'}
model = CatBoostRegressor(**params)
gc.collect()

model.fit(train_data)
gc.collect()

print('predict')
y_pred = model.predict(test_data)

print('saving predictions...')
sub = pd.DataFrame()
sub['test_id'] = idx
sub['price'] = np.abs(y_pred)
sub.to_csv('cat_predicts.csv', index=False, float_format='%.3f')