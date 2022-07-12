import numpy as np
import pandas as pd
import math

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')



# Data Processing
test['price'] = -1
df = pd.concat([train,test])

df['category_name'] = pd.factorize(df['category_name'])[0]
df['brand_name'] = pd.factorize(df['brand_name'])[0]

col = [c for c in df.columns if c not in ['train_id', 'test_id', 'price', 'name', 'item_description']]

test_df = df[df['price'] == -1]
df = df[df['price'] != -1]

df['price'] = np.log1p(df['price'])



# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))



# Categorical feats
et_model = ExtraTreesRegressor(n_jobs=-1, n_estimators=100,  random_state=42, min_samples_leaf=2)

et_model.fit(df[col], df['price'])

y_tr_1 = et_model.predict(df[col])

print('RMSLE: {0:.5f}'.format(rmsle(np.exp(df['price'])-1, np.exp(y_tr_1)-1)))


y_pred_1 = et_model.predict(test_df[col])



# Name feat
vectorizer = CountVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=50)

df_con = pd.concat([df, test_df], axis=0)

df_con = vectorizer.fit_transform(df_con['name'])
df_con = svd.fit_transform(df_con)


et_model = ExtraTreesRegressor(n_jobs=-1, n_estimators=50,  random_state=42, min_samples_leaf=3)

et_model.fit(df_con[:1482535], df['price'])

y_tr_2 = et_model.predict(df_con[:1482535])

print('RMSLE: {0:.5f}'.format(rmsle(np.exp(df['price'])-1, np.exp(y_tr_2)-1)))


y_pred_2 = et_model.predict(df_con[1482535:])


print('RMSLE: {0:.5f}'.format(rmsle(np.exp(df['price'])-1, np.exp(y_tr_1*0.5+y_tr_2*0.5)-1)))


test_df['price'] = np.exp(y_pred_1*0.5+y_pred_2*0.5)-1
test_df['test_id'] = test_df['test_id'].astype(np.int)
test_df[['test_id', 'price']].to_csv("et_submission.csv", index = False)