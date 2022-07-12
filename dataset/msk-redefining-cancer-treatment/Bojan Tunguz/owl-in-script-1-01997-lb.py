from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/stage2_test_variants.csv')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID')
y = train['Class'].values
train = train.drop('Class', axis=1)

test = pd.merge(test, testx, how='left', on='ID')
pid = test['ID'].values

for c in train.columns:
    if train[c].dtype == 'object' and c !='Text':
        lbl = preprocessing.LabelEncoder(); print(c)
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c+'_lbl_enc'] = lbl.transform(list(train[c].values))
        test[c+'_lbl_enc'] = lbl.transform(list(test[c].values))
    if train[c].dtype == 'object':
        train[c+'_len'] = train[c].map(lambda x: len(str(x)))
        train[c+'_words'] = train[c].map(lambda x: len(str(x).split(' ')))
        
        test[c+'_len'] = test[c].map(lambda x: len(str(x)))
        test[c+'_words'] = test[c].map(lambda x: len(str(x).split(' ')))

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
        ])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)

params = {
    'eta': 0.02,
    'max_depth': 5,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 9,
    'seed': 12,
    'silent': True
}
y = y - 1 #fix for zero bound array
fold = 5
for i in range(fold):
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 500,  watchlist, verbose_eval=5, early_stopping_rounds=30)
    if i != 0:
        pred += model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit) / fold
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit) / fold   
submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)