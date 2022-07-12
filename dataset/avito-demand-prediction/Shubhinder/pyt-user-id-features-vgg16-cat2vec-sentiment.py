#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
import time
notebookstart= time.time()
debug = False

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc, copy
import gzip
from pathlib import PurePath
print("Data:\n",os.listdir("../input"))
from sklearn.decomposition import TruncatedSVD 
from gensim.models import Word2Vec # categorical feature to vectors
from random import shuffle
import warnings
warnings.filterwarnings('ignore')


# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold as kfold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn import linear_model
from scipy.sparse import vstack, hstack, csr_matrix, load_npz
from nltk.corpus import stopwords 
from Levenshtein import  jaro_winkler

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

NFOLDS = 5
SEED = 9966
VALID = True
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

print("\nData Load Stage")
gp = pd.read_csv('../input/aggregated-features-lightgbm/aggregated_features.csv')
agg_cols = list(gp.columns)[1:]

training = pd.read_csv('../input/avito-demand-prediction/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
traindex = training.index
testing = pd.read_csv('../input/avito-demand-prediction/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index

#adding user_id features
training = training.reset_index().merge(gp, on='user_id', how='left').set_index('item_id')
testing = testing.reset_index().merge(gp, on='user_id', how='left').set_index('item_id')

# likelihood encoding
train_target_encoded = pd.read_csv('../input/likelihood-encoding/train_target_encoded.csv')
test_target_encoded = pd.read_csv('../input/likelihood-encoding/test_target_encoded.csv')

training = training.reset_index().merge(train_target_encoded , on='item_id', how='left').set_index('item_id')
testing = testing.reset_index().merge(test_target_encoded, on='item_id', how='left').set_index('item_id')


categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]

#adding sentiment features
spath = '../input/sentiment-lexicons-for-81-languages/sentiment-lexicons/'
sentiment_ru_n = pd.read_csv(spath + 'negative_words_ru.txt').rename(columns={'время':'Words'})
sentiment_ru_n['Polarity'] = -1
sentiment_ru_p = pd.read_csv(spath + 'positive_words_ru.txt').rename(columns={'как':'Words'})
sentiment_ru_p['Polarity'] = 1
sentiment_ru = pd.concat([sentiment_ru_n, sentiment_ru_p], ignore_index=True)
sentiment_ru_vect = CountVectorizer()
sentiment_ru2 = sentiment_ru_vect.fit_transform(sentiment_ru['Words'])
sentiment_ru_model = linear_model.LinearRegression(n_jobs=-1).fit(sentiment_ru2, sentiment_ru['Polarity'])

def getSentiment(s):
    return sentiment_ru_model.predict(sentiment_ru_vect.transform([s]))[0]

senti_cols = ['title', 'description']
for c in senti_cols:
    training[c + '_Pol'] = training[c].map(lambda x: getSentiment(str(x))) #getSentiment
    testing[c + '_Pol'] = testing[c].map(lambda x: getSentiment(str(x))) #getSentiment

del sentiment_ru_n,sentiment_ru_p,sentiment_ru,sentiment_ru_vect,sentiment_ru2,sentiment_ru_model,spath,train_target_encoded,test_target_encoded
gc.collect();

ntrain = training.shape[0]
ntest = testing.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)


del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(df.price.mean(),inplace=True)
df["image_top_1"].fillna(-999,inplace=True)
for col in agg_cols:
    df[col].fillna(-1, inplace=True)

#df["jaro_winkler_similarity"] = df[["title","description"]].apply(lambda x : jaro_winkler(str(x[0]),str(x[1])),axis = 1)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
#df["Weekd of Year"] = df['activation_date'].dt.week
#df["Day of Month"] = df['activation_date'].dt.day

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
print("Encoding :",categorical)
def apply_w2v(sentences, model, num_features):
    def _average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        n_words = 0.
        for word in words:
            if word in vocabulary: 
                n_words = n_words + 1.
                feature_vector = np.add(feature_vector, model[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)
        return feature_vector
    
    vocab = set(model.wv.index2word)
    feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences]
    return np.array(feats)

def gen_cat2vec_sentences(data):
    X_w2v = copy.deepcopy(data)
    names = list(X_w2v.columns.values)
    for c in names:
        X_w2v[c] = X_w2v[c].fillna('unknow').astype('category')
        X_w2v[c].cat.categories = ["%s %s" % (c,g) for g in X_w2v[c].cat.categories]
    X_w2v = X_w2v.values.tolist()
    return X_w2v


print('Cat2Vec...')
n_cat2vec_feature  = len(categorical) # define the cat2vecs dimentions
n_cat2vec_window   = len(categorical) * 2 # define the w2v window size


def fit_cat2vec_model():
    X_w2v = gen_cat2vec_sentences(df.loc[:,categorical].sample(frac=0.6))
    for i in X_w2v:
        shuffle(i)
    model = Word2Vec(X_w2v, size=n_cat2vec_feature, window=n_cat2vec_window)
    return model

print('Fit cat2vec model')
c2v_model = fit_cat2vec_model()


print('apply_w2v for cat2vec')
tr_c2v_matrix = apply_w2v(gen_cat2vec_sentences(df.loc[traindex,categorical]), c2v_model, n_cat2vec_feature)
te_c2v_matrix = apply_w2v(gen_cat2vec_sentences(df.loc[testdex,categorical]), c2v_model, n_cat2vec_feature)

# df.drop(["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"],axis=1,inplace=True)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown',inplace = True)
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")

# Feature Engineering 

# Meta Text Features
textfeats = ["description", "title"]
df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
    df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters

# Extra Feature Engineering
df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']
    
print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}


def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=17000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            max_features=130000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds'] = ridge_preds

del ridge_preds,gp,agg_cols,vectorizer,ridge_oof_train,ridge_oof_test
###VGG16 features

# Create a function to load image features
def load_imfeatures(folder):
    path = PurePath(folder)
    features = load_npz(str(path / 'features.npz'))
    
    if debug:
        features = features[:100000]
        
    return features

ftrain = load_imfeatures('../input/vgg16-train-features/')
ftest = load_imfeatures('../input/vgg16-test-features/')

# Merge two datasets
fboth = vstack([ftrain, ftest])
del ftrain, ftest
gc.collect()
assert fboth.shape[0]==df.shape[0]

# Categorical image feature (max and min VGG16 feature)
df['im_max_feature'] = fboth.argmax(axis=1)  # This will be categorical
df['im_min_feature'] = fboth.argmin(axis=1)  # This will be categorical

df['im_n_features'] = fboth.getnnz(axis=1)
df['im_mean_features'] = fboth.mean(axis=1)
df['im_meansquare_features'] = fboth.power(2).mean(axis=1)

# Reduce image features
tsvd = TruncatedSVD(32)
ftsvd = tsvd.fit_transform(fboth)
del fboth
gc.collect()

# Merge image features into data
df_ftsvd = pd.DataFrame(ftsvd, index=df.index).add_prefix('im_tsvd_')
df = pd.concat([df, df_ftsvd], axis=1)
del ftsvd,df_ftsvd
gc.collect()

cat2vecnames = ["vec_user_id","vec_region","vec_city","vec_parent_category_name","vec_category_name","vec_user_type","vec_image_top_1","vec_param_1","vec_param_2","vec_param_3"]
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),tr_c2v_matrix,ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),te_c2v_matrix,ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + cat2vecnames + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();

print("\nModeling Stage")

del ready_df
gc.collect();

print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 270,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0
}  


if VALID == False:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23)
        
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X, X_train; gc.collect()
    
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print("Model Evaluation Stage")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    del X_valid ; gc.collect()

else:
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X, y,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X; gc.collect()
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1550,
        verbose_eval=100
    )



# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')

print("Model Evaluation Stage")
lgpred = lgb_clf.predict(testing) 

#Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
#blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)
#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))