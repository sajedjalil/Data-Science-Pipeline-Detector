import numpy as np
import pandas as pd
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix, vstack
import lightgbm as lgb
import gc
import re
from multiprocessing import Pool
import multiprocessing as mp
import psutil
import numpy as np
import pandas as pd
import random
from textblob import TextBlob
import itertools

random.seed(42)

CHARS_TO_REMOVE = '!¡"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
MAX_LEN = 400000
#STOP_WORDS = list(stopwords.words('english'))
num_partitions = psutil.cpu_count() * 8
num_cores = psutil.cpu_count()
aug = False
identity_columns = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness', 'target']

vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                              min_df=3, max_df=0.3,
                                              strip_accents='unicode',
                                              use_idf=1,
                                              smooth_idf=1,
                                              sublinear_tf=1,
                                              max_features=MAX_LEN,
                                              lowercase=False)

def shuffler(df, n=50000):
    
    df = df[df.target >= 0.5]
    final_df = [' '.join([random.choice(df.iloc[random.randint(0, df.shape[0]-1),:].comment_text.split())
                    for i in range(random.randint(10, 30))]) for j in range(n)]
    target = [1 for _ in range(n)]
    return pd.DataFrame({'comment_text': final_df, 'target': target})

def read_data(path, tr=True, n_rows=None):

    if tr:
        if aug:
            df_ini = pd.read_csv(path, usecols=['comment_text']+identity_columns)#.fillna(' ')
            df_aug = shuffler(df_ini)
            df = pd.concat([df_ini, df_aug])
        else:
            df = pd.read_csv(path, usecols=['comment_text']+identity_columns)#.fillna(' ')
        return df
    else:
        return pd.read_csv(path, usecols=['comment_text', 'id']).fillna(' ')
    
def df_parallelize_run(df, func, vector=False):

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    if vector:
        df = vstack(pool.map(func, df_split), format='csr')
    else:
        df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

    
def remove_stop_words(text, sw):
    text = ' '.join([word for word in text.split() if word not in sw])
    return text

def remove_noise_chars(text, chars):
    text = ''.join([word for word in text if word not in chars])
    return text

def c_t2(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub(r'<.*?>', '', text)
    text = text.strip(' ')
    return text

def clean_text(df):
            
    df["ast"] = df["comment_text"].apply(lambda x: x.count('*'))
    df["ex"] = df["comment_text"].apply(lambda x: x.count('!'))
    df["qu"] = df["comment_text"].apply(lambda x: x.count('?'))
    df["ar"] = df["comment_text"].apply(lambda x: x.count('@'))
    df["ha"] = df["comment_text"].apply(lambda x: x.count('#'))
    df["sum_simb"] = df.ast + df.ex + df.qu + df.ar + df.ha
    df["len_pr"] = df["comment_text"].apply(lambda x: len(x))
    df["num_words"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["num_upper"] = df["comment_text"].apply(lambda x: len([i for i in x.split() if i.isupper()]))
    df["num_lower"] = df["comment_text"].apply(lambda x: len([i for i in x.split() if i.islower()]))
    df["len_max_word"] = df["comment_text"].apply(lambda x: max([len(i) for i in x.split()]))
    df["len_min_word"] = df["comment_text"].apply(lambda x: min([len(i) for i in x.split()]))
    df['num_unique_words'] = df['comment_text'].apply(lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['words_vs_upper'] = df['num_upper'] / df['num_words']
    df['words_vs_lower'] = df['num_lower'] / df['num_words']
    df['num_smilies'] = df['comment_text'].apply(lambda x: sum(x.count(w) for w in [':-)', ':)', ';-)', ';)']))
    df['title_word_count'] = df['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
#    df['stopword_count'] = df['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in STOP_WORDS]))
    df["comment_text"] = df["comment_text"].fillna(' ')
    df["comment_text"] = df["comment_text"].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    df["comment_text"] = df["comment_text"].apply(lambda x: c_t2(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: remove_noise_chars(x, CHARS_TO_REMOVE))
    df["comment_text"] = df["comment_text"].apply(lambda x: ''.join(''.join(s)[:1] for _, s in itertools.groupby(x)))
#    df['sw_ratio'] = df.stopword_count / df.num_words
    df['tw_ratio'] = df.title_word_count / df.num_words
    df['word_density'] = df['len_pr'] / (df['num_words']+1)
    df['ratio_max_len'] = df.len_max_word / df.len_pr

    return df

def clean_text_inverse(df):
    
    df["comment_text"] = df["comment_text"].fillna(' ')
    df["comment_text"] = df["comment_text"].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    df["comment_text"] = df["comment_text"].apply(lambda x: c_t2(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: remove_noise_chars(x, CHARS_TO_REMOVE))
    df["comment_text"] = df["comment_text"].apply(lambda x: ''.join(''.join(s)[:1] for _, s in itertools.groupby(x)))
    df["comment_text"] = df["comment_text"].apply(lambda x: ' '.join(x.split()[::-1]))

    return df


def vector(text_data, train=True, to_dataframe=False):

    if train:
        vectorizer.fit(text_data)
        
    X_words = vectorizer.transform(text_data)
    
    if to_dataframe:
        X_words = pd.DataFrame(X_words)            

    return X_words


def get_weights(path):
    
    train = pd.read_csv(path, usecols=identity_columns).fillna(0)
    weights = np.zeros((len(train),))
    weights += (train[identity_columns].values>=0.5).sum(axis=1).astype(bool).astype(np.int)
    
    loss_weight = 1.0 / weights.mean()
    
    del train
    gc.collect()
    
    return weights, loss_weight

    
def custom_loss(y_pred, y_true):
    precision, recall, thresholds = precision_recall_curve(np.where(y_true >= 0.5, 1, 0), y_pred)
    AUC = auc(recall, precision)
    if AUC != AUC:
        AUC = 0
    return 'PR_AUC', AUC, True

def _fit(X, y, verbose, esr):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    
    
    lgb_model.fit(X_train,
                       y_train,
                       eval_set=[(X_test, y_test)],
                       verbose=verbose,
                       early_stopping_rounds=esr,
                       eval_metric=custom_loss
                      )
    
def _re_fit(X, y, verbose, n_e):
  
    lgb_model.n_estimators = n_e
    lgb_model.fit(X, y,verbose=verbose)

            


verbose = 50
EARLY_STOPPING_ROUNDS = 100
FT_SEL = False
k = 3
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
params = {
    'max_depth': 21,
#    'metric': 'auc',
    'n_estimators': 20000,
    'learning_rate': 0.1,
    'colsample_bytree': 0.4,
    'objective': 'xentropy',
    'n_jobs': -1,
    'seed': 42,
    'bagging_fraction': 0.3,
    'lambda_l1': 0,
    'lambda_l2': 0,
}
lgb_model = lgb.LGBMClassifier(**params)
PATH = '../input/'
BALANCE_TRAIN = True


print('\n### TFIDF ###')          
print('\n| Data processing...\n')
print('\t- Reading data...')
train = read_data(PATH+'train.csv')\
#      .sample(50000, random_state=42).reset_index(drop=True)
# train_df_1 = train[train.target >= 0.5]
# train_df_0 = train[train.target < 0.5].sample(train_df_1.shape[0]*4, random_state=42)

# train = pd.concat([train_df_1, train_df_0])
# train = train.sample(frac=1, random_state=42).reset_index(drop=True)
test = read_data(PATH+'test.csv', tr=False)

train_rows = train.shape[0]

print('\t- Cleaning data...')
train_cleaned = df_parallelize_run(train, clean_text).reset_index(drop=True)
test_cleaned = df_parallelize_run(test, clean_text).reset_index(drop=True)

print('\t- Vectorizer...')
vector(pd.concat([train_cleaned['comment_text'], test_cleaned['comment_text']]))
X_words = vector(train_cleaned['comment_text'].values, train=False)
X_words_test = vector(test_cleaned['comment_text'].values, train=False)

print('\t- Generating final datasets...')
X_cols = ['ast', 'ex', 'qu', 'ar', 'ha', 'len_pr', 'tw_ratio',
       'num_words', 'len_max_word', 'len_min_word',
       'ratio_max_len', 'words_vs_unique', 'word_density', 'sum_simb',
       'words_vs_upper', 'words_vs_lower', 'num_smilies']


extra_data = csr_matrix(train_cleaned[X_cols])
#    del train
gc.collect()
X_train = hstack([X_words, extra_data]).tocsr()

del X_words
del extra_data
gc.collect()

extra_data = csr_matrix(test_cleaned[X_cols])
X_test = hstack([X_words_test, extra_data]).tocsr()
del X_words_test
del extra_data
gc.collect()

preds_dict1 = dict()
preds_dict2 = dict()
test_preds_dict = dict()

for target in identity_columns:
    
    print(f'\n\n| Modeling with target = {target}...\n')
    print('\t- Fitting Fold 1...')
    
    y_train = np.where(train[target].fillna(0) >= 0.5, 1, 0)

    _fit(X_train[train_rows//2:], y_train[train_rows//2:], verbose, EARLY_STOPPING_ROUNDS)
    
    preds_tdidf = lgb_model.predict_proba(X_test)[:,1]
    preds_dict1[target] = lgb_model.predict_proba(X_train[:train_rows//2])[:,1]
    
    print('\t- Fitting Fold 2...')
    
    _fit(X_train[:train_rows//2], y_train[:train_rows//2], verbose, EARLY_STOPPING_ROUNDS)
    
    preds_tdidf2 = lgb_model.predict_proba(X_test)[:,1]
    preds_dict2[target] = lgb_model.predict_proba(X_train[train_rows//2:])[:,1]
    
    preds_tdidf = (preds_tdidf + preds_tdidf2)/2
    
    test_preds_dict[target] = preds_tdidf

    
x_test_preds = csr_matrix(pd.DataFrame(test_preds_dict))

preds_df = csr_matrix(pd.concat([pd.DataFrame(preds_dict1), pd.DataFrame(preds_dict2)]))
gc.collect()
X_train2 = hstack([X_train, preds_df]).tocsr()
y_train = np.where(train['target'].fillna(0) >= 0.5, 1, 0)
X_test2 = hstack([X_test, x_test_preds]).tocsr()

print('\n\n| Modeling with target = target...\n')
print('\t- Fitting...')
_fit(X_train2, y_train, verbose, EARLY_STOPPING_ROUNDS)


print('\t- Re-Fitting with all data...')
_re_fit(X_train2, y_train, verbose, int(lgb_model.best_iteration_*1.15))


print('\n| Predictions...\n')    
print('\t- Making predictions...')
preds_tdidf = lgb_model.predict_proba(X_test2)[:,1]

#preds_dict[target] = preds_tdidf
#    
#preds_df = pd.DataFrame(preds_dict)

print('\n| Saving submission...')
submission = pd.DataFrame({'id': test['id'], 'prediction': preds_tdidf})
submission.to_csv("submission.csv", index=False)

