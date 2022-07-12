from contextlib import contextmanager
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
import keras as K
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import gc
import re
import string

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')
        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)
        self._check_if_all_columns_present(x)
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]

def fit_predict(xs) -> np.ndarray:
    X_train, y_train, train_ids, dev_ids = xs
    config = tf.ConfigProto()
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        K.backend.set_session(sess)
        model_in = K.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = K.layers.Dense(512, activation='relu')(model_in)
        out = K.layers.Concatenate()([K.layers.Dense(64, activation='tanh')(out),
                                      K.layers.Dense(64, activation='relu')(out)])
        out2= K.layers.Concatenate()([K.layers.Dense(32, activation='tanh')(out),
                                            K.layers.Dense(32, activation='relu')(out)])
        out = K.layers.Concatenate()([K.layers.Dropout(0.2)(out), out2])

        out = K.layers.Add()([K.layers.Dense(1, activation='linear')(out),
                              K.layers.Dense(1, activation='relu')(out),
                              ])
        out = K.layers.Add()([K.layers.Dense(1, activation='linear')(out),
                              K.layers.Dense(1, activation='relu')(out),
                              ])
        out = K.layers.Dense(1)(out)
        model = K.Model(model_in, out)
        model.compile(loss='logcosh', optimizer=K.optimizers.Adam(lr=3e-3), metrics=['accuracy'])
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train[train_ids,:], y=y_train[train_ids], batch_size=2**(11 + i), epochs=1, verbose=0)
        preds= np.zeros(X_train.shape[0])
        preds[dev_ids]= model.predict(X_train[dev_ids,:])[:, 0]
        return preds

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't":
                    "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will",
                "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've":
                    "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've":
                    "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",
                "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've":
                    "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
                "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've":
                    "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
                "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd":
                    "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're":
                    "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've":
                    "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're":
                    "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've":
                    "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll":
                    "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's":
                    "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've":
                    "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're":
                    "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour':
                    'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling':
                    'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation':
                    'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura':
                    'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo':
                    'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation':
                    'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18':
                    '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst":
                    'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization':
                    'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()

def main():
    vectorizer = make_union(
        on_field('question_text', TfidfVectorizer(max_features=13000, token_pattern='\w+', strip_accents='unicode',
                                                  tokenizer=tokenize, sublinear_tf=True)),
        on_field('question_text', TfidfVectorizer(ngram_range=(3, 3), analyzer='char', min_df=25)),
        make_pipeline(PandasSelector(columns=['num_words', 'num_singletons', 'caps_vs_length',
                                              ], return_vector=False), MaxAbsScaler()),
        )

    with timer('process train'):
        df_train = pd.read_csv("../input/train.csv")
        df_test = pd.read_csv("../input/test.csv")
        train_count= len(df_train)
        test_count= len(df_test)
        test_qids= df_test['qid']
        df_train= pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
        print(df_train.shape)

        df_train["question_text"] = df_train["question_text"].str.lower()\
            .map(clean_text).map(clean_numbers).map(replace_typical_misspell)

        wft = {}
        for line in df_train['question_text']:
            for word in line.lower().split():
                wft[word]= wft.get(word, 0)+1
        df_train["num_words"] = [len(str(x).split()) for x in df_train["question_text"]]
        df_train["num_singletons"] = [len([w for w in str(x).lower().split() if wft.get(word, 0)<=1]) for x in
                                                                 df_train['question_text']]
        df_train['capitals'] = [sum(1 for c in x if c.isupper()) for x in df_train['question_text']]
        df_train['caps_vs_length'] = df_train['capitals']/df_train['question_text'].str.len()
        X_train = vectorizer.fit_transform(df_train, df_train['target']).astype(np.float32)

        y_train = df_train['target'][:train_count].values
        del (df_train)
        gc.collect()


    n_reps = 8
    n_splits = 8

    y_pred = np.zeros(train_count + test_count)
    with timer('run kfold'):
        datasets= []
        for train_ids, dev_ids in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)\
                .split(range(train_count), y_train):
            dev_ids= list(dev_ids)+list(range(train_count, train_count+test_count))
            for rep in range(n_reps):
                datasets.append([X_train, y_train, train_ids, dev_ids])
        ###with ThreadPool(processes=2) as pool:
        with ThreadPool(processes=1) as pool:
            y_pred+= np.sum(pool.map(fit_predict, datasets), axis=0) / n_reps
    y_pred[train_count:]/= n_splits
    scores= []
    thresholds= np.arange(0.1, 0.501, 0.01)
    for thresh in thresholds:
        thresh = np.round(thresh, 2)
        scores.append(f1_score(y_train, (y_pred[:train_count] > thresh).astype(int)))
        print("F1 score at threshold {0} is {1}".format(thresh, scores[-1]))
    max_idx= np.argmax(scores)
    print(pd.Series(y_pred[:train_count]> thresholds[max_idx]).astype(int).value_counts())
    print(pd.Series(y_pred[train_count:]> thresholds[max_idx]).astype(int).value_counts())
    print(pd.Series(y_pred[:train_count]).describe())
    print(pd.Series(y_pred[train_count:]).describe())
    out_df = pd.DataFrame({"qid": test_qids,
                           "prediction":(y_pred[train_count:] > thresholds[max_idx]).astype(int)})
    print(thresholds[max_idx], scores[max_idx])
    out_df.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()