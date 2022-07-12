# Thanks olivier for his text clean  
# https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram?scriptVersionId=2694282

# I choose MultinomialNB as an example, preformance not so good, but runs fast.

FAST_RUN = False

if FAST_RUN == True:
    mnb_folds = 2
else:
    mnb_folds = 10
    
import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
import re
import string
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
import gc
from collections import defaultdict
import os
import psutil
from functools import reduce


# Contraction replacement patterns
cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
    (b'&lt;3', b' heart '),
    (b':d', b' smile '),
    (b':dd', b' smile '),
    (b':p', b' smile '),
    (b'8\)', b' smile '),
    (b':-\)', b' smile '),
    (b':\)', b' smile '),
    (b';\)', b' smile '),
    (b'\(-:', b' smile '),
    (b'\(:', b' smile '),
    (b'yay!', b' good '),
    (b'yay', b' good '),
    (b'yaay', b' good '),
    (b':/', b' worry '),
    (b':&gt;', b' angry '),
    (b":'\)", b' sad '),
    (b':-\(', b' sad '),
    (b':\(', b' sad '),
    (b':s', b' sad '),
    (b':-s', b' sad '),
    (b'\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}', b' '),
    (b'(\[[\s\S]*\])', b' '),
    (b'[\s]*?(www.[\S]*)', b' ')
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    
    # replace words like hhhhhhhhhhhhhhi with hi
    for ch in string.ascii_lowercase:
        pattern = bytes(ch+'{3,}', encoding="utf-8")
        clean = re.sub(pattern, bytes(ch, encoding="utf-8"), clean)
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')


def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    if len(text) == 0:
        return 0
    else:
        return len(re.findall(regexp, text)) / len(text)


def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))



def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


class qiaofeng_kfold_stack:
    def __init__(self, train, train_target, test, split_target, model, preprocess_func=None, score_func=None, kfolds=5, random_seed=9527, logger=None):
        self.train = train
        self.train_target = train_target
        self.test = test
        self.split_target = split_target
        self.model = model
        self.preprocess_func = preprocess_func
        self.score_func = score_func
        self.kfolds = kfolds
        self.random_seed = random_seed
        self.logger= logger
        self.skf = StratifiedKFold(n_splits=self.kfolds, random_state= self.random_seed)
        self.predict_test_kfolds = []
        self.predict_valid_kfolds = np.zeros((self.train.shape[0]))
    def print_params(self):
        print('kfolds : %s' % self.kfolds)
        print('random seed : %s' % self.random_seed)
    def preprocess(self):
        if self.preprocess_func != None:
            self.train, self.test = self.preprocess_func(self.train, self.test)
    def score(self, target, predict):
        return self.score_func(target, predict)
    def model_fit(self, train, train_target):
        self.model.fit(train, train_target)
    def model_predict(self, dataset):
        return self.model.predict(dataset)
    def model_fit_predict(self, train, train_target, dataset):
        self.model_fit(train, train_target)
        predict_train = self.model_predict(train)
        predict_valid = self.model_predict(dataset)
        predict_test = self.model_predict(self.test)
        return predict_train, predict_valid, predict_test
    def clear_predicts(self):
        self.predict_test_kfolds = []
        self.predict_valid_kfolds = np.zeros((self.train.shape[0]))
    def model_train_with_kfold(self):
        self.clear_predicts()
        for (folder_index, (train_index, valid_index)) in enumerate(self.skf.split(self.train, self.split_target)):
            x_train, x_valid = self.train[train_index], self.train[valid_index]
            y_train, y_valid = self.train_target[train_index], self.train_target[valid_index]
            predict_train, predict_valid, predict_test = self.model_fit_predict(x_train, y_train, x_valid)
            self.predict_test_kfolds.append(predict_test)
            self.predict_valid_kfolds[valid_index] = predict_valid
            if self.logger != None:
                train_score = self.score(y_train, predict_train)
                valid_score = self.score(y_valid, predict_valid)
                self.logger('Fold: %s, train score: %s valid score: %s' % (folder_index, train_score, valid_score))
    def predict_test_mean(self):
        return reduce(lambda x,y:x+y, self.predict_test_kfolds)  / self.kfolds

if __name__ == '__main__':
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    with timer("Reading input files"):
        train = pd.read_csv('../input/train.csv').fillna(' ')
        test = pd.read_csv('../input/test.csv').fillna(' ')
        if FAST_RUN == True:
            train = train[:5000]
            test = test[:100]
    gc.disable()
    with timer("Performing basic NLP"):
        get_indicators_and_clean_comments(train)
        get_indicators_and_clean_comments(test)
    gc.enable()
    gc.collect()
    gc.disable()

    # Scaling numerical features with MinMaxScaler though tree boosters don't need that
    with timer("Creating numerical features"):
        num_features = [f_ for f_ in train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address'] + class_names]

        skl = MinMaxScaler()
        train_num_features = csr_matrix(skl.fit_transform(train[num_features]))
        test_num_features = csr_matrix(skl.fit_transform(test[num_features]))

    # Get TF-IDF features
    train_text = train['clean_comment']
    test_text = test['clean_comment']
    all_text = pd.concat([train_text, test_text])

    # First on real words
    with timer("Tfidf on word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)
        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features = word_vectorizer.transform(test_text)

    del word_vectorizer
    gc.collect()

    # Now use the char_analyzer to get another TFIDF
    # Char level TFIDF would go through words when char analyzer only considers
    # characters inside a word
    with timer("Tfidf on char n_gram"):
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 1),
            max_features=50000)
        char_vectorizer.fit(all_text)
        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)

    del char_vectorizer
    gc.collect()

    print((train_char_features > 0).sum(axis=1).max())

    del train_text
    del test_text
    gc.collect()

    # Now stack TF IDF matrices
    with timer("Staking matrices"):
        csr_trn = hstack(
            [
                train_char_features,
                train_word_features,
                train_num_features
            ]
        ).tocsr()
        # del train_word_features
        del train_num_features
        del train_char_features
        gc.collect()

        csr_sub = hstack(
            [
                test_char_features,
                test_word_features,
                test_num_features
            ]
        ).tocsr()
        # del test_word_features
        del test_num_features
        del test_char_features
        gc.collect()
    submission = pd.DataFrame.from_dict({'id': test['id']})
    del test
    gc.collect()

    # Drop now useless columns in train and test
    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)
    gc.collect()

    class qiaofeng_mnb(qiaofeng_kfold_stack):
        def model_predict(self, dataset):
            return self.model.predict_proba(dataset)[:,1]
    
    split_target = np.zeros(train.shape[0])
    for index, lab in enumerate(class_names):
        split_target += train[lab] * (2 ** index) 
        
    pred_valid = np.zeros((csr_trn.shape[0], len(class_names)))
    pred_test_avg = np.zeros((csr_sub.shape[0], len(class_names)))
    for (lab_ind,lab) in enumerate(class_names):
        def label_logger(s):
            print('MultinomialNB Fitting %s, %s' % (lab, s))
        model = MultinomialNB(alpha=0.05)
        mnb_kfold_model = qiaofeng_mnb(train=csr_trn, train_target=train[lab].values, test=csr_sub, kfolds=mnb_folds,split_target=split_target,
                                          score_func=roc_auc_score, logger=label_logger, model=model)
        mnb_kfold_model.model_train_with_kfold()
        pred_valid[:,lab_ind] = mnb_kfold_model.predict_valid_kfolds
        pred_test_avg[:,lab_ind] = mnb_kfold_model.predict_test_mean()
        if FAST_RUN == True:
            break

    valid_id = pd.DataFrame({'id': train["id"]})
    valid_predict = pd.concat([valid_id, pd.DataFrame(pred_valid, columns = class_names)], axis=1)
    valid_predict.to_csv('mnb_valid_predict.csv', index= False)

    if FAST_RUN == False:
        sub = pd.read_csv('../input/sample_submission.csv')
        sub[class_names] = pred_test_avg.copy()
        sub.to_csv('mnb_test_predict.csv', index=False)