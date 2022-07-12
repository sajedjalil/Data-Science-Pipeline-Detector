#import modules
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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
from collections import defaultdict
import os
import psutil



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
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]



@contextmanager
def timer(name):
    """
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
    return len(re.findall(regexp, text))



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
        

    
with timer("Reading input files"):
    train = pd.read_csv('../input/train.csv',usecols=['comment_text','target','id']).fillna(' ')
    test = pd.read_csv('../input/test.csv').fillna(' ')
 
    
with timer("Performing basic NLP"):
    get_indicators_and_clean_comments(train)
    get_indicators_and_clean_comments(test)
 
    
# Scaling numerical features with MinMaxScaler though tree boosters don't need that
with timer("Creating numerical features"):
    num_features = [f_ for f_ in train.columns
                    if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address', 'target']]
    
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
        ngram_range=(1, 2),
        max_features=30000)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    
del word_vectorizer
gc.collect()

del train_text
del test_text
gc.collect()

# Now stack TF IDF matrices
with timer("Staking matrices"):
    csr_trn = hstack(
        [
            train_word_features,
            train_num_features
        ]
    ).tocsr()
    gc.collect()

    csr_sub = hstack(
        [
            test_word_features,
            test_num_features
        ]
    ).tocsr()
    gc.collect()
submission = pd.DataFrame.from_dict({'id': test['id']})
del test
gc.collect()

# Set LGBM parameters
params = {
    "objective": "binary",
    'metric': {'auc'},
    "boosting_type": "gbdt",
    "verbosity": -1,
    "num_threads": 4,
    # "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "learning_rate": 0.1,
    "verbose": -1,
    "min_split_gain": .1,
    "reg_alpha": .1,
    "max_bin" : 512,
    "num_leaves":64,
}

trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)  
train_target = np.where(train["target"] > 0.5,1,0)

with timer("Predicting probabilities"):
    trn_lgbset.set_label(train_target)
    # Train lgb
    model = lgb.train(
        params=params,
        train_set=trn_lgbset,
        num_boost_round=700,
    )
    submission["prediction"] = model.predict(csr_sub, num_iteration=model.best_iteration)
    
submission.to_csv("submission.csv", index=False)

# TODO : improved
#DONE : Basic Working
# Refrence :https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram
# If you ave any idea for improvement comment below :) Happy kaggleing