import os
import psutil
import gc
import time
import re
import string
import numpy as np
import pandas as pd
import lightgbm as lgb


from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from collections import defaultdict

# REFERENCE
#    https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram

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


def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]

if __name__ == '__main__':
    gc.enable()
    class_names = ['toxic',]

    with timer("Reading input files"):
        train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv").sample(n=200000, random_state=123)
        
        valid1 = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')
        valid2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')
        
        train = pd.DataFrame({
            #'id': np.concatenate([train.id.values, valid1.id.values, valid2.id.values], axis=0),
            'comment_text': np.concatenate([train.comment_text.values, valid1.comment_text_en.values, valid2.translated.values], axis=0),
            'toxic': np.concatenate([train.toxic.values, valid1.toxic.values, valid2.toxic.values], axis=0),
        })
        test1 = pd.read_csv('/kaggle/input/test-en-df/test_en.csv')
        test1 = pd.DataFrame({
            'id': test1.id.values,
            'comment_text': test1.content_en.values
        })
        
        test2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')
        test2 = pd.DataFrame({
            'id': test2.id.values,
            'comment_text': test2.translated.values
        })
        

    with timer("Performing basic NLP"):
        get_indicators_and_clean_comments(train)
        get_indicators_and_clean_comments(test1)
        get_indicators_and_clean_comments(test2)


    # Scaling numerical features with MinMaxScaler though tree boosters don't need that
    with timer("Creating numerical features"):
        num_features = [f_ for f_ in train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                      'has_ip_address'] + class_names]

        skl = MinMaxScaler()
        train_num_features = csr_matrix(skl.fit_transform(train[num_features]))
        test_num_features1 = csr_matrix(skl.fit_transform(test1[num_features]))
        test_num_features2 = csr_matrix(skl.fit_transform(test2[num_features]))

    # Get TF-IDF features
    train_text = train['clean_comment']
    test_text1 = test1['clean_comment']
    test_text2 = test2['clean_comment']
    all_text = pd.concat([train_text, test_text1, test_text2])

    # First on real words
    with timer("Tfidf on word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            max_features=20000)
        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features1 = word_vectorizer.transform(test_text1)
        test_word_features2 = word_vectorizer.transform(test_text2)

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
        test_char_features1 = char_vectorizer.transform(test_text1)
        test_char_features2 = char_vectorizer.transform(test_text2)

    del char_vectorizer
    gc.collect()

    print((train_char_features > 0).sum(axis=1).max())

    del train_text
    del test_text1
    del test_text2
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

        csr_sub1 = hstack(
            [
                test_char_features1,
                test_word_features1,
                test_num_features1
            ]
        ).tocsr()
        # del test_word_features
        del test_num_features1
        del test_char_features1
        gc.collect()
        
        csr_sub2 = hstack(
            [
                test_char_features2,
                test_word_features2,
                test_num_features2
            ]
        ).tocsr()
        # del test_word_features
        del test_num_features2
        del test_char_features2
        gc.collect()
        
    submission = pd.DataFrame.from_dict({'id': test1['id']})
    del test1
    del test2
    gc.collect()

    # Drop now useless columns in train and test
    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)
    gc.collect()

    # Set LGBM parameters
    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1
    }

    # Now go through folds
    # I use K-Fold for reasons described here : 
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49964
    with timer("Scoring Light GBM"):
        scores = []
        folds = KFold(n_splits=4, shuffle=True, random_state=1)
        lgb_round_dict = defaultdict(int)
        trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)
        del csr_trn
        gc.collect()
        
        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]
            trn_lgbset.set_label(train_target.values)
            
            lgb_rounds = 500

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                watchlist = [
                    trn_lgbset.subset(trn_idx),
                    trn_lgbset.subset(val_idx)
                ]
                # Train lgb l1
                model = lgb.train(
                    params=params,
                    train_set=watchlist[0],
                    num_boost_round=lgb_rounds,
                    valid_sets=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=0
                )
                class_pred[val_idx] = model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
                score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])
                
                # Compute mean rounds over folds for each class
                # So that it can be re-used for test predictions
                lgb_round_dict[class_name] += model.best_iteration
                print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))
            
            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))
            train[class_name + "_oof"] = class_pred

        # Save OOF predictions - may be interesting for stacking...
        #train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv("lvl0_lgbm_clean_oof.csv",
        #                                                                       index=False,
        #                                                                      float_format="%.8f")

        print('Total CV score is {}'.format(np.mean(scores)))

    with timer("Predicting probabilities"):
        # Go through all classes and reuse computed number of rounds for each class
        for class_name in class_names:
            with timer("Predicting probabilities for %s" % class_name):
                train_target = train[class_name]
                trn_lgbset.set_label(train_target.values)
                # Train lgb
                model = lgb.train(
                    params=params,
                    train_set=trn_lgbset,
                    num_boost_round=int(lgb_round_dict[class_name] / folds.n_splits)
                )
                submission[class_name] = (model.predict(csr_sub1, num_iteration=model.best_iteration) + model.predict(csr_sub2, num_iteration=model.best_iteration)) / 2

    submission.to_csv("submission.csv", index=False, float_format="%.8f")
    

