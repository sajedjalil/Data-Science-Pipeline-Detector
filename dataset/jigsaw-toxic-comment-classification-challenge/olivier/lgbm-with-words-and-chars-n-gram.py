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

def cpuStats(disp=""):
    """ @author: RDizzl3 @address: https://www.kaggle.com/rdizzl3"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print("%s MEMORY USAGE for PID %10d : %.3f" % (disp, pid, memoryUse))


@contextmanager
def timer(name):
    """ Taken from Konstantin """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def count_regexp_occ(regexp="", text=None):
    return len(re.findall(regexp, text))


def perform_nlp(df):
    # Check all sorts of content as it may help find toxic comment
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))

    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))

    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))

    ip_regexp = r"(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
    # df["has_ip_address"] = df["comment_text"].apply(lambda x: count_regexp_occ(ip_regexp, x))

    # Number of fuck - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))

    # Number of suck
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    df["nb_nigger"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))

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
    df["has_mail"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\S+\@\w+\.\w+", x))

    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))

    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))


    # Now clean comments
    # go lower case
    df["clean_comment"] = df["comment_text"].apply(lambda x: x.lower())
    # Drop \n
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\n", " ", x))
    # Drop ip if any
    ip_regexp = r"(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(ip_regexp, " ", x))
    # Drop start up columns
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"^\:+", " ", x))
    # Drop repeated columns anywhere in text
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\:{2,}", " ", x))
    # Drop timestamp
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\d{2}|:\d{2}", " ", x))
    # Drop http links
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"http[s]{0,1}://\S+", " ", x))
    # Drop mail
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\S+\@\w+\.\w+", " ", x))
    # Drop emphasize
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\={2}(.+)\={2}", r"\g<1>", x))
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\"{4}(\S+)\"{4}", r"\g<1>", x))
    # Drop dates
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", "", x))
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\D\d{1,2} \w+ \d{4}", " ", x))

    # Drop repetitions ?

    # Resolve contractions like u, r, y, weren't, ain't, isn't etc
    cont_patterns = [
        (r'US', 'United States'),
        (r'IT', 'Information Technology'),
        (r"[^a-zA-Z]u[^a-zA-Z]", "you"),
        (r"[^a-zA-Z]r[^a-zA-Z]", "are"),
        (r"[^a-zA-Z]y[^a-zA-Z]", "why"),
        (r"[^a-zA-Z]b4[^a-zA-Z]", "before"),
        (r'(W|w)on\'t', 'will not'),
        (r'(C|c)an\'t', 'can not'),
        (r'(I|i)\'m', 'i am'),
        (r'(A|a)in\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would'),
    ]
    patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]
    for (pattern, repl) in patterns:
        df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(pattern, repl, x))

    # Drop punctuation
    exclude = re.compile('[%s]' % re.escape(string.punctuation))
    df["clean_comment"] = df["clean_comment"].apply(
        lambda x: " ".join([exclude.sub(u'', token) for token in x.split()])
    )

    # Get remaining non word characters
    df["remaining_chars"] = df["clean_comment"].apply(lambda x: " ".join(list(set(re.findall(r"\W", x)))))

    # Drop non word characters
    df["clean_comment"] = df["clean_comment"].apply(lambda x: re.sub(r"\W", " ", x))

    # Finally drop extra spaces
    df["clean_comment"] = df["clean_comment"].apply(lambda x: " ".join(x.split()))

    # Get the exact length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))


if __name__ == '__main__':
    gc.enable()
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    with timer("Reading input files"):
        train = pd.read_csv('../input/train.csv').fillna(' ')
        test = pd.read_csv('../input/test.csv').fillna(' ')

    cpuStats()

    with timer("Performing basic NLP"):
        # perform_nlp(train)
        # perform_nlp(test)
        train["ant_slash_n"] = train["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
        test["ant_slash_n"] = test["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
        train['clean_comment'] = train["comment_text"]
        test['clean_comment'] = test["comment_text"]
    cpuStats()
    
    train_text = train['clean_comment']
    test_text = test['clean_comment']
    all_text = pd.concat([train_text, test_text])

    with timer("Creating numerical features"):
        num_features = [f_ for f_ in train.columns
                        if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars", 'has_ip_address'] + class_names]

        skl = MinMaxScaler()
        train_num_features = csr_matrix(skl.fit_transform(train[num_features]))
        test_num_features = csr_matrix(skl.fit_transform(test[num_features]))
    cpuStats()
    
    with timer("Tfidf on word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 1),
            max_features=20000)
        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features = word_vectorizer.transform(test_text)
    del word_vectorizer
    gc.collect()
    cpuStats()
    
    # with timer("Tfidf on char n_gram"):
    #     char_vectorizer = TfidfVectorizer(
    #         sublinear_tf=True,
    #         strip_accents='unicode',
    #         analyzer='char',
    #         stop_words='english',
    #         ngram_range=(2, 6),
    #         max_features=50000)
    #     char_vectorizer.fit(all_text)
    #     train_char_features = char_vectorizer.transform(train_text)
    #     test_char_features = char_vectorizer.transform(test_text)
    # del char_vectorizer
    # gc.collect()
    
    del train_text
    del test_text
    gc.collect()
    cpuStats()
    
    with timer("Staking matrices"):
        csr_trn = hstack(
            [
                # train_char_features, 
                train_word_features, 
                train_num_features
            ]
        ).tocsr()
        del train_word_features, train_num_features, #train_char_features
        gc.collect()
        
        csr_sub = hstack(
            [
                # test_char_features, 
                test_word_features, 
                test_num_features
            ]
        ).tocsr()
        del test_word_features, test_num_features, # test_char_features
        gc.collect()
    submission = pd.DataFrame.from_dict({'id': test['id']})
    del test
    gc.collect()
    
    drop_f = [f_ for f_ in train if f_ not in ["id"] + class_names]
    train.drop(drop_f, axis=1, inplace=True)
    gc.collect()
    cpuStats()
    
    with timer("Scoring LogisticRegression"):
        scores = []
        folds = KFold(n_splits=3, shuffle=True, random_state=1)
        lgb_round_dict = defaultdict(int)
        trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False) 
        cpuStats("LGB Dataset created")
        del csr_trn
        gc.collect()
        cpuStats("Training csr matrix freed")
        for class_name in class_names:
            print("Class %s scores : " % class_name)
            class_pred = np.zeros(len(train))
            train_target = train[class_name]
            trn_lgbset.set_label(train_target.values)
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
            lgb_rounds = 10  # 500
            
            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train, train_target)):
                # d_train = lgb.Dataset(csr_trn[trn_idx], label=train_target.values[trn_idx])
                # d_valid = lgb.Dataset(csr_trn[val_idx], label=train_target.values[val_idx])
                
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
                class_pred[val_idx] = model.predict(model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
                score = roc_auc_score(train_target.values[val_idx], class_pred[val_idx])
                lgb_round_dict[class_name] += model.best_iteration
                print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))
                cpuStats("End of fold %d" % (n_fold + 1))
            print("full score : %.6f" % roc_auc_score(train_target, class_pred))
            scores.append(roc_auc_score(train_target, class_pred))

        print('Total CV score is {}'.format(np.mean(scores)))

    with timer("Predicting probabilities"):
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
        lgb_rounds = 500
        for class_name in class_names:
            with timer("Predicting probabilities for %s" % class_name):
                train_target = train[class_name]
                d_train = lgb.Dataset(csr_trn, label=train_target.values)
                # Train lgb
                model = lgb.train(
                    params=params,
                    train_set=d_train,
                    num_boost_round=int(lgb_round_dict[class_name] / folds.n_splits)
                )
                submission[class_name] = model.predict(csr_sub, num_iteration=model.best_iteration)

submission.to_csv('submission.csv', index=False)