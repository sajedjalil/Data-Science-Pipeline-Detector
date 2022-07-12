import time
start_time = time.time()
import numpy as np
import pandas as pd
import sys
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.models import FTRL, FM_FTRL

def normalize_text(text):
    return u" ".join([x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")]
                      if len(x) > 1 and x not in stopwords])
if __name__ == '__main__':
    sys.path.insert(0, "../input/wordbatch/wordbatch/")

    # train = pd.read_csv("../input/train.csv")
    # test = pd.read_csv("../input/test.csv")
    train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
    test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")

    category = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ncat = len(category)

    merge = pd.concat([train.drop(["id"] + category, axis=1),
                       test.drop(["id"], axis=1)], axis=0)
    merge.fillna(value="no comment", inplace=True)

    import wordbatch
    from wordbatch.extractors import WordBag, WordHash
    from nltk.corpus import stopwords

    stopwords = {x: 1 for x in stopwords.words("english")}
    non_alphanums = re.compile(u"[^A-Za-z0-9]+")





    wb = wordbatch.WordBatch(normalize_text,
                             extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                  "idf": None}), procs=8)
    wb.dictionary_freeze = True
    word_comment = wb.fit_transform(merge["comment_text"])
    word_comment = word_comment[:, np.array(np.clip(word_comment.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    wb = wordbatch.WordBatch(normalize_text,
                             extractor=(WordBag, {"hash_ngrams": 1, "hash_ngrams_weights": [1.0, 1.0],
                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                  "idf": None}), procs=8)
    wb.dictionary_freeze = True
    char_comment = wb.fit_transform(merge["comment_text"])
    char_comment = char_comment[:, np.array(np.clip(char_comment.getnnz(axis=0) - 10, 0, 1), dtype=bool)]

    from scipy.sparse import hstack

    comment = hstack((word_comment, char_comment)).tocsr()
    nrow_train = train.shape[0]
    X_train = comment[:nrow_train]
    Y_train = train[category]
    print(Y_train.shape)
    X_test = comment[nrow_train:]
    print("D shape",comment.shape[1])

    # lr_model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=comment.shape[1], alpha_fm=0.01, L2_fm=0.0,
    #                 init_fm=0.01,
    #                 D_fm=200, e_noise=0.0001, iters=15, inv_link="sigmoid", threads=4)
    lr_model= FTRL(alpha=0.02, beta=0.1, L1=0.00001, L2=0.1, D=comment.shape[1], inv_link="sigmoid", iters=20, threads=4)
    lr_model.fit(X_train, Y_train)
    # lr_model = LogisticRegression(C=4, solver="sag", max_iter=100, random_state=5)
    # cv_result = np.zeros((6, 3))
    # for i in range(ncat):
    #     cv_result[i, :] = -cross_val_score(lr_model, X_train, Y_train[category[i]], scoring="neg_mean_squared_error")
    # print(cv_result)

    lr_prediction = np.zeros((test.shape[0], ncat))
    for i in range(ncat):
        model = lr_model.fit(X_train, Y_train[category[i]])
        lr_prediction[:, i] = model.predict_proba(X_test)[:, 1]

    # Create submission
    submid = pd.DataFrame({'id': test["id"]})
    submission = pd.concat([submid, pd.DataFrame(lr_prediction, columns=category)], axis=1)
    submission.to_csv('wordbatch_simple_sub.csv', index=False)