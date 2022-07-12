import logging
import multiprocessing
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from nltk import word_tokenize


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
CONFIG FOR OFFLINE / ONLINE

- Online Changes Filepaths
- Set toy to true to use only 10% of data
"""

ONLINE = True
TOY = False

if ONLINE:
    X_train_file = "../input/donors-choose-preprocess/X_train.npy"
    X_test_file = "../input/donors-choose-preprocess/X_test.npy"
    y_train_file = "../input/donors-choose-preprocess/y_train.npy"
    y_test_file = "../input/donors-choose-preprocess/y_test.npy"
    mapper_file = "../input/donors-choose-preprocess/mapper.p"
    RESOURCE_FILE = "../input/donorschoose-application-screening/resources.csv"
    TEST_FILE = "../input/donorschoose-application-screening/test.csv"
    WORKERS = 32

else:
    X_train_file = "X_train.npy"
    X_test_file = "X_test.npy"
    y_train_file = "y_train.npy"
    y_test_file = "y_test.npy"
    RESOURCE_FILE = "resources.csv"
    TEST_FILE = "test.csv"
    WORKERS = 4


# From [Github Gist](https://gist.github.com/tejaslodaya/562a8f71dc62264a04572770375f4bba)

def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))])
    pool.close()
    result = sorted(result, key=lambda x: x[0])
    return pd.concat([i[1] for i in result])


def tokenize(x):
    return word_tokenize(x)


def count_punctuation(tokens, punctuation_char):
    return len([token for token in tokens if token == punctuation_char])


def preprocess_df(df, workers):
    if __name__ == "__main__":
        dfr = pd.read_csv(RESOURCE_FILE)
        dfr['total'] = dfr['price'] * dfr['quantity']
        dfr['has_zero'] = dfr['price'].apply(lambda x: 1 if x == 0 else 0)
        dfr = dfr.groupby('id').agg('sum').reset_index()

        # merging essays
        # borrowed
        df['student_description'] = df['project_essay_1']
        df.loc[df.project_essay_3.notnull(), 'student_description'] = df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_1'] + \
                                                                      df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_2']
        df['project_description'] = df['project_essay_2']

        df.loc[df.project_essay_3.notnull(), 'project_description'] = df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_3'] + \
                                                                      df.loc[
                                                                          df.project_essay_3.notnull(), 'project_essay_4']

        df['project_subject_categories'] = df['project_subject_categories'].apply(lambda x: x.split(", "))
        df['project_subject_subcategories'] = df['project_subject_subcategories'].apply(lambda x: x.split(", "))
        df['teacher_prefix'] = df['teacher_prefix'].fillna('None')

        # merge in resource data from resources.csv
        df = df.merge(dfr, how='inner', on='id')

        # tokenize student description, calculate some text stats
        df['student_tokens'] = apply_by_multiprocessing(df['student_description'], tokenize, workers=workers)
        df['student_word_count'] = df['student_tokens'].apply(lambda x: len(x))
        df['student_unique_words'] = df['student_tokens'].apply(lambda x: len(set(x)))
        df['student_n_periods'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '.'))
        df['student_n_commas'] = df['student_tokens'].apply(lambda x: count_punctuation(x, ','))
        df['student_n_questions'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '?'))
        df['student_n_exclamations'] = df['student_tokens'].apply(lambda x: count_punctuation(x, '!'))
        df['student_word_len'] = df['student_tokens'].apply(lambda x: np.mean([len(token) for token in x]))

        # no longer needed
        del (df['student_tokens'])

        # same as above
        df['project_tokens'] = apply_by_multiprocessing(df['project_description'], tokenize, workers=workers)
        df['project_word_count'] = df['project_tokens'].apply(lambda x: len(x))
        df['project_unique_words'] = df['project_tokens'].apply(lambda x: len(set(x)))
        df['project_n_periods'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '.'))
        df['project_n_commas'] = df['project_tokens'].apply(lambda x: count_punctuation(x, ','))
        df['project_n_questions'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '?'))
        df['project_n_exclamations'] = df['project_tokens'].apply(lambda x: count_punctuation(x, '!'))
        df['project_word_len'] = df['project_tokens'].apply(lambda x: np.mean([len(token) for token in x]))

        # del remaining unused data
        del (df['project_tokens'])
        del (df['project_essay_1'])
        del (df['project_essay_2'])
        del (df['project_essay_3'])
        del (df['project_essay_4'])
        return df


if __name__ == "__main__":

    train_array = np.load(X_train_file)
    test_array = np.load(X_test_file)
    y_train = np.load(y_train_file)
    y_test = np.load(y_test_file)
    mapper_fp = open("../input/donors-choose-preprocess/mapper.p", "rb")
    mapper = pickle.load(mapper_fp)
    mapper_fp.close()
    
    # Calculate ratio of labels
    n_pos = np.count_nonzero(y_train)
    n_neg = len(y_train) - n_pos
    ratio = n_neg / n_pos
    

    # Start Training
    clf = xgb.XGBRegressor(eta=0.3, max_depth=4, max_delta_step=1, subsample=1, colsample_bytree=1, n_estimators=1000,
                           objective='binary:logistic', booster='dart', n_jobs=32, sample_type="weighted", normalize_type="forest")
    clf.fit(train_array, y_train, early_stopping_rounds=40, eval_metric="auc", verbose=2, eval_set=[(test_array, y_test)])

    # Open and preprocess Test file

    df_test = pd.read_csv(TEST_FILE)
    TEST = preprocess_df(df_test, WORKERS)
    TEST = mapper.transform(TEST)
    # Actual predictions

    predictions = clf.predict(TEST)
    my_submission = pd.DataFrame({'id': df_test.id, 'project_is_approved': predictions})
    my_submission.to_csv('submission_scores.csv', index=False)
