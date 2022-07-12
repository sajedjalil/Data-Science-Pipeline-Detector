import gc
import time
import numpy as np
import pandas as pd
import sys, os, psutil
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from multiprocessing import Process, Pool, Queue, JoinableQueue
import functools
from scipy.special import erfinv
import re
import unidecode
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import TweetTokenizer
import warnings
import math
from threading import Thread

warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 5000

Hash_binary = True
ensemble = True
OOF = True

non_alphanums = re.compile(u'[^A-Za-z0-9]+')

#####################################################################
# Multi processing classes
#####################################################################


class BaseWorker(Process):

    def __init__(self, q_in, q_out):
        self.task_queue: Queue = q_in
        self.result_queue: Queue = q_out
        super(BaseWorker, self).__init__()

    def check_mem(self, dsp=""):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30
        print("%s MEMORY USAGE for PID %5d : %.3f" % (dsp, pid, memoryUse))


class HashingWorker(BaseWorker):

    def run(self):
        # Get what's on the queue
        # The queue should contain :
        # an id for ordering purposes
        # a data set
        # a vectorizer that implements fit_transform
        pid = os.getpid()
        while True:
            # self.check_mem("Hashing Worker")
            id_, data_, hashing_vectorizer_ = self.task_queue.get(block=True)
            new_data = hashing_vectorizer_.fit_transform(data_)
            self.task_queue.task_done()
            self.result_queue.put([id_, new_data])
            del new_data, data_, hashing_vectorizer_, id_
            gc.collect()


class ApplySeriesWorker(BaseWorker):
    def run(self):
        # Get what's on the queue
        # The queue should contain :
        # an id for ordering purposes
        # a pd.Series
        # a function to apply to the Series
        pid = os.getpid()
        while True:
            # self.check_mem("Apply Series Worker")
            id_, data_, func_ = self.task_queue.get(block=True)
            new_data = data_.apply(func_)
            self.task_queue.task_done()
            self.result_queue.put([id_, new_data])
            del new_data, data_, func_, id_
            gc.collect()


class ApplyWorker(BaseWorker):
    def run(self):
        # Get what's on the queue
        # The queue should contain
        # an id for ordering purposes
        # a pd.DataFrame to apply a function on
        # a function to be applied on the pd.DataFrame
        # the axis on which the function has to be applied (usually 1)
        # a Boolean to say if data is passed as raw data to the function
        pid = os.getpid()
        while True:
            # self.check_mem("Apply Worker")
            id_, data_, func_, axis_, raw_ = self.task_queue.get(block=True)
            new_data = data_.apply(func_, axis=axis_, raw=raw_)
            self.task_queue.task_done()
            self.result_queue.put([id_, new_data])
            del new_data, data_, func_, id_
            gc.collect()


class HashingManager(object):
    def __init__(self, nb_workers=1):
        self.q_jobs = JoinableQueue()
        self.q_results = Queue()
        self.nb_workers = nb_workers

        self.hashing_workers = [HashingWorker(q_in=self.q_jobs, q_out=self.q_results)
                                for _ in range(self.nb_workers)]

        for wk in self.hashing_workers:
            wk.start()

    def apply(self, data_, hashing_vectorizer_):
        # Split data in chuncks and put on queue
        for id_, chunk_ in enumerate(np.array_split(data_, self.nb_workers)):
            self.q_jobs.put([id_, chunk_, hashing_vectorizer_])
            del id_, chunk_
            gc.collect()

        # Wait for tasks to complete
        # Useless to wait, the get statement will do this
        # plus the join will create a deadlock... stupid me
        # self.q_jobs.join()

        data_list = []
        for i in range(self.nb_workers):
            id, result = self.q_results.get()
            data_list.append([id, result])
            del id, result
            gc.collect()

        the_result = vstack([data_ for id_, data_ in sorted(data_list, key=lambda x: x[0])]).tocsr()

        del data_list
        gc.collect()

        return the_result

    def __del__(self):
        for wk in self.hashing_workers:
            wk.terminate()


class ApplySeriesManager(object):
    def __init__(self, nb_workers=1):
        self.q_jobs = JoinableQueue()
        self.q_results = Queue()
        self.nb_workers = nb_workers
        self.apply_series_workers = [ApplySeriesWorker(self.q_jobs, self.q_results) for _ in range(self.nb_workers)]

        for wk in self.apply_series_workers:
            wk.start()

    def apply(self, data_, func_):
        # Split data in chuncks and put on queue
        for id_, chunk_ in enumerate(np.array_split(data_, self.nb_workers)):
            self.q_jobs.put([id_, chunk_, func_])
            del id_, chunk_
            gc.collect()

        # Wait for tasks to complete
        # self.q_jobs.join()

        data_list = []
        for i in range(self.nb_workers):
            id, result = self.q_results.get()
            data_list.append([id, result])
            del id, result
            gc.collect()

        the_result = pd.concat([data_ for id_, data_ in sorted(data_list, key=lambda x: x[0])],
                               axis=0,
                               ignore_index=True)

        del data_list
        gc.collect()

        return the_result

    def __del__(self):
        for wk in self.apply_series_workers:
            wk.terminate()


class ApplyManager(object):
    def __init__(self, nb_workers=1):
        self.q_jobs = JoinableQueue()
        self.q_results = Queue()
        self.nb_workers = nb_workers
        self.apply_workers = [ApplyWorker(self.q_jobs, self.q_results) for _ in range(self.nb_workers)]

        for wk in self.apply_workers:
            wk.start()

    def apply(self, df=None, func=None, axis=0, raw=True):
        # Split data in chuncks and put on queue
        for id_, chunk_ in enumerate(np.array_split(df, self.nb_workers)):
            self.q_jobs.put([id_, chunk_, func, axis, raw])
            del id_, chunk_
            gc.collect()

        # Wait for tasks to complete
        # self.q_jobs.join()

        data_list = []
        for i in range(self.nb_workers):
            id, result = self.q_results.get()
            data_list.append([id, result])
            del id, result
            gc.collect()

        the_result = pd.concat([data_ for id_, data_ in sorted(data_list, key=lambda x: x[0])],
                               axis=0,
                               ignore_index=True)

        del data_list
        gc.collect()

        return the_result

    def __del__(self):
        for wk in self.apply_workers:
            wk.terminate()


def fit_sgd_models(csr_ridge_trn, folds, y):
    # print("THREADING EXPERIMENT")
    sgd_list = []
    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(csr_ridge_trn)):
        sgd_list.append((
            "liblinear_fold_" + str(fold_n),
            LinearSVR(C=0.025,
                      dual=True,
                      epsilon=0.0,
                      fit_intercept=True,
                      intercept_scaling=1.0,
                      loss='squared_epsilon_insensitive',
                      max_iter=50,
                      random_state=0,
                      tol=0.0001,
                      verbose=0),
            trn_idx,
            val_idx
        ))
        sgd_list.append((
            "ridge_fold_" + str(fold_n),
            Ridge(solver="sag",
                  fit_intercept=True,
                  alpha=0.5,
                  tol=0.05,
                  random_state=666,
                  max_iter=100),
            trn_idx,
            val_idx
        ))
    model_list = []
    for i in range(folds.n_splits):
        # print("Create a thread for %s" % sgd_list[i * 2][0])
        th1 = FitterThread(
            name=sgd_list[i * 2][0],
            model=sgd_list[i * 2][1],
            data=csr_ridge_trn[sgd_list[i * 2][2]],
            target=y[sgd_list[i * 2][2]]
        )
        # print("Create a thread for %s" % sgd_list[i * 2 + 1][0])
        th2 = FitterThread(
            name=sgd_list[i * 2 + 1][0],
            model=sgd_list[i * 2 + 1][1],
            data=csr_ridge_trn[sgd_list[i * 2 + 1][2]],
            target=y[sgd_list[i * 2 + 1][2]]
        )
        th1.start()
        th2.start()
        th1.join()
        th2.join()

        # Check the model has been fitted
        val_preds_1 = sgd_list[i * 2][1].predict(csr_ridge_trn[sgd_list[i * 2][3]])
        val_preds_2 = sgd_list[i * 2 + 1][1].predict(csr_ridge_trn[sgd_list[i * 2 + 1][3]])

        print("Validation score for %s = %.6f"
              % (sgd_list[i * 2][0], mean_squared_error(y[sgd_list[i * 2][3]], val_preds_1) ** .5))
        print("Validation score for %s = %.6f"
              % (sgd_list[i * 2 + 1][0], mean_squared_error(y[sgd_list[i * 2 + 1][3]], val_preds_2) ** .5))
        model_list.append(sgd_list[i * 2][1])
        model_list.append(sgd_list[i * 2 + 1][1])

    return model_list


class FitterThread(Thread):
    """ Thread that fits a model on given data """

    def __init__(self, name, model, data, target):
        Thread.__init__(self)
        self.model = model
        self.name = name
        self.data = data
        self.target = target

    def run(self):
        self.model.fit(self.data, self.target)

#####################################################################
# Text preprocessing
#####################################################################

def preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    try:
        text = unidecode.unidecode(text)
        text = str(text).lower()
    except:
        text="missing"
    text = text.lower()
    text = text.replace('❌', ' <HECROSS_MA ')
    text = text.replace('⛔', ' <NO_ENT_EM')
    text = text.replace('‼️', ' <HEAVY_EXCLAMATION> ')
    text = text.replace('⭕', ' <HEAVY_LARGE_CIRCLE> ')
    text = text.replace('❤️', ' <HEAVY_HEART_MARK> ')
    text = text.replace('❗️', ' <HEAVY_EXCLAMATION_MARK ')
    text = text.replace('✔', ' <HEAVY_CHECK_MARK> ')
    text = text.replace('⭐️', ' <WHITE_MEDIUM_STAR_MARK> ')
    text = text.replace('✅', ' <WHITE_HEAVY_CHECK_MARK> ')
    text = text.replace('☺️', ' <SMILING_FACE_EMOJI> ')
    text = text.replace('《', ' <DOUBLE_BRACKET_QUOTES> ')
    text = text.replace('➡', ' <BLACK_RIGHTWARDS_ARROW> ')
    text = text.replace('✴️', ' <EIGHT_POINTEDP_STAR> ')
    # text = re.sub('\&', " and ", text, flags=re.IGNORECASE)
    text = re.sub('\%', " percent ", text, flags=re.IGNORECASE)
    text = text.replace('.', ' <.> ')
    text = text.replace(',', ' <,> ')
    text = text.replace('，', ' <HEAVY_COMMA> ')
    text = text.replace('"', ' <"> ')
    text = text.replace("”", ' <RIGHT_DBLE_QUOT> ')
    text = text.replace("''", ' <''> ')
    text = text.replace('=', ' <=> ')
    text = text.replace('+', ' <+> ')
    text = text.replace('^^', ' <^^> ')
    text = text.replace('^', ' <^> ')
    text = text.replace('@', ' <@> ')
    text = text.replace('*', ' <*> ')
    text = text.replace(';', ' <;> ')
    text = re.sub('\$', " dollar ", text, flags=re.IGNORECASE)
    text = text.replace('!', ' <!> ')
    text = text.replace('|', ' <|> ')
    text = text.replace('∥', ' <PARALLEL_MARK> ')
    text = text.replace('?', ' <?> ')
    text = text.replace('~', ' <~> ')
    text = text.replace('[', ' <[> ')
    text = text.replace(']', ' <]> ')
    text = text.replace('{', ' <{> ')
    text = text.replace('}', ' <}> ')
    text = text.replace('(', ' <(> ')
    text = text.replace(')', ' <)> ')
    text = text.replace('--', ' <--> ')
    text = text.replace('-', ' <-> ')
    # text = text.replace("\", ' <BLACKSLASH_MARK> ')
    text = text.replace("/", ' </> ')
    # text = text.replace('[rm]', ' <REMOVED_PRICE> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <:> ')
    text = text.replace('#', ' <#> ')
    text = text.replace('gb', ' gb ')
    text = text.replace('tb', ' tb ')
    text = text.replace('karat', ' carat ')
    text = text.replace('14k', ' 14carat ')
    text = text.replace('14kt', ' 14carat ')
    text = text.replace('18k', ' 18carat ')
    text = text.replace('10k', ' 10carat ')
    text = text.replace('nmd', ' nmds ')
    text = text.replace('oz', ' oz ')
    words = text.split()

    words = ' '.join(words)
    return words


def name_preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    try:
        text = unidecode.unidecode(text)
        text = str(text).lower()
    except:
        text = "missing"

    text = text.replace('❌', ' <HEAVY_CROSS_')
    text = text.replace('⭕', ' <HEAVY_LARGE_CIRCLE> ')
    text = text.replace('⏳', ' <Hourglass_With_Flowing_Sand> ')

    text = text.replace('♨', ' <HOT_SPRINGS> ')
    text = text.replace('✌️️', ' <VICTORY_HAND> ')
    text = text.replace('⛅', ' <SUN_BEHIND_CLOUD> ')
    text = text.replace('♌', ' <LEO_MARK> ')
    text = text.replace('☠', ' <SKULL_CROSSBONES> ')
    text = text.replace('⬇', ' <DOWNWARDS_BLACK_ARROW> ')
    text = text.replace('♠️️', ' <BLACK_SPADE> ')
    text = text.replace('♤', ' <WHITE_SPADE> ')
    text = text.replace('⚫️', ' <MEDIUM_BLACK_CIRCLE> ')
    text = text.replace('⁉️', ' <EXCLAMATION_QUESTION_MARK> ')
    text = text.replace('⛄', ' <SNOW_MAN_NO_SNOW> ')
    text = text.replace('⚁', ' <DIE_FACE_EMOJI> ')
    text = text.replace('✈️', ' <Airplane_Emoji> ')
    text = text.replace('♢', ' <WHITE_DIAMON> ')
    text = text.replace('➰', ' <Curly_Loop> ')
    text = text.replace('➕', ' <HEAVY_PLUS_SIGN_EMOJI> ')
    text = text.replace('✂', ' <Black_Scissors_EMOJI> ')
    text = text.replace('❄️', ' <SNOWFLAKE_EMOJI> ')
    text = text.replace('☒', ' <BALLOT_BOX> ')
    text = text.replace('☘️', ' <SHAMROCK_EMOJI> ')
    text = text.replace('⚠', ' <WARNING_SIGN_EMOJI> ')
    text = text.replace('⚜', ' <Fleur_De_Lis_EMOJI> ')
    text = text.replace('☮', ' <PEACE_SYMBOL> ')
    text = text.replace('☄', ' <COMET_EMOJI> ')
    text = text.replace('❣', ' <HEAVY_HEART_EXCLAMATION_Mark> ')
    text = text.replace('❥', ' <ROTATE_HEAVY_BLACK_HEART_BULLET> ')
    text = text.replace('✉️', ' <ENVELOPE_EMOJI> ')
    text = text.replace('✖︎', ' <HEAVY_BLACK_CROSS> ')
    text = text.replace('‼️', ' <HEAVY_EXCLAMATION> ')
    text = text.replace('✮', ' <HEAVY_OUTLINED_BLACK_STAR> ')
    text = text.replace('★', ' <HEAVY_OUTLINED_WHITE_STAR> ')
    text = text.replace('⛔', ' <NO_ENTRY_EMOJI> ')
    text = re.sub('⚡️', ' <HEAVY_VOLTAGE_SIGN_EMOJI> ', text, flags=re.IGNORECASE)
    text = re.sub('⚡', ' <HEAVY_VOLTAGE_SIGN_EMOJI> ', text, flags=re.IGNORECASE)
    text = text.replace('❤️', ' <HEAVY_HEART_MARK> ')
    text = text.replace('☕️', ' <HOT_BEVERAGE_EMOJI> ')
    text = text.replace('❗️', ' <HEAVY_EXCLAMATION_MARK> ')
    text = text.replace('✔', ' <HEAVY_CHECK_MARK> ')
    text = re.sub('⭐️', ' <WHITE_MEDIUM_STAR_MARK> ', text, flags=re.IGNORECASE)
    text = re.sub('⭐', ' <WHITE_MEDIUM_STAR_MARK> ', text, flags=re.IGNORECASE)
    text = text.replace('❤⭐', ' <HEART_WHITE_MEDIUM_STAR_MARK> ')
    text = text.replace('❤', ' <HEAVY_HEART_MARK> ')
    text = text.replace('▪️', ' <BLACK_SMALL_SQUARE> ')
    text = text.replace('✅', ' <WHITE_HEAVY_CHECK_MARK> ')
    text = text.replace('❎', ' <Negative_Squared_Cross_Mark> ')
    text = text.replace('•', ' <BULLET_MARK> ')
    text = text.replace('●', ' <HEAVY_BULLET_MARK> ')
    text = text.replace('©', ' <COPY_RIGHT_SIGN> ')
    text = text.replace('®', ' <REGISTERED_SIGN_MARK> ')
    text = text.replace('♡', ' <HEART_SYMBOL> ')
    text = text.replace('☆', ' <WHITE_STAR_SYMBOL> ')
    text = text.replace('★', ' <BLACK_STAR_SYMBOL> ')
    text = text.replace('✨', ' <SPARKLES_EMOJI ')
    text = text.replace('☺️', ' <SMILING_FACE_EMOJI> ')
    text = text.replace('《', ' <DOUBLE_LEFT_BRACKET_QUOTES> ')
    text = text.replace('》', ' <DOUBLE_RIGHT_BRACKET_QUOTES> ')
    text = text.replace('〰', ' <WAVY_DASH_EMOJI> ')
    text = text.replace('➡', ' <BLACK_RIGHTWARDS_ARROW> ')
    text = text.replace('✴️', ' <EIGHT_POINTED_STAR> ')
    # text = re.sub('\&', " and ", text, flags=re.IGNORECASE)
    # text = re.sub('\%', " percent ", text, flags=re.IGNORECASE)
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('，', ' <HEAVY_COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace("”", ' <RIGHT_DOUBLE_QUOTATION_MARK> ')
    text = text.replace("''", ' <DOUBLE_QUOTATION_MARK> ')
    text = text.replace('=', ' <EQUAL_SIGN_MARK> ')
    text = text.replace('+', ' <PLUSL_SIGN> ')
    text = text.replace('^^', ' <_DOUBLE_CARET_MARK> ')
    text = text.replace('^', ' <CARET_MARK> ')
    text = text.replace('@', ' <AT_SIGN> ')
    text = text.replace('*', ' <STAR_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    # text = re.sub('\$', " dollar ", text, flags=re.IGNORECASE)
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('|', ' <VERTICAL_BAR_MARK> ')
    text = text.replace('∥', ' <PARALLEL_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('~', ' <TILDE_MARK> ')
    text = text.replace('[', ' <LEFT_SQUARE_BRACKET> ')
    text = text.replace(']', ' <RIGHT_SQUARE_BRACKET> ')
    text = text.replace('{', ' <LEFT_CURLY_BRACKET> ')
    text = text.replace('}', ' <RIGHT_CURLY_BRACKET> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('-', ' <HYPHENS> ')
    # text = text.replace("\", ' <BLACKSLASH_MARK> ')
    text = text.replace("/", ' <SLASH_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('[rm]', ' <REMOVED_PRICE> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    text = text.replace('#', ' <HASH> ')
    text = text.replace('gb', ' gb ')
    text = text.replace('tb', ' tb ')
    text = text.replace('karat', ' carat ')
    text = text.replace('14k', ' 14carat ')
    text = text.replace('14kt', ' 14carat ')
    text = text.replace('18k', ' 18carat ')
    text = text.replace('10k', ' 10carat ')
    text = text.replace('nmd', ' nmds ')
    text = text.replace('oz', ' oz ')
    text = re.sub("\'ve", " have ", text, flags=re.IGNORECASE)
    text = re.sub("n't", " not ", text, flags=re.IGNORECASE)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text, flags=re.IGNORECASE)
    text = re.sub("\'d", " would ", text, flags=re.IGNORECASE)
    text = re.sub("\'ll", " will ", text, flags=re.IGNORECASE)
    text = re.sub("\'s", "", text, flags=re.IGNORECASE)
    words = text.split()
    words = ' '.join(words)
    return words


def simple_preprocess(text):
    """Just making sure we have text"""
    try:
        text = unidecode.unidecode(text)
        text = str(text).lower()
    except:
        text = "missing"

    return text


def process_cond_id(z):
    try:
        if z > 5:
            return 1
        elif z < 1:
            return 1
        else:
            return z
    except:
        return 1


def process_shipping(z):
    try:
        if z not in [0, 1]:
            return 0
        else:
            return z
    except:
        return 0


def simulate_test(test):
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_.copy()
    else:
        return test


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")]])


def cpuStats(disp=""):
    """ @author: RDizzl3 @address: https://www.kaggle.com/rdizzl3"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print("%s MEMORY USAGE for PID %10d : %.3f" % (disp, pid, memoryUse))


def handle_missing_inplace(dataset):
    dataset['name'].fillna(value='missing', inplace=True)
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)


def mix_cat_name(row):
    """
      Mixes words in name with categories in the category feature
      categories are expected to be in the first field of row and separated by /
      name is expected to be in the second field of row
      """
    return " ".join([category + "_" + word
                     for category in row[0].lower().split("/") for word in row[1].lower().split()])


def mix_cat_name_cond(row):
    """
      Mixes words in name with categories in the category feature and with the item condition
      categories are expected to be in the first field of row and separated by /
      name is expected to be in the second field of row
      item condition is expected to be in the third field
      """
    return " ".join([category + "_" + word + " " + category + "_" + word + "_" + str(row[2])
                     for category in row[0].lower().split("/") for word in row[1].lower().split()])


def mix_cat_cond(row):
    """
      Mixes categories in the category feature with the item condition
      categories are expected to be in the first field of row and separated by /
      item condition is expected to be in the second field
      """
    return " ".join([category + "_" + str(row[2])
                     for category in row[0].lower().split("/")])


def add_price_statistics_on_train(trn, target, feature=None):
    """
      Prices are aggregated by brand to compute statistics
      These stats are then merged back into train and test datasets
      @author: Kueipo @address: https://www.kaggle.com/kueipo
      @param: trn: training data taht is expected to contain a 'price' feature
      @param: sub: test data
      """
    train = trn[[feature]].copy()
    train["price"] = target
    train = train[train[feature].notnull()]
    stats = train.groupby(feature)['price'].agg({'median', "mean", 'std', 'min', 'max'}).reset_index()
    stats["std"].fillna(0, inplace=True)
    stats.columns = [feature, feature + "_median", feature + "_mean",
                     feature + "_std", feature + "_min", feature + "_max"]
    trn = pd.merge(trn, stats, how='left', on=feature)

    # Now set unknown values to the overall median, std, min or max
    trn.loc[trn[feature + "_median"].isnull(), feature + "_median"] = np.median(target)
    trn.loc[trn[feature + "_mean"].isnull(), feature + "_mean"] = np.mean(target)
    trn.loc[trn[feature + "_std"].isnull(), feature + "_std"] = np.std(target)
    trn.loc[trn[feature + "_min"].isnull(), feature + "_min"] = np.min(target)
    trn.loc[trn[feature + "_max"].isnull(), feature + "_max"] = np.max(target)

    del train
    gc.collect()

    return trn, stats


def add_price_statistics_on_test(sub, stats, target, feature=None):
    """
      Prices are aggregated by brand to compute statistics
      These stats are then merged back into train and test datasets
      @author: Kueipo @address: https://www.kaggle.com/kueipo
      @param: trn: training data taht is expected to contain a 'price' feature
      @param: sub: test data
      """
    sub = pd.merge(sub, stats, how='left', on=feature)
    # Now set unknown values to the overall median, std, min or max
    sub.loc[sub[feature + "_median"].isnull(), feature + "_median"] = np.median(target)
    sub.loc[sub[feature + "_mean"].isnull(), feature + "_mean"] = np.mean(target)
    sub.loc[sub[feature + "_std"].isnull(), feature + "_std"] = np.std(target)
    sub.loc[sub[feature + "_min"].isnull(), feature + "_min"] = np.min(target)
    sub.loc[sub[feature + "_max"].isnull(), feature + "_max"] = np.max(target)

    return sub


def string_len(x):
    """ Simple function that returns the len of a string """
    try:
        return len(str(x))
    except:
        return 0


def word_count(x, sep=None):
    """ Simple function that returns the number of words in a string """
    try:
        return len(str(x).split(sep))
    except:
        return 0


def add_character_and_word_lengths(data, app_series_man_):
    """
      @author: Olivier @address: https://www.kaggle.com/ogrellier
      Function used to create additional features.
      All of this is parallelized using process pooling
      """
    # Apply description length in parallel
    data["desc_len"] = app_series_man_.apply(data_=data["item_description"].fillna("missing"), func_=string_len)
    data["desc_word_len"] = app_series_man_.apply(data_=data["item_description"].fillna("missing"),
                                                  func_=functools.partial(word_count, sep=None))
    data["nb_categories"] = app_series_man_.apply(data_=data["category_name"].fillna("missing"),
                                                  func_=functools.partial(word_count, sep="/"))
    data["name_len"] = app_series_man_.apply(data_=data["name"].fillna("missing"), func_=string_len)
    data["name_word_len"] = app_series_man_.apply(data_=data["name"].fillna("missing"),
                                                  func_=functools.partial(word_count, sep=None))
    # Add ratios
    # data["ratio_1"] = data["name_len"] / (data["name_word_len"] + 1)
    # data["ratio_2"] = data["desc_len"] / (data["desc_word_len"] + 1)
    # data["ratio_3"] = data["name_len"] / (data["desc_len"] + 1)
    # data["ratio_4"] = data["name_word_len"] / (data["desc_word_len"] + 1)


def add_combination_category_name(data, app_man_):

    data["mix_cat_name"] = app_man_.apply(df=data[["category_name", "name"]].fillna("missing"),
                                          func=mix_cat_name,
                                          axis=1,
                                          raw=True)



def add_categories_and_mix_with_condition(df):
    for i in range(3):
        # Create new features
        df["category_name_" + str(i)] = df["category_name"].str.split("/").str[i].fillna("no_cat")
        df["cat_cond_" + str(i)] = df["category_name_" + str(i)] + '|' + df["item_condition_id"].apply(lambda x: str(x))


def preprocess_text_features(df, app_series_man_):
    """
      Utility function to apply text pre-processing by Kueipo to name, brand and description
      but in parallel
      """
    df["item_description"] = app_series_man_.apply(data_=df["item_description"].fillna("missing"), func_=preprocess)
    df["name"] = app_series_man_.apply(data_=df["name"].fillna("missing"), func_=preprocess)
    df["brand_name"] = app_series_man_.apply(data_=df["brand_name"].fillna("missing"), func_=simple_preprocess)
    df["category_name"] = app_series_man_.apply(data_=df["category_name"].fillna("missing"), func_=simple_preprocess)
    df["shipping"] = app_series_man_.apply(data_=df["shipping"].fillna(-1), func_=process_shipping)
    df["item_condition_id"] = app_series_man_.apply(data_=df["item_condition_id"].fillna(-1), func_=process_cond_id)


def add_d(text):
    """
      Simple text modification used on description to ensure
      words in description are not hashed in the same space as name
      """
    return " ".join(["d_" + w for w in TweetTokenizer().tokenize(text)])


def add_b(text):
    """
      Simple text modification used brand to ensure
      brands are not hashed in the same space as name and description
      """
    return " ".join(["b_" + w for w in text.split()])


# def get_hashing_features(train, test, Hash_binary, start_time):
def get_hashing_features(df, hash_binary, start_time, app_series_man_, app_man_, hash_man_):
    # df = pd.concat([train, test])
    dim = 24
    cv_name = HashingVectorizer(
        n_features=2 ** dim,
        ngram_range=(1, 2),
        norm=None,
        alternate_sign=False,
        tokenizer=TweetTokenizer().tokenize,
        binary=hash_binary
    )
    X_name = hash_man_.apply(data_=df["name"], hashing_vectorizer_=cv_name)

    cv_cat_name = HashingVectorizer(
        n_features=2 ** dim,
        ngram_range=(1, 2),
        norm=None,
        alternate_sign=False,
        tokenizer=None,  # TweetTokenizer().tokenize,
        binary=hash_binary
    )
    add_combination_category_name(df, app_man_)
    X_name += hash_man_.apply(data_=df["mix_cat_name"], hashing_vectorizer_=cv_cat_name)

    desc_hash = HashingVectorizer(n_features=2 ** dim,
                                  norm=None,
                                  alternate_sign=False,
                                  tokenizer=None,  # TweetTokenizer().tokenize,
                                  binary=hash_binary
                                  # stop_words='english'
                                  )
    df["get_desc_hash_out"] = app_series_man_.apply(data_=df["item_description"].fillna("missing"), func_=add_d)

    X_name += hash_man_.apply(data_=df["get_desc_hash_out"], hashing_vectorizer_=desc_hash)

    df.drop("get_desc_hash_out", axis=1, inplace=True)
    gc.collect()

    df["get_brand_hash_out"] = app_series_man_.apply(data_=df["brand_name"].fillna("missing"), func_=add_b)

    brd_hash = HashingVectorizer(n_features=2 ** dim,
                                 norm=None,
                                 alternate_sign=False,
                                 binary=hash_binary
                                 )
    X_name += hash_man_.apply(data_=df["get_brand_hash_out"], hashing_vectorizer_=brd_hash)

    df.drop("get_brand_hash_out", axis=1, inplace=True)
    gc.collect()
    print('[{}] Finished hashing dataset'.format(time.time() - start_time))

    return X_name


def get_tfidf_features_for_train(train, hash_man_):
    # Create wordbatch tfidf
    wb = HashingVectorizer(
        n_features=2 ** 20,
        ngram_range=(1, 1),
        norm=None,
        alternate_sign=False,
        tokenizer=TweetTokenizer().tokenize,
        binary=Hash_binary
    )
    X_name = hash_man_.apply(data_=train["name"], hashing_vectorizer_=wb)

    # print("Wordbatch hashing done")

    # Remove features with document frequency <=1
    # This is not a stateless step
    # If clipping is an np.array it will take a massive amount of memory
    # clipping = np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    clipping = np.array((X_name.sum(axis=0) >= 1))[0]
    # cpuStats()
    print("Clipping computed")
    X_name = X_name[:, clipping]
    gc.collect()
    # cpuStats()
    print("X_name reduced")
    # return matrix and wordbatch for future use
    return X_name, wb, clipping


def get_tfidf_features_for_test(test, wb, clipping, hash_man_):

    X_name = hash_man_.apply(data_=test["name"], hashing_vectorizer_=wb)
    X_name = X_name[:, clipping]

    return X_name


class OHEManager(object):

    def __init__(self, feature_name=None, min_df=5):
        self.name = feature_name
        self.indexer = None
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.indices = None
        self.cols = None
        self.min_df = min_df

    def add_factorized_feature_on_train(self, trn):
        trn["fact_" + self.name], self.indexer = pd.factorize(trn[self.name])

    def add_factorized_feature_on_test(self, sub):
        if self.indexer is None:
            raise ValueError("indexer has not been fitted yet")

        sub["fact_" + self.name] = self.indexer.get_indexer(sub[self.name])

    def get_feature_for_sgd_train(self, trn):
        dummies = self.ohe.fit_transform(trn[["fact_" + self.name]].replace(-1, 999))
        self.indices = np.arange(dummies.shape[1])
        self.cols = np.array((dummies.sum(axis=0) >= self.min_df))[0]
        return dummies[:, self.indices[self.cols]]

    def get_feature_for_sgd_test(self, sub):
        dummies = self.ohe.transform(sub[["fact_" + self.name]].replace(-1, 999))
        return dummies[:, self.indices[self.cols]]


def get_numerical_features_for_sgd(df):
    # Factors cannot be used by linear models
    numericals = [
        "fact_category_name_0", "fact_category_name_1", "fact_category_name_2",
        "fact_cat_cond_0", "fact_cat_cond_1", "fact_cat_cond_2",
        "fact_brand_name",
        "item_condition_id",
        "shipping",
        "desc_len", "desc_word_len", "name_len", "name_word_len", "nb_categories",
        "brand_name_median", "brand_name_std", "brand_name_min", "brand_name_max", "distance",
        "category_name_median", "category_name_std", "category_name_min", "category_name_max",
        "ratio_1", "ratio_2", "ratio_3", "ratio_4",
    ]
    return [f_ for f_ in numericals if f_ in df]


def get_numerical_features_for_lgb(df):
    numericals = [
        "fact_category_name_0", "fact_category_name_1", "fact_category_name_2",
        "fact_cat_cond_0", "fact_cat_cond_1", "fact_cat_cond_2",
        "fact_brand_name",
        "item_condition_id", "shipping",
        "desc_len", "desc_word_len", "name_len", "name_word_len", "nb_categories",
        "brand_name_median", "brand_name_std", "brand_name_min", "brand_name_max", "distance",
        "category_name_median", "category_name_std", "category_name_min", "category_name_max",
        "sgd_liblinear", "sgd_ridge", "liblinear_ridge",
        "fact_name_0", "fact_name_1", "fact_name_2", "fact_name_3", "fact_name_4", "fact_name_5",
        "ratio_1", "ratio_2", "ratio_3", "ratio_4",
    ]

    return [f_ for f_ in numericals if f_ in df]


def get_numerical_features(df, numericals=None,
                           gaussian=True, rank=False,
                           minmax_skl=None):
    num_feats = [f_ for f_ in numericals if f_ in df]
    # print(num_feats)

    if gaussian:
        if rank:
            num_df = df[num_feats].copy()
            for f_ in num_feats:
                num_df[f_] = (num_df[f_].rank() - num_df.shape[0] * .5) / (num_df.shape[0] * .5)
            num_df[num_df >= 1.0] = 0.99999
            num_df[num_df <= -1.0] = -0.99999
        else:
            if minmax_skl is None:
                minmax_skl = MinMaxScaler(feature_range=(-1 + 1e-6, 1 - 1e-6)).fit(df[num_feats])

            num_df = pd.DataFrame(data=minmax_skl.transform(df[num_feats]),
                                  columns=num_feats)

            # minmax_skl can be used on data with min max different
            # than the data it used to fit on, so we need to clip it
            num_df = np.clip(a=num_df, a_min=-1 + 1e-6, a_max=1 - 1e-6)

        # Use Inverse of error function to shape like gaussian
        for f_ in num_feats:
            num_df[f_] = erfinv(num_df[f_].values)
            the_mean = num_df[f_].mean()
            num_df[f_] -= the_mean
            # print(f_, the_mean)

        return csr_matrix(num_df[num_feats].values), minmax_skl
    else:
        return csr_matrix(df[num_feats].values)


PROD = "production"
PROD_OOF = "production_with_oof"
VALID_TRN = "train_only_validation"
FAST_VALID = "fast_validation_set"
STAGE2_OOF = "stage2_with_OOF_validation"
STAGE2_PROD = "Complete_Stage2_rehearsal"


class DataManager(object):
    def __init__(self, mode, ratio):
        self.mode = mode
        self.idx = None
        self.ratio = ratio

    def get_train_data(self):
        if self.mode in [PROD, PROD_OOF]:
            train = pd.read_table('../input/train.tsv', engine='c').rename(columns={"train_id": "id"})
            train.ix[train.brand_name == 'PINK', 'brand_name'] = 'PINKBRAND'
        elif self.mode in [STAGE2_OOF, STAGE2_PROD]:
            train = pd.read_table('../input/train.tsv', engine='c').head(500000).rename(columns={"train_id": "id"})
            train.ix[train.brand_name == 'PINK', 'brand_name'] = 'PINKBRAND'
        elif self.mode == FAST_VALID:
            np.random.seed(0)
            data = pd.read_table('../input/train.tsv', engine='c')
            if self.idx is None:
                self.idx = np.arange(data.shape[0])
                np.random.shuffle(self.idx)
            data = data.iloc[self.idx]
            train = data.head(int(data.shape[0] * (1 - self.ratio) / 10)).rename(columns={"train_id": "id"})
            # test = data.tail(int(data.shape[0] *self. ratio / 10)).rename(columns={"train_id": "id"})
            del data
            gc.collect()
        else:
            # Use train for train and test
            # Used to check the whole process fully works and scores are fine
            # in particular makes sure train and test matrices in all steps are in sync
            np.random.seed(0)
            data = pd.read_table('../input/train.tsv', engine='c')
            if self.idx is None:
                self.idx = np.arange(data.shape[0])
                np.random.shuffle(self.idx)
            data = data.iloc[self.idx]
            train = data.head(int(data.shape[0] * (1 - self.ratio))).rename(columns={"train_id": "id"})
            del data
            gc.collect()
        return train

    def get_test_data(self):
        if self.mode in [PROD, PROD_OOF]:
            test = pd.read_table('../input/test.tsv', engine='c').rename(columns={"test_id": "id"})
            test.ix[test.brand_name == 'PINK', 'brand_name'] = 'PINKBRAND'
        elif self.mode in [STAGE2_OOF, STAGE2_PROD]:
            test = pd.read_table('../input/test.tsv', engine='c').rename(columns={"test_id": "id"})
            test.ix[test.brand_name == 'PINK', 'brand_name'] = 'PINKBRAND'
            test = simulate_test(test)
        elif self.mode == FAST_VALID:
            np.random.seed(0)
            data = pd.read_table('../input/train.tsv', engine='c')
            if self.idx is None:
                self.idx = np.arange(data.shape[0])
                np.random.shuffle(self.idx)
            data = data.iloc[self.idx]
            test = data.tail(int(data.shape[0] * self.ratio / 10)).rename(columns={"train_id": "id"})
            del data
            gc.collect()
        else:
            # Use train for train and test
            # Used to check the whole process fully works and scores are fine
            # in particular makes sure train and test matrices in all steps are in sync
            np.random.seed(0)
            data = pd.read_table('../input/train.tsv', engine='c')
            if self.idx is None:
                self.idx = np.arange(data.shape[0])
                np.random.shuffle(self.idx)
            data = data.iloc[self.idx]
            test = data.tail(int(data.shape[0] * self.ratio)).rename(columns={"train_id": "id"})
            del data
            gc.collect()
        return test


def add_name_features_for_train(df=None):
    indexers = []
    for i in range(6):
        # print(f_ + "_" + str(i))
        df["fact_name_" + str(i)] = df["name"].fillna("").str.split().str[i].fillna("no_name")
        df["fact_name_" + str(i)], indexer = pd.factorize(df["fact_name_" + str(i)])
        indexers.append(indexer)
        gc.collect()

    return indexers


def add_name_features_for_test(df=None, indexers=None):
    for i in range(6):
        # print(f_ + "_" + str(i))
        df["fact_name_" + str(i)] = df["name"].fillna("").str.split().str[i].fillna("no_name")
        df["fact_name_" + str(i)] = indexers[i].get_indexer(df["fact_name_" + str(i)])


def get_sgd_oof_predictions(csr_ridge_trn, folds, models_list, y):
    len_pred = csr_ridge_trn.shape[0]
    liblinear_preds = np.zeros(len_pred)
    ridge_preds = np.zeros(len_pred)
    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(csr_ridge_trn)):
        liblinear_preds[val_idx] = models_list[fold_n * 2].predict(csr_ridge_trn[val_idx])
        ridge_preds[val_idx] = models_list[fold_n * 2 + 1].predict(csr_ridge_trn[val_idx])
        if y is not None:
            score_liblinear = mean_squared_error(y[val_idx], liblinear_preds[val_idx]) ** .5
            print("SGD L2 Fold %2d : %.6f" % (fold_n + 1, score_liblinear))
            score_ridge = mean_squared_error(y[val_idx], ridge_preds[val_idx]) ** .5
            print("SGD L1 Fold %2d : %.6f" % (fold_n + 1, score_ridge))
    return liblinear_preds, ridge_preds


def get_sgd_test_predictions(csr_ridge_sub, folds, models_list, y=None):
    len_pred = csr_ridge_sub.shape[0]
    liblinear_preds = np.zeros(len_pred)
    ridge_preds = np.zeros(len_pred)
    for fold_n in range(folds.n_splits):
        liblinear_preds += models_list[fold_n * 2].predict(csr_ridge_sub) / folds.n_splits
        ridge_preds += models_list[fold_n * 2 + 1].predict(csr_ridge_sub) / folds.n_splits
    if y is not None:
        score_liblinear = mean_squared_error(y, liblinear_preds) ** .5
        print("liblinear Test  : %.6f" % score_liblinear)
        score_ridge = mean_squared_error(y, ridge_preds) ** .5
        print("SGD Test L1 : %.6f" % score_ridge)
    return liblinear_preds, ridge_preds


# def corrupt(df):
#     # type: (pd.DataFrame) -> pd.DataFrame
#
#     print("TEST DATA WILL BE CORRUPTED")
#     # condition
#     idx = df.index
#     df.loc[idx[0], "item_condition_id"] = 10
#     df.loc[idx[1], "item_condition_id"] = "b"
#     df.loc[idx[2], "item_condition_id"] = np.inf
#     df.loc[idx[3], "item_condition_id"] = np.nan
#     df.loc[idx[4], "item_condition_id"] = -np.inf
#     # name
#     df.loc[idx[5], "name"] = 10
#     df.loc[idx[6], "name"] = np.inf
#     df.loc[idx[7], "name"] = np.nan
#     # item description
#     df.loc[idx[8], "item_description"] = 10
#     df.loc[idx[9], "item_description"] = np.inf
#     df.loc[idx[10], "item_description"] = np.nan
#     # category_name
#     df.loc[idx[11], "category_name"] = 10
#     df.loc[idx[12], "category_name"] = np.inf
#     df.loc[idx[13], "category_name"] = np.nan
#     # brand_name
#     df.loc[idx[14], "brand_name"] = 10
#     df.loc[idx[15], "brand_name"] = np.inf
#     df.loc[idx[16], "brand_name"] = np.nan
#     # shipping
#     df.loc[idx[17], "shipping"] = 856
#     df.loc[idx[18], "shipping"] = np.inf
#     df.loc[idx[19], "shipping"] = np.nan
#     df.loc[idx[20], "shipping"] = "shipping_cost"
#
#     print(df.isnull().sum()[df.isnull().sum()>0])
#
#     return df


def main(mode=PROD, ratio=0.1):
    # Set start time
    start_time = time.time()

    main_apply_series_man = ApplySeriesManager(nb_workers=4)
    main_apply_man = ApplyManager(nb_workers=4)
    main_hash_man = HashingManager(nb_workers=4)

    # Read training data and test dataset
    data_man = DataManager(mode=mode, ratio=ratio)
    train = data_man.get_train_data()

    print(train["shipping"].value_counts())
    print(train["item_condition_id"].value_counts())

    # Remove 0 price in train dataset
    train = train.loc[train.price > 0]
    train.reset_index(drop=True, inplace=True)

    # Get target out of train  dataset
    y = np.log1p(train["price"].values)

    print('[{}] Finished to load train data'.format(time.time() - start_time), ' train shape: ', train.shape)

    # Create price statistics by brand
    # brand price mappers should contain mapping for mean, std,
    train, brand_price_mappers = add_price_statistics_on_train(trn=train, target=y, feature="brand_name")
    train, cat_price_mappers = add_price_statistics_on_train(trn=train, target=y, feature="category_name")

    # Replace NaN
    handle_missing_inplace(train)
    print('[{}] Finished to handle missing'.format(time.time() - start_time))

    # Add raw text character and word length - this is stateless
    add_character_and_word_lengths(data=train, app_series_man_=main_apply_series_man)
    print('[{}] Finished to add text lengths in train and test'.format(time.time() - start_time))

    # Pre-process text features - this is stateless
    preprocess_text_features(df=train, app_series_man_=main_apply_series_man)
    cpuStats()
    print('[{}] Finished to pre process text in train'.format(time.time() - start_time))

    # Get hashing space - this is stateless
    csr_name_trn = get_hashing_features(train, Hash_binary, start_time,
                                        main_apply_series_man, main_apply_man, main_hash_man)
    cpuStats()

    # Reduce Hashing space
    trn_not_zeros = np.array((csr_name_trn.sum(axis=0) != 0))[0]
    cpuStats()
    print("Zeros found created")
    csr_name_trn = csr_name_trn[:, trn_not_zeros]
    gc.collect()
    cpuStats()

    # Get brand as dummies - this is stateful so we may need to do this again
    # Keep for now
    brand_man = OHEManager(feature_name="brand_name")
    brand_man.add_factorized_feature_on_train(trn=train)
    csr_brand_trn = brand_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get brand for sgd in train'.format(time.time() - start_time))

    cond_man = OHEManager(feature_name="item_condition_id")
    cond_man.add_factorized_feature_on_train(trn=train)
    csr_cond_trn = cond_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get condition for sgd in train'.format(time.time() - start_time))

    ship_man = OHEManager(feature_name="shipping")
    ship_man.add_factorized_feature_on_train(trn=train)
    csr_ship_trn = ship_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get shipping for sgd in train'.format(time.time() - start_time))

    add_categories_and_mix_with_condition(df=train)

    # Get categories as dummies
    cat0_man = OHEManager(feature_name="category_name_0")
    cat0_man.add_factorized_feature_on_train(trn=train)
    csr_cat0_trn = cat0_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get category_name_0 for sgd in train'.format(time.time() - start_time))

    cat1_man = OHEManager(feature_name="category_name_1")
    cat1_man.add_factorized_feature_on_train(trn=train)
    csr_cat1_trn = cat1_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get category_name_1 for sgd in train'.format(time.time() - start_time))

    cat2_man = OHEManager(feature_name="category_name_2")
    cat2_man.add_factorized_feature_on_train(trn=train)
    csr_cat2_trn = cat2_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get category_name_2 for sgd in train'.format(time.time() - start_time))

    csr_cat_trn = hstack((csr_cat0_trn, csr_cat1_trn, csr_cat2_trn)).tocsr()
    del csr_cat0_trn, csr_cat1_trn, csr_cat2_trn
    gc.collect()

    # Get categories + condition as dummies
    cat0_cond_man = OHEManager(feature_name="cat_cond_0")
    cat0_cond_man.add_factorized_feature_on_train(trn=train)
    csr_cat0_cond_trn = cat0_cond_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get category_name_0 for sgd in train'.format(time.time() - start_time))

    cat1_cond_man = OHEManager(feature_name="cat_cond_1")
    cat1_cond_man.add_factorized_feature_on_train(trn=train)
    csr_cat1_cond_trn = cat1_cond_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get category_name_1 for sgd in train'.format(time.time() - start_time))

    cat2_cond_man = OHEManager(feature_name="cat_cond_2")
    cat2_cond_man.add_factorized_feature_on_train(trn=train)
    csr_cat2_cond_trn = cat2_cond_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get category_name_2 for sgd in train'.format(time.time() - start_time))

    csr_cat_cond_trn = hstack((csr_cat0_cond_trn, csr_cat1_cond_trn, csr_cat2_cond_trn)).tocsr()
    del csr_cat0_cond_trn, csr_cat1_cond_trn, csr_cat2_cond_trn
    gc.collect()

    train["brd_cond"] = train["brand_name"].astype(str) + "|" + train["item_condition_id"].astype(str)
    brd_cond_man = OHEManager(feature_name="brd_cond")
    brd_cond_man.add_factorized_feature_on_train(trn=train)
    csr_brd_cond_trn = brd_cond_man.get_feature_for_sgd_train(trn=train)
    print('[{}] Finished get brand_cond for sgd in train'.format(time.time() - start_time))

    cpuStats("csr matrices creation done for train")

    csr_num_trn, trn_minmax_skl = get_numerical_features(
        train,
        numericals=get_numerical_features_for_sgd(train),
        gaussian=True,
        rank=False
    )
    print('[{}] Finished get numerical features'.format(time.time() - start_time))

    # Build csr matrices for SGD fitting and predictions
    print("Cond       : ", csr_cond_trn.shape)
    print("Ship       : ", csr_ship_trn.shape)
    print("Brand      : ", csr_brand_trn.shape)
    print("Brand Cond : ", csr_brd_cond_trn.shape)
    print("Num        : ", csr_num_trn.shape)
    print("Cat        : ", csr_cat_trn.shape)
    print("Cat_Cond   : ", csr_cat_cond_trn.shape)
    print("Name       : ", csr_name_trn.shape)
    csr_ridge_trn = hstack((
        csr_cond_trn,
        csr_ship_trn,
        csr_brand_trn,
        csr_brd_cond_trn,
        csr_num_trn,
        csr_cat_trn,
        csr_cat_cond_trn,
        csr_name_trn,
    )).tocsr()

    del csr_num_trn
    del csr_brand_trn
    del csr_cat_cond_trn
    del csr_cat_trn
    gc.collect()
    cpuStats()
    print("hstack done for train matrices with shape : ", csr_ridge_trn.shape)

    # Fit SGD models
    folds = KFold(n_splits=5, shuffle=True, random_state=10)
    models_list = fit_sgd_models(csr_ridge_trn, folds, y)

    # Predict OOF data
    oof_liblinear_preds, oof_ridge_preds = get_sgd_oof_predictions(csr_ridge_trn, folds, models_list, y)

    del csr_ridge_trn
    gc.collect()
    cpuStats()
    print('[{}] Finished delete training csr matrices'.format(time.time() - start_time))

    print("=" * 50)
    print("Finished training ridge/liblinear")
    print("=" * 50)

    # Add OOF predictions to train data
    train["sgd_liblinear"] = np.expm1(oof_liblinear_preds)
    train["sgd_ridge"] = np.expm1(oof_ridge_preds)
    train["liblinear_ridge"] = .50 * np.expm1(oof_liblinear_preds) + .50 * np.expm1(oof_ridge_preds)

    # add name features
    name_indexers = add_name_features_for_train(df=train)
    cpuStats()
    print('[{0:7.1f}] Finished create name features'.format(time.time() - start_time))
    print(train.isnull().sum())
    csr_num_trn = get_numerical_features(
        train,
        numericals=get_numerical_features_for_lgb(train),
        gaussian=False,
        rank=True
    )
    print('[{}] Finished get train numerical data for Light GBM'.format(time.time() - start_time))

    print("NAN IN NUM : ", np.isnan(np.array((csr_num_trn.sum(axis=0)))[0]).sum())

    name_trn = train[["name"]].copy()
    del train
    gc.collect()
    cpuStats()
    print('[{}] Finished deleting train data'.format(time.time() - start_time))

    indices_low = np.arange(csr_name_trn.shape[1])
    util_cols_low_trn = np.array((csr_name_trn.sum(axis=0) >= 500))[0]
    csr_name_trn = csr_name_trn[:, indices_low[util_cols_low_trn]]

    indices_high = np.arange(csr_name_trn.shape[1])
    util_cols_high_trn = np.array((csr_name_trn.sum(axis=0) < 200000))[0]
    csr_name_trn = csr_name_trn[:, indices_high[util_cols_high_trn]]

    gc.collect()
    cpuStats()
    print("Feature reduction done, X_name shape ", csr_name_trn.shape, " after col pruning")

    csr_tfidfname_trn, wordbatch_tfidf, clipping = get_tfidf_features_for_train(name_trn, hash_man_=main_hash_man)
    cpuStats()
    print('[{}] TFIDF WordBatch features done for name'.format(time.time() - start_time))

    csr_lgb_trn = hstack((
        csr_tfidfname_trn,
        csr_name_trn,
        csr_num_trn,
    )).tocsr()

    del csr_name_trn, csr_num_trn, csr_tfidfname_trn
    gc.collect()
    cpuStats("After csr_lgb_trn stacking")

    # Create parameters
    params = {
        "objective": "regression",
        'metric': {'rmse'},
        "boosting_type": "gbdt",
        "verbosity": 0,
        "num_threads": 4,
        "bagging_fraction": 0.78,
        "feature_fraction": 0.76,
        "learning_rate": 0.4,
        "min_child_weight": 197,
        "min_data_in_leaf": 197,
        "num_leaves": 103,
    }

    params_l2 = {
        'learning_rate': 0.4,
        'application': 'regression_l2',
        'max_depth': 4,
        'num_leaves': 70,
        'verbosity': -1,
        "min_split_gain": 0,
        'lambda_l1': 4,
        'subsample': 1,
        "bagging_freq": 1,
        'colsample_bytree': 1,
        'metric': 'RMSE',
        'nthread': 4
    }

    lgb1_rounds = 500
    lgb2_rounds = 4000

    if ensemble:
        # Run LGB 1 and LGB 2
        # Reuse folds defined for ridge/sgd
        # to avoid overfitting
        if mode in [PROD_OOF, VALID_TRN, STAGE2_OOF, FAST_VALID]:
            for fold_n, (trn_idx, val_idx) in enumerate(folds.split(csr_lgb_trn)):
                d_train = lgb.Dataset(csr_lgb_trn[trn_idx], label=y[trn_idx])  # , max_bin=8192)
                d_valid = lgb.Dataset(csr_lgb_trn[val_idx], label=y[val_idx])  # , max_bin=8192)
                watchlist = [d_train, d_valid]
                cpuStats()
                # Train lgb l1
                lgb_l1 = lgb.train(
                    params=params,
                    train_set=d_train,
                    num_boost_round=lgb1_rounds,
                    valid_sets=watchlist,
                    verbose_eval=100)
                # Train lgb l2
                lgb_l2 = lgb.train(
                    params=params_l2,
                    train_set=d_train,
                    num_boost_round=lgb2_rounds,
                    valid_sets=watchlist,
                    verbose_eval=500)

                break
            # Check OOF score of ensemble ?
            oof_l1 = lgb_l1.predict(csr_lgb_trn[val_idx])
            oof_l2 = lgb_l2.predict(csr_lgb_trn[val_idx])

            print("OOF error L1   : %.6f "
                  % mean_squared_error(y[val_idx], oof_l1) ** .5)
            print("OOF error L2   : %.6f "
                  % mean_squared_error(y[val_idx], oof_l2) ** .5)
            print("OOF error Mix1 : %.6f "
                  % mean_squared_error(y[val_idx], oof_l1 * .3 + oof_l2 * .7) ** .5)
            oof_preds = np.expm1(oof_l2) * .7 + np.expm1(oof_l1) * .3
            print("OOF error Mix2 : %.6f "
                  % mean_squared_error(y[val_idx], np.log1p(oof_preds)) ** .5)
        else:
            d_train = lgb.Dataset(csr_lgb_trn, label=y)
            watchlist = [d_train]
            # Train lgb l1
            lgb_l1 = lgb.train(
                params=params,
                train_set=d_train,
                num_boost_round=lgb1_rounds,  # was 700, 400 comes from OOF
                valid_sets=watchlist,
                verbose_eval=100)
            # Train lgb l2
            lgb_l2 = lgb.train(
                params=params_l2,
                train_set=d_train,
                num_boost_round=lgb2_rounds,  # Beware this is not the same as in OOF mode
                valid_sets=watchlist,
                verbose_eval=500)

    else:
        # Only LGB L2 is trained
        if mode in [PROD_OOF, VALID_TRN, STAGE2_OOF, FAST_VALID]:
            for fold_n, (trn_idx, val_idx) in enumerate(folds.split(csr_lgb_trn)):
                d_train = lgb.Dataset(csr_lgb_trn[trn_idx], label=y[trn_idx])  # , max_bin=8192)
                d_valid = lgb.Dataset(csr_lgb_trn[val_idx], label=y[val_idx])  # , max_bin=8192)
                watchlist = [d_train, d_valid]
                cpuStats()
                # Train lgb l2
                lgb_l2 = lgb.train(
                    params=params_l2,
                    train_set=d_train,
                    num_boost_round=lgb2_rounds,
                    valid_sets=watchlist,
                    verbose_eval=500)

                break
                # Check OOF score of ensemble ?

        else:
            d_train = lgb.Dataset(csr_lgb_trn, label=y)
            watchlist = [d_train]
            # Train lgb l2
            lgb_l2 = lgb.train(
                params=params_l2,
                train_set=d_train,
                num_boost_round=lgb2_rounds,  # Beware this is not the same as in OOF mode
                valid_sets=watchlist,
                verbose_eval=500)

    del csr_lgb_trn
    gc.collect()
    cpuStats()

    print("=" * 50)
    print("TRAINING PART HAS NOW COMPLETE, STARTING PREDICTION PART")
    print("=" * 50)

    # Read test dataset
    test2 = data_man.get_test_data()

    # test2 = corrupt(test2)

    print('[{}] Finished to load test data'.format(time.time() - start_time), ' test shape: ', test2.shape)

    y_test2 = None
    if "price" in test2:
        y_test2 = np.log1p(test2["price"].values)
    submission = test2[["id"]].copy()

    # Building and predicting using batches
    batch_size = 100000
    submission_preds = np.zeros(len(test2))

    # Go through batches
    for i_b in range(math.ceil(len(test2) / batch_size)):
        print("Building and predicting for batch %4d" % i_b)
        s_b, e_b = i_b * batch_size, (i_b + 1) * batch_size

        test = test2.iloc[s_b: e_b].copy()

        y_test = None
        if "price" in test:
            y_test = np.log1p(test["price"].values)

        # Create price statistics by brand
        # brand price mappers should contain mapping for mean, std,
        test = add_price_statistics_on_test(sub=test, stats=brand_price_mappers, target=y, feature="brand_name")
        test = add_price_statistics_on_test(sub=test, stats=cat_price_mappers, target=y, feature="category_name")

        # Replace NaN
        handle_missing_inplace(test)
        # print('[{}] Finished to handle missing'.format(time.time() - start_time))

        # Now create test set matrices, the one we miss is the hashing matrix
        add_character_and_word_lengths(data=test, app_series_man_=main_apply_series_man)
        # cpuStats()
        preprocess_text_features(df=test, app_series_man_=main_apply_series_man)
        # cpuStats()

        csr_name_sub2 = get_hashing_features(test, Hash_binary, start_time,
                                             main_apply_series_man, main_apply_man, main_hash_man)
        # Reduce hashing space like train
        csr_name_sub = csr_name_sub2[:, trn_not_zeros]
        del csr_name_sub2
        gc.collect()
        # cpuStats()

        # Get brand as dummies - this is stateful so we may need to do this again
        # Keep for now
        brand_man.add_factorized_feature_on_test(sub=test)
        csr_brand_sub = brand_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get brand for sgd in test'.format(time.time() - start_time))

        cond_man.add_factorized_feature_on_test(sub=test)
        csr_cond_sub = cond_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get condition for sgd in test'.format(time.time() - start_time))

        ship_man.add_factorized_feature_on_test(sub=test)
        csr_ship_sub = ship_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get shipping for sgd in test'.format(time.time() - start_time))

        add_categories_and_mix_with_condition(df=test)

        # Get categories as dummies
        cat0_man.add_factorized_feature_on_test(sub=test)
        csr_cat0_sub = cat0_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get category_name_0 for sgd in test'.format(time.time() - start_time))

        cat1_man.add_factorized_feature_on_test(sub=test)
        csr_cat1_sub = cat1_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get category_name_1 for sgd in test'.format(time.time() - start_time))

        cat2_man.add_factorized_feature_on_test(sub=test)
        csr_cat2_sub = cat2_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get category_name_2 for sgd in test'.format(time.time() - start_time))

        csr_cat_sub = hstack((csr_cat0_sub, csr_cat1_sub, csr_cat2_sub)).tocsr()
        del csr_cat0_sub, csr_cat1_sub, csr_cat2_sub
        gc.collect()

        # Get categories + condition as dummies
        cat0_cond_man.add_factorized_feature_on_test(sub=test)
        csr_cat0_cond_sub = cat0_cond_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get category_name_0 for sgd in test'.format(time.time() - start_time))

        cat1_cond_man.add_factorized_feature_on_test(sub=test)
        csr_cat1_cond_sub = cat1_cond_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get category_name_1 for sgd in test'.format(time.time() - start_time))

        cat2_cond_man.add_factorized_feature_on_test(sub=test)
        csr_cat2_cond_sub = cat2_cond_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get category_name_2 for sgd in test'.format(time.time() - start_time))

        csr_cat_cond_sub = hstack((csr_cat0_cond_sub, csr_cat1_cond_sub, csr_cat2_cond_sub)).tocsr()
        del csr_cat0_cond_sub, csr_cat1_cond_sub, csr_cat2_cond_sub
        gc.collect()

        test["brd_cond"] = test["brand_name"].astype(str) + "|" + test["item_condition_id"].astype(str)
        brd_cond_man.add_factorized_feature_on_test(sub=test)
        csr_brd_cond_sub = brd_cond_man.get_feature_for_sgd_test(sub=test)
        # print('[{}] Finished get brand_cond for sgd in train'.format(time.time() - start_time))

        csr_num_sub, _ = get_numerical_features(
            test,
            numericals=get_numerical_features_for_sgd(test),
            gaussian=True,
            minmax_skl=trn_minmax_skl,
            rank=False
        )
        # print('[{}] Finished getting numerical features for test'.format(time.time() - start_time))

        # for i_b in range(math.ceil(len(test) / batch_size)):
        # s_b, e_b = i_b * batch_size, (i_b + 1) * batch_size

        csr_ridge_sub = hstack((
            csr_cond_sub,
            csr_ship_sub,
            csr_brand_sub,
            csr_brd_cond_sub,
            csr_num_sub,
            csr_cat_sub,
            csr_cat_cond_sub,
            csr_name_sub
        )).tocsr()

        # print('[{}] Finished creating batch for test'.format(time.time() - start_time))
        sub_liblinear_preds = np.empty(len(test))
        sub_ridge_preds = np.empty(len(test))
        # Get sgd predictions for test
        if y_test is not None:
            sub_liblinear_preds, sub_ridge_preds = \
                get_sgd_test_predictions(csr_ridge_sub,
                                         folds,
                                         models_list,
                                         y_test)
        else:
            sub_liblinear_preds, sub_ridge_preds = \
                get_sgd_test_predictions(csr_ridge_sub, folds, models_list)

        # Delete csr matrices we do not need anymore
        del csr_ridge_sub
        gc.collect()

        del csr_cond_sub, csr_ship_sub
        del csr_num_sub
        del csr_brand_sub
        del csr_cat_cond_sub
        del csr_cat_sub
        gc.collect()

        # Now make sure all went well if possible
        print("Done predicting Ridge/LinearSVR")
        if y_test is not None:
            print("Test liblinear error : ", mean_squared_error(y_test, sub_liblinear_preds) ** .5)
            print("ridge error : ", mean_squared_error(y_test, sub_ridge_preds) ** .5)

        # cpuStats()
        # print('[{}] Finished delete test csr matrices'.format(time.time() - start_time))

        test["sgd_liblinear"] = np.expm1(sub_liblinear_preds)
        test["sgd_ridge"] = np.expm1(sub_ridge_preds)
        test["liblinear_ridge"] = .50 * np.expm1(sub_liblinear_preds) + .50 * np.expm1(sub_ridge_preds)

        # Reduce csr_name_sub
        csr_name_sub = csr_name_sub[:, indices_low[util_cols_low_trn]]
        csr_name_sub = csr_name_sub[:, indices_high[util_cols_high_trn]]
        gc.collect()
        # cpuStats()
        # print("Feature reduction done, X_name shape ", csr_name_sub.shape, " after col pruning")

        add_name_features_for_test(df=test, indexers=name_indexers)

        # print(test.columns)

        # cpuStats()
        # print('[{0:7.1f}] Finished create name features'.format(time.time() - start_time))

        csr_num_sub = get_numerical_features(test, numericals=get_numerical_features_for_lgb(test), gaussian=False,
                                             rank=True)
        # cpuStats()
        # print('[{}] Finished get test numerical data for Light GBM'.format(time.time() - start_time))

        name_sub = test[["name"]].copy()

        # We do not need test anymore
        del test
        gc.collect()
        # cpuStats()
        # print('[{}] Finished deleting test data'.format(time.time() - start_time))

        # Compute test matrices
        csr_tfidfname_sub = get_tfidf_features_for_test(name_sub, wordbatch_tfidf, clipping, hash_man_=main_hash_man)
        # cpuStats()
        # print('[{}] Finished get test tfidf data for Light GBM'.format(time.time() - start_time))

        if ensemble:

            predsL1 = np.empty(len(submission))
            predsL2 = np.empty(len(submission))

            csr_lgb_sub = hstack((
                csr_tfidfname_sub,
                csr_name_sub,
                csr_num_sub,
            )).tocsr()

            # print('[{}] Finished creating batch for test'.format(time.time() - start_time))

            # Get predictions for test
            predsL1 = lgb_l1.predict(csr_lgb_sub)
            predsL2 = lgb_l2.predict(csr_lgb_sub)

            # Delete csr matrices we do not need anymore
            del csr_lgb_sub
            gc.collect()

            del csr_num_sub, csr_name_sub, csr_tfidfname_sub
            gc.collect()
            print("Finished predicting LGB1 and LGB2")
            if y_test is not None:
                preds = np.expm1(predsL2) * .7 + np.expm1(predsL1) * .3
                print("Test error L1   : %.6f "
                      % mean_squared_error(y_test, predsL1) ** .5)
                print("Test error L2   : %.6f "
                      % mean_squared_error(y_test, predsL2) ** .5)
                print("Test error Mix  : %.6f "
                      % mean_squared_error(y_test, np.log1p(preds)) ** .5)

            submission_preds[s_b: e_b] = np.expm1(predsL2) * .65 + np.expm1(predsL1) * .35

        else:

            predsL2 = np.empty(len(submission))

            csr_lgb_sub = hstack((
                csr_tfidfname_sub,
                csr_name_sub,
                csr_num_sub,
            )).tocsr()

            # print('[{}] Finished creating batch for test'.format(time.time() - start_time))

            # Get predictions for test
            predsL2[s_b: e_b] = lgb_l2.predict(csr_lgb_sub)

            # Delete csr matrices we do not need anymore
            del csr_lgb_sub
            gc.collect()

            del csr_num_sub, csr_name_sub, csr_tfidfname_sub
            gc.collect()
            print("Finished predicting LGB2")
            if y_test is not None:
                print("Test error L2  : %.6f " % mean_squared_error(y_test, predsL2) ** .5)

            submission_preds[s_b: e_b] = np.expm1(predsL2)

    print("=" * 50)
    print("PREDICTION PART FINISHED")
    print("=" * 50)

    if y_test2 is not None:
        print("Full test error : %.6f " % mean_squared_error(y_test2, np.log1p(submission_preds)) ** .5)
    else:
        # Write submission
        submission["price"] = np.clip(submission_preds, 0, 100000)
        submission["test_id"] = test2["id"]
        submission[['test_id', 'price']].to_csv("ridge_lgbm_around_the_world_submission.csv", index=False)


if __name__ == '__main__':
    # enable garbage collection
    gc.enable()

    # Run main in selected mode
    main(mode=PROD, ratio=0.2)