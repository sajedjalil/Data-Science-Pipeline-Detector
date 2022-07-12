# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import errno
import shutil
import tempfile

import pandas as pd
import tensorflow as tf

#Wide and Deep Classifier implemented in Tensorflow - 4 Categories with Acc ~ 82.5%
tf.set_random_seed(42)
tf.reset_default_graph()

train_file = '../input/train.csv'

test_file = '../input/test.csv'
sample_submission_file = '../input/sample_submission.csv'

df_train = pd.read_csv(train_file)

df_test = pd.read_csv(test_file)
print(len(df_train))
cols = df_train.columns.values
CATEGORICAL_COLUMNS = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
LABEL_COLUMN = 'y'
ID_COL = 'ID'
other_cols = CATEGORICAL_COLUMNS + ['ID', 'y']
print('CATEGORICAL_COLUMNS')
print(CATEGORICAL_COLUMNS)
CONTINUOUS_COLUMNS = [col for col in cols if col not in other_cols]
print('CONTINUOUS_COLUMNS')
print(CONTINUOUS_COLUMNS)
LABEL_COLUMN_BINARY = 'binary'


def binarize(row):
    if row[LABEL_COLUMN] <= 80:
        return 1
    elif row[LABEL_COLUMN] <= 97:
        return 2
    elif row[LABEL_COLUMN] <= 104:
        return 3
    return 0


df_train[LABEL_COLUMN_BINARY] = df_train.apply(lambda row: binarize(row), axis=1)
print(df_train[LABEL_COLUMN_BINARY])

def input_fn(df):
    feature_cols = get_feature_columns(df)
    label = tf.constant(df[LABEL_COLUMN_BINARY].values)
    return feature_cols, label


def get_feature_columns(df):
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                       for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    return feature_cols

def train_input_fn(start, stop):
    return input_fn(df_train[start:stop])

def eval_input_fn(start, stop):
    return input_fn(df_train[start:stop])

def test_input_fn():
    return get_feature_columns(df_test)


BASE_CATEGORICAL_FEATURE_COLUMNS = [tf.contrib.layers.sparse_column_with_hash_bucket(x, hash_bucket_size=10)
                                    for x in CATEGORICAL_COLUMNS]
BASE_CONTINUES_FEATURE_COLUMNS = [tf.contrib.layers.real_valued_column(x)
                                  for x in CONTINUOUS_COLUMNS]
BASE_FEATURES = BASE_CATEGORICAL_FEATURE_COLUMNS + BASE_CONTINUES_FEATURE_COLUMNS
CROSS_FEATURES1 = [tf.contrib.layers.crossed_column(BASE_CATEGORICAL_FEATURE_COLUMNS, hash_bucket_size=int(1e8))]
CROSS_FEATURES2 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[1]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES3 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[2]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES4 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[3]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES5 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[4]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES6 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[5]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES7 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[6]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES8 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[0], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES10 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[1], BASE_CATEGORICAL_FEATURE_COLUMNS[2]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES11 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[1], BASE_CATEGORICAL_FEATURE_COLUMNS[3]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES12 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[1], BASE_CATEGORICAL_FEATURE_COLUMNS[4]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES13 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[1], BASE_CATEGORICAL_FEATURE_COLUMNS[5]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES14 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[1], BASE_CATEGORICAL_FEATURE_COLUMNS[6]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES15 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[1], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES18 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[2], BASE_CATEGORICAL_FEATURE_COLUMNS[3]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES19 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[2], BASE_CATEGORICAL_FEATURE_COLUMNS[4]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES20 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[2], BASE_CATEGORICAL_FEATURE_COLUMNS[5]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES21 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[2], BASE_CATEGORICAL_FEATURE_COLUMNS[6]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES22 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[2], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES26 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[3], BASE_CATEGORICAL_FEATURE_COLUMNS[4]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES27 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[3], BASE_CATEGORICAL_FEATURE_COLUMNS[5]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES28 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[3], BASE_CATEGORICAL_FEATURE_COLUMNS[6]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES29 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[3], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES34 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[4], BASE_CATEGORICAL_FEATURE_COLUMNS[5]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES35 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[4], BASE_CATEGORICAL_FEATURE_COLUMNS[6]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES36 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[4], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES37 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[5], BASE_CATEGORICAL_FEATURE_COLUMNS[6]],
                                     hash_bucket_size=int(1e4))]
CROSS_FEATURES38 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[5], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES39 = [
    tf.contrib.layers.crossed_column([BASE_CATEGORICAL_FEATURE_COLUMNS[6], BASE_CATEGORICAL_FEATURE_COLUMNS[7]],
                                     hash_bucket_size=int(1e4))]

CROSS_FEATURES = CROSS_FEATURES1 + CROSS_FEATURES2 + CROSS_FEATURES3 + CROSS_FEATURES4 + CROSS_FEATURES5 \
                 + CROSS_FEATURES6 + CROSS_FEATURES7 + CROSS_FEATURES8 + CROSS_FEATURES10 \
                 + CROSS_FEATURES11 + CROSS_FEATURES12 + CROSS_FEATURES13 + CROSS_FEATURES14 + CROSS_FEATURES15 \
                 + CROSS_FEATURES18 + CROSS_FEATURES19 + CROSS_FEATURES20 + CROSS_FEATURES21 + CROSS_FEATURES22 \
                 + CROSS_FEATURES26 + CROSS_FEATURES27 + CROSS_FEATURES28 + CROSS_FEATURES29 + CROSS_FEATURES34 \
                 + CROSS_FEATURES35 + CROSS_FEATURES36 + CROSS_FEATURES37 + CROSS_FEATURES38 + CROSS_FEATURES39

print('Number of cross features')
print(len(CROSS_FEATURES))
feature_columns_wide = BASE_FEATURES + CROSS_FEATURES
DEEP_CATEGORICAL_FEATURE_COLUMNS = [tf.contrib.layers.embedding_column(x, dimension=8)
                                    for x in BASE_CATEGORICAL_FEATURE_COLUMNS]
DEEP_CATEGORICAL_CROSS_FEATURE_COLUMNS = [tf.contrib.layers.embedding_column(x, dimension=8)
                                    for x in CROSS_FEATURES]
feature_columns_deep = DEEP_CATEGORICAL_FEATURE_COLUMNS + BASE_CONTINUES_FEATURE_COLUMNS \
                       + DEEP_CATEGORICAL_CROSS_FEATURE_COLUMNS
print('Number of wide features:')
print(len(feature_columns_wide))
print('Number of deep features')
print(len(feature_columns_deep))

def get_model():
    return tf.contrib.learn.DNNLinearCombinedClassifier(
    n_classes=4,
    enable_centered_bias=True,
    model_dir=model_dir,
    linear_feature_columns=feature_columns_wide,
    linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.001),
    dnn_feature_columns=feature_columns_deep,
    dnn_hidden_units=[200, 50],
    dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.1),
    fix_global_step_increment_bug=True)

def save_validation_results(results, filename):
    file = open(filename, 'w')
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
        file.write("%s: %s\n" % (key, results[key]))
    file.close()

def fit_and_validate(m, start_train, stop_train, start_test, stop_test, filename):
    m.fit(input_fn=lambda: train_input_fn(start_train, stop_train), steps=200)
    results = m.evaluate(input_fn=lambda: eval_input_fn(start_test, stop_test), steps=1)
    save_validation_results(results, filename)
try:
    model_dir = tempfile.mkdtemp()
    model = get_model()
    # model.fit(input_fn=lambda: train_input_fn(0, 4208), steps=200)
    fit_and_validate(model, 0, 2800, 2801, 4208, 'validation.csv')
    predictions = model.predict(input_fn=test_input_fn)
    df_pred = pd.DataFrame(predictions)
    sample = pd.read_csv(sample_submission_file)
    sample['y'] = df_pred.values
    sample.to_csv('results.csv')
finally:
    try:
        shutil.rmtree(model_dir)  # delete directory
    except OSError as exc:
        if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
            raise  # re-raise exception