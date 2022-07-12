# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

import pandas as pd
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.columns)
print(test_df.columns)
train_df['target'] = 1-train_df['target']
print(train_df.columns)
train_df.drop("ID_code", axis=1, inplace=True)
#y_train = pd.get_dummies(train_df, columns=['target'],dtype=np.int32)
#y_train = np.asarray(y_train.values)
y_train = train_df['target']
train_df.drop("target", axis=1, inplace=True)
train_df.drop("var_45", axis=1, inplace=True)
train_df.drop("var_68", axis=1, inplace=True)
tr_df = train_df.apply(lambda x: [y if y < 60.0 else 60.0 for y in x])
train_df = tr_df.apply(lambda x: [y if y > -60.0 else -60.0 for y in x])

x_train = train_df
test_id = test_df.ID_code
test_df.drop("ID_code", axis=1, inplace=True)
test_df.drop("var_45", axis=1, inplace=True)
test_df.drop("var_68", axis=1, inplace=True)
ts_df = test_df.apply(lambda x: [y if y < 55.0 else 55.0 for y in x])
test_df = ts_df.apply(lambda x: [y if y > -55.0 else -55.0 for y in x])

x_test = test_df
x_train = (x_train - x_train.min())/(x_train.max()-x_train.min())
x_test = (x_test - x_test.min())/(x_test.max()-x_test.min())
#x_train = x_train - x_train.mean()
#x_test = x_test - x_test.mean()
#print(x_train)
#tt=pd.DataFrame()
#print(x_train, y_train, x_test)
#for i in range(200):
#    for j in range(1):
#x_train.append(x_train.multiply(x_train['var_0'], axis='index'), True)
#print(x_train.columns)
#feature_columns = [tf.feature_column.bucketized_column(tf.feature_column.numeric_column('var_'+str(i)), (0, 1)) for i in range(200)]
numeric_columns = [tf.feature_column.numeric_column('var_'+str(i)) for i in range(200)]
print(numeric_columns.pop(45))
print(numeric_columns.pop(67))
#crossed_columns = tf.feature_column.crossed_column(['var_'+str(i) for i in range(200)], 1000)
#features_columns = [numeric_columns, crossed_columns]
'''
classifier = tf.estimator.LinearRegressor(
                feature_columns = numeric_columns,
                model_dir='/var/tmp/model_dnn',
                label_dimension=1,
                weight_column=None,
                optimizer='Adam')
'''
classifier = tf.estimator.DNNLinearCombinedClassifier(
                linear_feature_columns = numeric_columns,
                linear_optimizer='Adam',
                dnn_feature_columns = numeric_columns,
                dnn_hidden_units = [256, 256, 256, 256, 256],
                model_dir = "/var/tmp/model_dnn",
                n_classes = 2,
                dnn_activation_fn = None, #tf.nn.sigmoid,
                dnn_dropout=0.7,
                batch_norm = True,
                dnn_optimizer = #tf.train.GradientDescentOptimizer(),
                    lambda: tf.train.AdamOptimizer(
                        learning_rate=tf.train.exponential_decay(
                        learning_rate=0.001,
                        global_step=tf.train.get_global_step(),
                        decay_steps=900,
                        decay_rate=1)))

'''
classifier = tf.estimator.DNNClassifier(
                feature_columns = numeric_columns,
                hidden_units = [256, 256, 256, 256],
                model_dir = "/var/tmp/model_dnn",
                n_classes = 2,
                activation_fn = None,
                #dropout=0.2,
                batch_norm = False,
                optimizer =  #tf.train.GradientDescentOptimizer(),
                    lambda: tf.train.AdamOptimizer(
                        learning_rate=tf.train.exponential_decay(
                        learning_rate=0.0001,
                        global_step=tf.train.get_global_step(),
                        decay_steps=5000,
                        decay_rate=0.96))) 
'''                        
'''                 
### it is observed that with bucketized column, boosted trees and dnn is underperforming..

classifier = tf.estimator.BoostedTreesClassifier(
                feature_columns = feature_columns,
                model_dir = "/var/tmp/model_dnn",
                n_batches_per_layer = 2500,
                n_trees = 100,
                max_depth=12,
                learning_rate=0.01,
                l1_regularization=0.0005,
                l2_regularization=0.0005,
                tree_complexity=0.8,
                min_node_weight=0.0,
                pruning_mode='pre',
                )
'''

folds = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=31415)
for train_index, test_index in folds.split(x_train, y_train):
#    print(len(train_index), len(test_index))
    classifier.train(input_fn = tf.estimator.inputs.pandas_input_fn(
        x_train.iloc[train_index], y_train.iloc[train_index],
#        x_train, y_train,
        batch_size=100, num_epochs=1, shuffle=True), steps= 500000)
        
    rslts = classifier.evaluate(input_fn = tf.estimator.inputs.pandas_input_fn(
        x_train.iloc[test_index], y_train.iloc[test_index],
#        x_train, y_train,
        shuffle=True), steps= 100)
    print(rslts)


pred = classifier.predict(input_fn = tf.estimator.inputs.pandas_input_fn(
        x_test, batch_size=100, shuffle=False))
#predictions=list()
submission = pd.read_csv('../input/sample_submission.csv', index_col='ID_code', dtype={"target": np.float32})
for i, prd in enumerate(pred):
    #print(prd)
    [pp] = prd['logistic']
    #print(1-pp)
    #predictions.append(pp)
    submission.target[i] = 1-pp
    
submission.to_csv('submission.csv')
print(submission.head())

            
'''
def _parse_func():
    return [tf.feature_column.numeric_column(name) for name in features[:,1:]], tf.feature_column.numeric_column(featrues[:,0])

file = "../input/train.csv"
record_defaults = [tf.float32]*201
dataset = tf.data.experimental.CsvDataset(file, record_defaults, header=True, select_cols=[k+1 for k in range(201)])
dataset = dataset.map(_parse_func)
dataset = dataset.batch(100)
#tf.print(dataset)
#dataset = dataset.map(lambda x, y: (tf.math.l2_normalize(x),y))
itr  = dataset.make_one_shot_iterator()
x_train, y_train = itr.get_next()
#tf.print("data:",data)
tf.print(x_train, y_train)
'''