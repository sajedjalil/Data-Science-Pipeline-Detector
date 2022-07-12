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

# coding:utf-8
import tensorflow as tf
import numpy as np
import math
import sys, csv, h5py
import pandas as pd

from scipy.sparse import coo_matrix, hstack
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# pandasのデータフレームを返す
# train_or_testには'train'か'test'を入れる
def load_data(path,train_or_test,brand_threshold = 100,category_threshold = 50,frequent_brands=None,frequent_categories=None):
    data_pd = pd.read_csv(path, error_bad_lines=False, encoding='utf-8', header=0, delimiter='\t')
    #ブランド名がないものを'NO_BRAND'とする
    data_pd['brand_name'] = data_pd['brand_name'].fillna('NO_BRAND')
    data_pd=data_pd.fillna("")

    if train_or_test == 'train':
        frequent_brands = data_pd['brand_name'].value_counts()[data_pd['brand_name'].value_counts()>brand_threshold].index
        frequent_categories = data_pd['category_name'].value_counts()[data_pd['category_name'].value_counts()>category_threshold].index
    elif train_or_test != 'test':
        print('Error : Please input "train" or "test" in train_or_test')
        return
    
    if type(frequent_brands)==type(None) or type(frequent_categories)==type(None):
        print('Error : Please load train data first')
        return
    else:
        data_pd.loc[~data_pd['brand_name'].isin(frequent_brands),'brand_name']= 'SOME_BRAND'
        data_pd.loc[~data_pd['category_name'].isin(frequent_categories),'category_name'] = 'SOME_CATEGORY'
        
    return data_pd,frequent_brands,frequent_categories
 
 
# データ取得   
csv_train_path = u'../input/mercari-price-suggestion-challenge/train.tsv'
csv_test_path = u'../input/mercari-price-suggestion-challenge/test.tsv'
train_data_pd, frequent_brands, frequent_categories = load_data(csv_train_path,'train',brand_threshold=100,category_threshold=50)
test_data_pd, _, _ = load_data(csv_test_path,'test',frequent_brands=frequent_brands,frequent_categories=frequent_categories)
print('loading data completed')

use_cols = ['item_condition_id','brand_name','shipping','category_name']
train_num = len(train_data_pd)
test_num = len(test_data_pd)

# price_logの正規化
ms = MinMaxScaler()
prices = np.array(train_data_pd['price'])
prices_log = np.log(prices+1)
prices_log = ms.fit_transform(prices_log.reshape(-1, 1)).reshape(-1)

# scipyのsparse matrix(coo_matrix)X_transform と 変数のリストvariables を返す
# save_pathに何も指定しない場合ファイルを保存しない 指定した場合指定したディレクトリ内に保存する
def make_onehot(use_cols,data_pd,train_or_test,save_path=None):
    variables = []
    flag = 0
    for use_col in use_cols:
        dummy_pd = pd.get_dummies(data_pd[use_col]).astype(np.uint8)
        if flag==0:
            X_transform = coo_matrix(dummy_pd.values)
            flag=1
        else:
            X_transform = hstack([X_transform,coo_matrix(dummy_pd.values)])
        
        variables.extend( list( dummy_pd.columns ) )
        
        if save_path is not None:
            if train_or_test != 'test' and train_or_test != 'train':
                print('Error : Please input "train" or "test" in train_or_test')
                return
            save_path_ = '{}/{}_{}.csv'.format(save_path,use_col,train_or_test)
            dummy_pd.to_csv(save_path_,index=False,encoding="utf8")
            
    if save_path is not None:
        # sparse matrixの保存
        io.savemat("{}/X_transform_{}".format(save_path,train_or_test), {"X_transform":X_transform})
        print('sparse matrixを保存しました。次回からはsparse matrixを読み込んで学習に利用してください')

    return X_transform,np.array(variables)

X_transform_train,variables = make_onehot(use_cols,train_data_pd,'train',save_path=None)
X_transform_test,variables_ = make_onehot(use_cols,test_data_pd,'test',save_path=None)
print('converting data completed')


import copy
from nltk import stem

def normalize(df):
    try:
        df['item_description'] = df['item_description'].apply(lambda x: x.lower())
        df['item_description'] = df['item_description'].apply(lambda x: x.replace(".", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace(")", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("(", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("*", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace(":", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace(",", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("/", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("#", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("\\", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("1", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("2", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("3", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("4", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("5", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("6", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("7", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("8", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("9", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("0", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("!", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("$", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("%", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("&", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("-", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("+", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace(";", ""))
        df['item_description'] = df['item_description'].apply(lambda x: x.replace("[rm]", ""))
        df['item_description'] = df['item_description'].apply(lambda x: str(x))
    except:
        print("There is no attribute named 'item_description'.")
        df['item_description'] = df['item_description'].apply(lambda x: str(x))

    finally:
        return df


def clean_up(df):
    with open("../data/long_stop_word_list.txt", "r") as file:
        stop_words = file.readlines()
        words = list(map((lambda x: x.replace("\n","")), stop_words))
        stemmer = stem.LancasterStemmer()

    for index, sentence in df.iteritems():
        splitted = sentence.split()
        splitted = list(map(lambda x: stemmer.stem(x), splitted))
        copy_list = copy.copy(splitted)
        for splitted_word in splitted:
            if splitted_word in words:
                copy_list.remove(splitted_word)
            else:
                pass

        df[index] = copy_list

    return df


def frequency(df, price1, price2):
    histogram = dict()
    source = df['item_description'][df['price']<price1]
    source = source[df['price']>price2]
    for splitted in source:
        for word in splitted:
            if word in histogram.keys():
                histogram[word] += 1
            else:
                histogram[word] = 1

    return histogram

def cleaning(df):
    try:
        df = df.apply(lambda x: x.lower())
        df = df.apply(lambda x: x.replace(".", ""))
        df = df.apply(lambda x: x.replace(")", ""))
        df = df.apply(lambda x: x.replace("(", ""))
        df = df.apply(lambda x: x.replace("*", ""))
        df = df.apply(lambda x: x.replace(":", ""))
        df = df.apply(lambda x: x.replace(",", ""))
        df = df.apply(lambda x: x.replace("/", ""))
        df = df.apply(lambda x: x.replace("#", ""))
        df = df.apply(lambda x: x.replace("\\", ""))
        df = df.apply(lambda x: x.replace("1", ""))
        df = df.apply(lambda x: x.replace("2", ""))
        df = df.apply(lambda x: x.replace("3", ""))
        df = df.apply(lambda x: x.replace("4", ""))
        df = df.apply(lambda x: x.replace("5", ""))
        df = df.apply(lambda x: x.replace("6", ""))
        df = df.apply(lambda x: x.replace("7", ""))
        df = df.apply(lambda x: x.replace("8", ""))
        df = df.apply(lambda x: x.replace("9", ""))
        df = df.apply(lambda x: x.replace("0", ""))
        df = df.apply(lambda x: x.replace("!", ""))
        df = df.apply(lambda x: x.replace("$", ""))
        df = df.apply(lambda x: x.replace("%", ""))
        df = df.apply(lambda x: x.replace("&", ""))
        df = df.apply(lambda x: x.replace("-", ""))
        df = df.apply(lambda x: x.replace("+", ""))
        df = df.apply(lambda x: x.replace(";", ""))
        df = df.apply(lambda x: x.replace("[rm]", ""))
        df = df.apply(lambda x: x.replace("?", ""))
        df = df.apply(lambda x: x.replace("~", ""))
        df = df.apply(lambda x: x.replace("\"", ""))
        df = df.apply(lambda x: str(x))
    except:
        print("There is no attribute named 'item_description'.")
        df = df.apply(lambda x: str(x))

    finally:
        return df


def get_word_feature_mat(df, save_path):
    """
    使い方：
    適切なディレクトリにfeatures.csvを配置してその相対ディレクトリとスクリプト
    内に記述したパスが一致することを確認してください。
    同一ファイル内またはスコープ内にcleaning()関数が配置されているようにしてください。
    また、この関数は同一ディレクトリ内に.matファイルを生成します。
    .matファイルの保管場所を指定したい場合はio.savemat()関数内のパスを変更してください。
    データの保管をしたくない場合はio.savemat()関数をコメントアウトしてください。

    引数: DataFrameオブジェクト(Seriesオブジェクトは非対応です)
    戻り値: Sparse化した単語特徴行列

    item_description以外の特徴から生成したsparseマトリクスとhstackで結合させて分類器に
    投げ込めば上手く行くはずです。
    """
    #  必要ライブラリのimport(行儀は良くありませんが笑)
    import csv
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy import io

    #  item_descriptionの切り出し
    try:
        descr = df['item_description']
    except KeyError:
        print("キー['item_description']は存在しないようです。")
        print("['item_description']をキーとして持つデータフレームを引数に入れてください。")
        return

    #  大文字を小文字にし、記号等を消す
    try:
        descr = cleaning(descr)
    except NameError:
        print("関数cleaning()が未定義のようです。")
        print("cleaning()をスコープ内に定義してください。")
        print("参照 => useful_functions.cleaning()")
        return

    #  descriptionを単語分割、保存用の辞書作成
    splitted = descr.apply(lambda x: x.split())
    worddict = dict()

    #  各特徴語ごとに、データ内にその単語があるかないかのlistを値に持つ辞書を作成
    with open("../input/features-csv/features.csv", mode="r") as f:
        reader = csv.reader(f)
        for line in reader:
            for word in line:
                num_list = []
                for sentence in splitted:
                    if word in sentence:
                        num_list.append(1)
                    else:
                        num_list.append(0)

                worddict[word] = num_list

    #  numpyのmatrix形式にする
    word_feature_df = pd.DataFrame.from_dict(worddict)
    word_mat = word_feature_df.values

    #  sparse化
    sparsed_word_mat = coo_matrix(word_mat)
    io.savemat(save_path, {"sparsed_word_mat": sparsed_word_mat})
    print("Sparse行列を保存しました。")

    return sparsed_word_mat



train_path = "sparsed_word_feature_train"
test_path = "sparsed_word_feature_test"
train_sparsed = get_word_feature_mat(train_data_pd, save_path=train_path)
test_sparsed = get_word_feature_mat(test_data_pd, save_path=test_path)

X_transform_train = hstack([X_transform_train,train_sparsed]).tocsr()
X_transform_test = hstack([X_transform_test,test_sparsed]).tocsr()
# print('X_transform_train[1:3]')
# print(X_transform_train[1:3])
# print('X_transform_test[1:3]')
# print(X_transform_test[1:3])


train_X, test_X, train_y, test_y = train_test_split(X_transform_train, prices_log, test_size=0.1, random_state=42)
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)
original_valid_y = ms.inverse_transform(valid_y.reshape(-1,1)).reshape(-1)
print('train_X[1:3]')
print(train_X[1:3])
# train_X = train_X[:max_size]
# train_y = train_y[:max_size]
# valid_max_size = max_size // 4
# test_X = test_X[:valid_max_size]
# test_y = test_y[:valid_max_size]

MAX_EPOCH = 15
BATCH_SIZE = 32
UNIT_NOS = [100]
features = train_X.shape[1]
patience = 3
decay = 1e-4  # 正則化の係数

hidden_no = len(UNIT_NOS)
x = tf.placeholder(tf.float32, [None, features])
lr = tf.placeholder(tf.float32, [1])
W_list = []
b_list = []

old_unit_no = features
z = x
for i, unit_no in enumerate(UNIT_NOS):
    W = tf.Variable(tf.random_normal([old_unit_no, unit_no], mean=0.0, stddev=0.05))
    b = tf.Variable(tf.constant(0.1, shape=[unit_no]))
    W_list.append(W)
    b_list.append(b)
    z = tf.nn.relu(tf.matmul(z, W) + b)
    old_unit_no = unit_no
W_last = tf.Variable(tf.random_normal([UNIT_NOS[-1], 1], mean=0.0, stddev=0.05))
b_last = tf.Variable(tf.random_normal([1], mean=0.0, stddev=0.05))
W_list.append(W_last)
b_list.append(b_last)
y = tf.matmul(z, W_last) + b_last

y_ = tf.placeholder(tf.float32, [None, 1])
mse = tf.reduce_mean((y - y_) * (y - y_))

for W_,b_ in zip(W_list,b_list):
    mse += decay * (tf.nn.l2_loss(W_) + tf.nn.l2_loss(b_))

train_step = tf.train.AdamOptimizer(lr[0]).minimize(mse)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

count = 0

print('data_length:'+str(train_X.shape[0]))
batch_no = int( (train_X.shape[0] - 1) / BATCH_SIZE + 1)
batch_no_test = int( (test_X.shape[0] - 1) / BATCH_SIZE + 1)
batch_no_final = int( (X_transform_test.shape[0] - 1) / BATCH_SIZE + 1)
# print('batch_no:'+str(batch_no))

min_cost = np.inf
learning_rate = np.array([1e-2])
train_costs = []
valid_costs = []
valid_scores = []

for i in range(MAX_EPOCH):
    print("epoch:"+str(i+1))

    # SGDを実装している
    for j in range(batch_no):
        batch_xs = (train_X[j * BATCH_SIZE : min((j + 1) * BATCH_SIZE , train_X.shape[0])]).toarray()
        batch_ys = train_y[j * BATCH_SIZE:min((j + 1) * BATCH_SIZE , train_X.shape[0])].reshape(-1, 1)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,lr: learning_rate})
        if j%5000==0:
            print('   batch:'+str(j))
    train_cost = sess.run(mse, feed_dict={x: (train_X[:BATCH_SIZE]).toarray(), y_: train_y[:BATCH_SIZE].reshape(-1, 1)})
    [valid_cost,prediction_log] = sess.run([mse,y], feed_dict={x: valid_X.toarray(), y_: valid_y.reshape(-1, 1)})
    prediction_log = ms.inverse_transform(prediction_log).reshape(-1)
    valid_score = np.sqrt( np.mean((prediction_log-original_valid_y)*(prediction_log-original_valid_y)) )
    print (str(i + 1) + " epoch:train cost=" + str(train_cost) + ", valid cost=" + str(valid_cost) )
    print("valid score="+str(valid_score))
    train_costs.append(train_cost)
    valid_costs.append(valid_cost)
    valid_scores.append(valid_score)

    if valid_cost < min_cost:
        count = 0
        # for i, W, b in zip(range(hidden_no), W_list, b_list):
        #     np.savetxt(path + "W" + str(i + 1) + ".csv", sess.run(W), delimiter=",")
        #     np.savetxt(path + "b" + str(i + 1) + ".csv", sess.run(b), delimiter=",")
        # np.savetxt(path + "W.csv", sess.run(W_last), delimiter=",")
        # np.savetxt(path + "b.csv", sess.run(b_last), delimiter=",")
        min_cost = valid_cost
    else:
        count += 1
        learning_rate /= 5
        print('reduced learning rate : '+str(learning_rate))

    # 改善されなかった回数がpatience回以上で学習終了
    if count >= patience:
        break
    
    
prediction_logs = np.array([])
for j in range(batch_no_test):
    batch_xs = (test_X[j * BATCH_SIZE:min((j + 1) * BATCH_SIZE , test_X.shape[0])]).toarray()
    batch_ys = test_y[j * BATCH_SIZE:min((j + 1) * BATCH_SIZE , test_X.shape[0])].reshape(-1, 1)
    [test_cost,prediction_log] = sess.run([mse,y], feed_dict={x: batch_xs, y_: batch_ys})
    prediction_logs = np.append(prediction_logs, ms.inverse_transform(prediction_log).reshape(-1))

# [test_cost,prediction_log] = sess.run([mse,y], feed_dict={x: test_X.toarray(), y_: test_y.reshape(-1, 1)})
# prediction_log = ms.inverse_transform(prediction_log).reshape(-1)
test_y = ms.inverse_transform(test_y.reshape(-1,1)).reshape(-1)
test_score = np.sqrt( np.mean((prediction_logs-test_y)*(prediction_logs-test_y)) )
print ("test cost=" + str(test_cost))
print("test score="+str(test_score))


prediction_logs_final = np.array([])
for j in range(batch_no_final):
    batch_xs = (X_transform_test[j * BATCH_SIZE  : min((j + 1) * BATCH_SIZE , X_transform_test.shape[0])]).toarray()
    prediction_log_final = sess.run(y, feed_dict={x: batch_xs})
    prediction_logs_final = np.append(prediction_logs_final, (np.exp(ms.inverse_transform(prediction_log_final.reshape(-1,1)))-1).reshape(-1))

# prediction_log = sess.run(y,feed_dict={x:X_transform_test.toarray()}).reshape(-1)
# prediction =  (np.exp(ms.inverse_transform(prediction_log.reshape(-1,1)))-1).reshape(-1)
test_id = np.arange(prediction_logs_final.shape[0])

submission = pd.DataFrame([])
submission['test_id'] = test_id
submission['price'] = pd.DataFrame(prediction_logs_final)
submission.to_csv('submission.csv',index=None)
print(submission.iloc[:10])