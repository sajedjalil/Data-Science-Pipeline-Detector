# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gc
import os
print(os.listdir("../input"))

'''
# Any results you write to the current directory are saved as output.
# Import data
sales = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/shops.csv')
items = pd.read_csv('../input/items.csv')
cats = pd.read_csv('../input/item_categories.csv')
val = pd.read_csv('../input/test.csv')

# Rearrange the raw data to be monthly sales by item-shop
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
print(df)
df["item_cnt_day"].clip(0.,20.,inplace=True)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
#print("df:", df)
# Merge data from monthly sales to specific item-shops in test data
test = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

# Strip categorical data so keras only sees raw timeseries
test = test.drop(labels=['ID','item_id','shop_id'],axis=1)

# Rearrange the raw data to be monthly average price by item-shop
# Scale Price
#scaler = MinMaxScaler(feature_range=(0, 1))
#sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

# Merge data from average prices to specific item-shops in test data
price = pd.merge(val,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

# Create x and y training sets from oldest data points
y_train = test['2015-10']
x_sales = test #.drop(labels=['2015-10'],axis=1)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
#x_prices = price #.drop(labels=['2015-10'],axis=1)
#x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
#X = np.append(x_sales,x_prices,axis=2)
X=x_sales

Y = y_train.values.reshape((214200, 1))
print("Training Predictor Shape: ",X.shape) ##214200, 34,2
print("Training Predictee Shape: ",Y.shape) ## 214200, 1
'''

import seaborn as sns

default_path = '../input/'
train_df = pd.read_csv(default_path+'sales_train.csv')
items_df = pd.read_csv(default_path+'items.csv')
test_df = pd.read_csv(default_path+'test.csv')
print(train_df.shape, test_df.shape)

train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
dataset = train_df.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)
dataset = dataset.reset_index()
dataset = pd.merge(test_df, dataset, on=['item_id', 'shop_id'], how='left')
dataset = dataset.fillna(0)
dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]

X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)

XX = X_train
YY = y_train
TEST = X_test
TIME_STEP=1
SEQ_LEN = 33

ffss=list()
llbbs=list()
test=list()
for j in range(214200):
    fs=list()
    lbs=list()
    test_fs=list()
    xx = X[j,:,:]
    for i in range(TIME_STEP):
        fs.append(xx[i:i+SEQ_LEN,:])
        lbs.append(xx[SEQ_LEN+i,:])
        test_fs.append(xx[i+1:i+1+SEQ_LEN,:])
    ffss.append(fs)
    llbbs.append(lbs)
    test.append(test_fs)
XX = np.reshape(np.asarray(ffss), (214200, TIME_STEP, SEQ_LEN))
YY = np.asarray(llbbs)
TEST = np.reshape(np.asarray(test), (214200, TIME_STEP, SEQ_LEN))
print(XX.shape, YY.shape, TEST.shape)

#XX = np.reshape(XX, (214200, 33))
#TEST = np.reshape(TEST, (214200, 33))
import tensorflow as tf
#x = tf.placeholder(tf.float32, [None, 2])
#y = tf.placeholder(tf.float32, [None, 1])
#X = np.reshape(X, (33,214200,2))
BATCH_SIZE=100
def rnn(x):
    lstm = tf.nn.rnn_cell.LSTMCell(64, activation=None) #, state_is_tuple=True, reuse=tf.AUTO_REUSE) #, activation=tf.nn.relu)
    #lstm1 = tf.nn.rnn_cell.LSTMCell(2,state_is_tuple=True, reuse=tf.AUTO_REUSE)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm)
    #initial_state = state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)
    print(x)
    #x= tf.reshape(x, [-1,1,33])
    x= tf.reshape(x, [-1,TIME_STEP,SEQ_LEN])
    #for i in range(28):
        #print(x.shape)
    #    x1 = tf.reshape(x[i,:], [1,5,2])
    output, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x, dtype=tf.float32, sequence_length=[SEQ_LEN for _ in range(BATCH_SIZE)])
    f_state = state
    print("output:", output)
    tf.print("output:", output)
    output_ = tf.reshape(output, (-1,TIME_STEP*16))
    output_ = tf.nn.dropout(output_, 0.5)
    output_ = tf.layers.dense(inputs=output_, units=TIME_STEP*16, activation=None)
    output_ = tf.nn.dropout(output_, 0.5)
    
    #output_ = tf.layers.dense(inputs=output_, units=TIME_STEP*8, activation=tf.nn.relu)
    dense_out = tf.layers.dense(inputs=output_, units=TIME_STEP, activation=tf.nn.relu)
    return dense_out
    
def model_fn(features, labels, mode):
    y = rnn(features)
    y = tf.reshape(y, [BATCH_SIZE,TIME_STEP])
    train_op = None
    loss = tf.convert_to_tensor(0.)
    predictions = None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):
        labels=  tf.reshape(labels, [BATCH_SIZE,TIME_STEP])
        loss = tf.losses.mean_squared_error(labels, y) + tf.losses.get_regularization_loss()
    if(mode == tf.estimator.ModeKeys.TRAIN):
        lr = tf.train.exponential_decay(0.001, global_step, 1000, 0.99000, staircase=False)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)
    if(mode == tf.estimator.ModeKeys.PREDICT):
    	predictions = {"predictions":y} #[:,-1]} # tf.slice(y, [0,27],[100,1])}
    if(mode == tf.estimator.ModeKeys.EVAL):
        eval_metric_ops = {"mae error": tf.metrics.mean_squared_error(labels,y)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
    				train_op=train_op, predictions=predictions, eval_metric_ops = eval_metric_ops)

est = tf.estimator.Estimator(model_fn=model_fn, model_dir="/var/tmp/model_dir")	

for _ in range(2):
    est.train(input_fn=tf.estimator.inputs.numpy_input_fn(
        #dict({"features":X}), Y,
        XX.astype(np.float32), YY.astype(np.float32),
        batch_size=BATCH_SIZE, num_epochs=5, shuffle=True), steps= None)
    est.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(
        XX.astype(np.float32), YY.astype(np.float32),
        batch_size=BATCH_SIZE, num_epochs=1, shuffle=True), steps= 100)

    pred=est.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
        TEST.astype(np.float32),
        batch_size=BATCH_SIZE, num_epochs=1, shuffle=False))
#print(next(pred))
    for i,p in enumerate(pred):
        pp=p["predictions"]
        print(p,pp)
        if (i==100): break
    
# Transform test set into numpy matrix
#test = test.drop(labels=['2013-01'],axis=1)
#x_test_sales = test.values.reshape((test.shape[0], test.shape[1], 1))
#x_test_prices = price #.drop(labels=['2013-01'],axis=1)
#x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))
#print(test.shape)
# Combine Price and Sales Df
#test = np.append(x_test_sales,x_test_prices,axis=2)
#print(test.shape)

BATCH_SIZE=100
pred=est.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
        TEST.astype(np.float32),
        batch_size=BATCH_SIZE, num_epochs=1, shuffle=False))
predictions=list()
for i,p in enumerate(pred):
    pp = p["predictions"]
    print(p,np.round(pp*10000)/10000)
    predictions.append(np.round(pp*10000)/10000)
#print(len(predictions))
#print(predictions)
submission = pd.DataFrame(predictions,columns=['item_cnt_month'])
submission.to_csv('submission.csv',index_label='ID')
print(submission.head())

'''
for i in range(10):
    xx,ddd = rnn(XX[i,:,:])
    print(xx)
    tf.print(xx)
'''
'''
import tensorflow as tf
tf.enable_eager_execution()
record_defaults = [tf.string, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32]
dataset = tf.data.experimental.CsvDataset("sales_train.csv", record_defaults=record_defaults)
dataset = dataset.batch(10)
print(dataset)
'''
'''    
loss = 0.0
with tf.Session() as sess:
    sess.run(init_op)
    for j in range(500):
        [lss]= sess.run([cost], feed_dict={x:X[j,:,:], y:Y[j].reshape(1,1).astype(np.float32)})
        loss += lss
        print('loss=',loss)
'''