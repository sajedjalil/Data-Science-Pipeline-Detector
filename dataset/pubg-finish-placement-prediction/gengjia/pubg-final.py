import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale

# One-Hot
sample = []
for i in range(0, 21):
    sample.append(i * 0.05)
sample = np.array(sample, dtype=np.float32).reshape(-1, 1) * 20
enc = OneHotEncoder()
enc.fit(sample)

# read dataset
df = pd.read_csv("../input/preprocsv/final.csv")
print("------------------read dataset finish------------------")
# label one-hot
label = np.array(df["winPlacePerc"]).reshape(-1, 1) * 20
label = enc.transform(label).toarray()
print("trans trainVector...............")
train = []
train_title = ["assists", "boosts", "damageDealt", "DBNOs", "headshotKills", "heals", "killPlace", "killPoints",
               "kills", "killStreaks", "longestKill", "revives", "rideDistance", "roadKills", "swimDistance",
               "teamKills", "vehicleDestroys", "walkDistance", "weaponsAcquired", "winPoints", "maxPlace", "numGroups"]
for index, row in df.iterrows():
    train_row = []
    for i in range(0, len(train_title)):
        train_row.append(df[train_title[i]][index])
    train.append(train_row)

print("------------------train vector finish------------------")

# split trainset
x_train = train[0:int(len(train) * 0.8)]
x_test = train[int(len(train) * 0.8):len(train)]
y_train = label[0:int(len(train) * 0.8)]
y_test = label[int(len(train) * 0.8):len(train)]

x = tf.placeholder(tf.float32, shape=(None, 22))
y = tf.placeholder(tf.float32, shape=(None, 21))
keep = tf.placeholder(tf.float32)

# layer1
var1 = tf.Variable(tf.truncated_normal([22, 512], stddev=0.1))
bias1 = tf.Variable(tf.zeros([512]))
hc1 = tf.add(tf.matmul(x, var1), bias1)
h1 = tf.nn.leaky_relu(hc1, alpha=0.1)
h1 = tf.nn.dropout(h1, keep_prob=keep)

# layer2
var2 = tf.Variable(tf.truncated_normal([512, 1024], stddev=0.1))
bias2 = tf.Variable(tf.zeros([1024]))
hc2 = tf.add(tf.matmul(h1, var2), bias2)
h2 = tf.nn.leaky_relu(hc2, alpha=0.1)
h2 = tf.nn.dropout(h2, keep_prob=keep)

# layer3
var3 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1))
bias3 = tf.Variable(tf.zeros([1024]))
hc3 = tf.add(tf.matmul(h2, var3), bias3)
h3 = tf.nn.leaky_relu(hc3, alpha=0.1)
h3 = tf.nn.dropout(h3, keep_prob=keep)

# layer4
var4 = tf.Variable(tf.truncated_normal([1024, 21], stddev=0.1))
bias4 = tf.Variable(tf.zeros([21]))
hc4 = tf.add(tf.matmul(h3, var4), bias4)
h4 = tf.nn.softmax(hc4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h4, labels=y))

optimzer = tf.train.AdamOptimizer(1e-4).minimize(loss)
epoch = 500
batch_size = 512
size = len(x_train)
batch_time = int(size / batch_size)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test.csv")
test = []

for i in range(0, len(train_title)):
    df_test[train_title[i]] = minmax_scale(df_test[train_title[i]])
    
for index, row in df_test.iterrows():
    train_row = []
    for i in range(0, len(train_title)):
        train_row.append(df_test[train_title[i]][index])
    test.append(train_row)
test = np.array(test)
print("test_vector finish")
sample = sample / 20
rets = tf.matmul(h4, sample)
print("train start...........................")
with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(0, epoch):
        for j in range(0, batch_time - 1):
            sess.run(optimzer, feed_dict={x: x_train[j * batch_size: j * batch_size + batch_size],
                                          y: y_train[j * batch_size:j * batch_size + batch_size], keep: 0.5})
        if i % 10 == 0:
            print("epoch" + str(i))
            los = sess.run(loss,feed_dict={x: x_train[j * batch_size: j * batch_size + batch_size],
                                          y: y_train[j * batch_size:j * batch_size + batch_size], keep: 1})
            print("loss:"+str(loss))
            
    print("train end...........................")
    test_batch = 1000
    times = int(len(df_test["Id"]) / test_batch)
    print("test start..........................")
    pre_res = np.array([])
    for i in range(0,times):
        pre = sess.run(rets, feed_dict={x: test[i*test_batch:i*test_batch+test_batch], keep: 1})
        pre = np.array(pre).reshape(-1)
        pre_res = np.append(pre_res,pre)
        if i == times-1 and times*test_batch != len(df_test["Id"]):
            pre = sess.run(rets, feed_dict={x: test[times*test_batch:len(df_test["Id"])], keep: 1})
            pre = np.array(pre).reshape(-1)
            print(pre)
            pre_res = np.append(pre_res,pre)
    print("test end..........................")
    result = {'Id': df_test["Id"],
              'winPlacePerc': pre_res}
    print(len(df_test["Id"]))
    print(len(pre_res))
    res = pd.DataFrame(result)
    print(pre_res)
    res.to_csv('submission.csv', index=None)