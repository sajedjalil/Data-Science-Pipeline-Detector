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

# Coding
import pandas as pd
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

import chainer

# Step1 データセットの準備
x = df.iloc[:, 1:-1]
t = df.iloc[:, -1]

# pandas(df)からnp(ndarray)変換
x = x.values
t = t.values

x = x.astype('float32')
t = t.astype('int32')

# 分割
from sklearn.model_selection import train_test_split
# 訓練データセットと検証データセットに分割
x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.3, random_state=0)

# Step2 ネットワークを決める(全結合3層, ReLU関数)
# リンク: パラメータを持つ関数(全結合など)
# ファンクション: パラメータを持たない関数(シグモイド関数やReLU関数)
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

n_input = 256
n_hidden = 512
n_output = 2

net = Sequential(
    # Linear: 全結合(入力数, 出力数), 非線形変換
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)



# Step3 目的関数を決める
# 交差エントロピーと決め手、ネットワークの訓練の中で実行

# Step4 最適化手法
optimizer = chainer.optimizers.SGD(lr=0.01)
optimizer.setup(net)

# Step5 ネットワークを訓練する
n_epoch = 3
n_batchsize = 256

import numpy as np

iteration = 0
# ログの保存
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}

import time

start = time.time()

# n_epoch分学習を実行
for epoch in range(n_epoch):
  
  # データセット並び替え順番を取得
  order = np.random.permutation(range(len(x_train)))
  # 各バッチ毎の目的関数の出力と分類精度の保存
  loss_list = []
  accuracy_list = []
  
  # 1 iteration(全ミニバッチ)を実行
  for i in range(0, len(order), n_batchsize):
    # バッチを準備
    index = order[i:i+n_batchsize]
    x_train_batch = x_train[index,:]
    t_train_batch = t_train[index]
    
    # 予測値を出力
    y_train_batch = net(x_train_batch)
    
    # 目的関数を適用し、分類精度を計算
#     print(t_train_batch.shape)
#     print(y_train_batch.shape)
#     print(t_train_batch)
#     print(y_train_batch)
#     print(type(t_train_batch))
#     print(type(y_train_batch))
    
    loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
    accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)
    # 結果の保存
    loss_list.append(loss_train_batch.array)
    accuracy_list.append(accuracy_train_batch.array)
    # 勾配のリセットと勾配の計算
    net.cleargrads()
    loss_train_batch.backward()
    # パラメータの更新★初期値は？
    optimizer.update()
    # カウントアップ
    iteration += 1
  # -----------------------------------------------------
  # 訓練データに対する目的関数の出力と分類精度を集計
  loss_train = np.mean(loss_list)
  accuracy_train = np.mean(accuracy_list)
  # 1エポックを終えたら、検証データを評価★わからん
  with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_val = net(x_val)
  # 目的関数を適用し、分類精度を計算(内部で計算してるのでは？)
  loss_val = F.softmax_cross_entropy(y_val, t_val)
  accuracy_val = F.accuracy(y_val,  t_val)
  # 結果の表示
  print('epoch: {}, iteration: {}, loss(train): {:.4f}, loss(valid): {:.4f}'.format(
      epoch, iteration, loss_train, loss_val.array))
  # ログを保存
  results_train['loss'].append(loss_train)
  results_train['accuracy'].append(accuracy_train)
  results_valid['loss'].append(loss_val.array)
  results_valid['accuracy'].append(accuracy_val.array)

print(time.time() - start)
# ----
# Execute
x_test = df_test.iloc[:, 1:]
# pandas(df)からnp(ndarray)変換
x_test = x_test.values
x_test = x_test.astype('float32')
# 実行
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)
y_pd = pd.DataFrame(y_test.array)
y_pd.columns = ['target', 'dummy']

result_data = pd.concat([df_test['id'], y_pd['target']], axis=1)
# result_data.head()
result_data.to_csv('submisson.csv',index=False)

#  sample = pd.read_csv("../input/sample_submission.csv")
# sample.to_csv('submisson.csv',index=False)
