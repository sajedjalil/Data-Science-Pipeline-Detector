import numpy as np
np.random.seed(7)

import kagglegym
import pandas as pd
import tensorflow as tf
import gc
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import time
import sys
import io
import itertools

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Highway, GaussianNoise
from keras.optimizers import Adam, RMSprop, SGD, Nadam

hist_length = 600
s_length = 2
corr_drop = 40
pca_components = 0.75
learning_rate = 0.5
learning_rate_momentum = 0.9
learning_rate_decay = 0.00001
batch_size = 600000
nb_epoch = 20

quantile_delete = 0.001
quantile_mean = 0.01
sample_min = 300000

total_time = time.time()
print(time.asctime(time.localtime(time.time())))
tf.logging.set_verbosity(tf.logging.ERROR)
f_null = open('/dev/null', 'w')
import warnings
warnings.filterwarnings("ignore")
stdout = sys.stdout

env = kagglegym.make()
o = env.reset()
o_cols = [c for c in o.train.columns if not c in ['id', 'timestamp', 'y', 'sample']]
y_cols = ['y']


df = o.train
df = df[~(df.y > 0.093497) & ~(df.y < -0.086093)]
df = df[df.timestamp > df.timestamp.unique()[-s_length - hist_length]]
df.loc[:, 't'] = df.timestamp
o_cols = o_cols + ['t']
print('y', 'mean', df.y.mean(), 'std', df.y.std())

mean = {}
for c in o_cols:
    mean[c] = df[c].mean()
    df.loc[df[c].apply(np.isnan), c] = mean[c]
mean['y_'] = df.y.mean()
instruments = df.id.unique()
timestamps = df.timestamp.unique()

print(df.shape[0], 'samples')
print(len(o_cols), 'features')
print(len(instruments), 'instruments')
print(len(timestamps), 'timestamps')

# Remove outliers
for c in o_cols:
    df = df[df[c].between(df[c].quantile(quantile_delete), df[c].quantile(1 - quantile_delete))]
print('{} quantile outliers removed, {} samples left'.format(quantile_delete, len(df)))
sample_size = min(len(df), sample_min)
quantile = {}
for c in o_cols:
    s = df[c]
    quantile[c] = (s.quantile(quantile_mean), s.quantile(1 - quantile_mean))
    is_p = s.between(quantile[c][0], quantile[c][1])
    df.loc[~is_p, c] = df.loc[is_p, c].mean()
quantile['y_'] = (-1, 1)
print(quantile_mean, 'quantile outliers changed to mean')

# Box Cox
'''
bc_min = {}
bc_range = {}
bc_mult = 0.1
bc_range_mult = 3.
bc_lambda = {}
for c in o_cols:
    bc_min[c] = df[c].min()
    bc_range[c] = df[c].max() - bc_min[c]
    #print(c, bc_min[c], bc_range[c], len(df[c]), len(df[c].unique()))
    df[c], bc_lambda[c] = boxcox((df[c] - bc_min[c] + bc_range_mult * bc_range[c]) * bc_mult)
print('Box Cox done')
'''

df['y_'] = 0
for id in df.id.unique():
    y_ = df.loc[df.id == id, 'y'].shift(1).fillna(0)
    df.loc[df.id == id, 'y_'] = y_
o_cols = o_cols + ['y_']
    
# Scaling
x_scaler = StandardScaler().fit(df[o_cols].sample(sample_size, random_state = 7))
df.loc[:, o_cols] = x_scaler.transform(df[o_cols])
print("Scaling done")

# Dropping correlated features
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
def get_top_abs_correlations(df, n):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels = labels_to_drop).sort_values(ascending = False)
    return au_corr[0:n]
drop_cols = get_top_abs_correlations(df[o_cols], corr_drop).reset_index().level_0
drop_cols = list(set(drop_cols) - set(['t', 'y_']))
df = df.drop(drop_cols, axis = 1)
d_cols = list(set(o_cols) - set(drop_cols) - set(['y_']))
print('Dropped most correlated columns, {} features left'.format(len(d_cols)))

# PCA
pca = PCA(n_components = pca_components).fit(df[d_cols].sample(sample_size, random_state = 7))
pca_cols = ["x{}".format(x) for x in np.arange(len(pca.components_))]
Xpca = pd.DataFrame(pca.transform(df[d_cols]), index = df.index, columns=pca_cols, dtype='float16')
df = pd.concat([df[['id', 'timestamp', 'y_']], Xpca, df[y_cols]], axis = 1)
train_cols = pca_cols + y_cols
del Xpca
print("PCA done, {} features left".format(len(pca_cols)))


# RNN sequence creation
df['y_'] = 0
pred_cols = pca_cols + ['y_']
Xit = {id:df.loc[df.id == id].set_index('timestamp')[pred_cols] for id in df.id.unique()}
Xis = {}
for id in Xit.keys():
    Xid = Xit[id]
    Xid = list(Xid.T.to_dict('list').values())
    xs = []
    for i in range(s_length, len(Xid)):
        s = Xid[i - s_length:i]
        xs.append(s)
    Xis[id] = xs
X = list(itertools.chain.from_iterable(Xis.values()))
Yit = {id:list(df.loc[df.id == id].set_index('timestamp')[y_cols].T.to_dict('list').values())[s_length:] for id in df.id.unique()}
y = list(itertools.chain.from_iterable(Yit.values()))
# In a stateful network, you should only pass inputs with a number of samples that can be divided by the batch size
#X = X[:-(len(X) % batch_size)]
#y = y[:-(len(y) % batch_size)]
del df
print('RNN training sequences created')

# RNN
def create_keras_model(feature_count):
    #optimizer = Nadam(lr = learning_rate)
    optimizer = SGD(lr = learning_rate, momentum = learning_rate_momentum, decay = learning_rate_decay, nesterov = True)
    #optimizer = RMSprop(lr = learning_rate, decay = learning_rate_decay)
    
    model = Sequential()
    model.add(LSTM(10, input_dim = feature_count, input_length = s_length,
        return_sequences = False, consume_less = 'cpu',
        init = 'glorot_normal', inner_init = 'glorot_normal'))
    
    model.add(GaussianNoise(0.01))
    
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    model.add(Highway(activation = 'relu'))
    
    model.add(Dense(1, init = 'glorot_normal', activation = 'tanh'))
    model.compile(loss = 'mean_squared_error', optimizer = optimizer)
    return model

# Training
train_timer = time.time()
#net = create_net(len(pca_cols), learning_rate = learning_rate)
#model = tflearn.DNN(net)
model = create_keras_model(len(pred_cols))
print('X', len(X), 'y', len(y))
stderr = sys.stderr
#sys.stderr = f_null
model.fit(X, y, nb_epoch = nb_epoch, batch_size = batch_size, shuffle = True)


#sys.stderr = stderr
print('Training completed in {:.0f}s'.format(time.time() - train_timer))
print('           total time {:.0f}s'.format(time.time() - total_time))

'''
for l in model.layers:
    w = l.get_weights()
    print('layer weights', w)
'''

print('Starting prediction loop')
timestamps = timestamps.tolist()[-s_length:]
rewards = []
for id in Xit.keys():
    X = Xit[id][-s_length:]
    Xdf = pd.DataFrame(X, index = timestamps[-len(X):], dtype = 'float16')
    Xdf.y_ = Xdf.y_.fillna(0)
    y = Yit[id][-2][0] if len(Yit[id]) > 1 else 0
    y = x_scaler.mean_[-1] + x_scaler.scale_[-1] * y_
    Xdf.loc[timestamps[-1], ['y_']] = y
    ys = Yit[id][-1][0] if len(Yit[id]) > 1 else 0
    ys = x_scaler.mean_[-1] + x_scaler.scale_[-1] * ys
    Xdf['ys'] = 0
    Xdf.loc[timestamps[-1], ['ys']] = ys
    Xit[id] = Xdf
pn = pd.Panel.from_dict(Xit)
del Xit

timer = time.time()
while True:
    timestamp = o.features.timestamp[0]
    timestamps.append(timestamp)
    o.features.loc[:, 't'] = timestamp
    o.features['y_'] = 0
    o.features['ys'] = 0
    for c in o_cols:
        #print(c, bc_min[c], bc_range[c])
        #print('*', o.features[c].min(), o.features[c].max())
        '''
        o.features.loc[:, c] = boxcox(bc_mult * (o.features[c]
            .fillna(mean[c])
            .clip(quantile[c][0], quantile[c][1]) + bc_min[c] + bc_range_mult * bc_range[c]),
            lmbda = bc_lambda[c])
        '''
        Fc = o.features[c]
        Mc = mean[c]
        Qc0 = quantile[c][0]
        Qc1 = quantile[c][1]
        o.features.loc[:, c] = Fc.fillna(Mc).clip(Qc0, Qc1)

    X = x_scaler.transform(o.features[o_cols])
    X = pd.DataFrame(X, columns = o_cols)[d_cols]
    X = pca.transform(X)
    X = pd.DataFrame(X, columns = pca_cols, dtype = 'float16', index = o.features.index)
    X = pd.concat([o.features[['id']], X], axis = 1).set_index('id')
    pn.loc[:, timestamp, :] = X
    pn.loc[:, timestamp, 'y_'] = pn[:, timestamps[-2], 'ys'].fillna(0)
    #print('y_', pn.loc[:, timestamp, 'y_'])

    ids = o.target.id
    X = pn[ids, timestamps[-s_length]: timestamp, pred_cols].fillna(0).values

    ## Padding for stateful RNN
    ##Xpad = np.zeros((batch_size - len(X), s_length, len(pca_cols)))
    ##X = np.concatenate((X, Xpad))
    o.target.y = model.predict(X, batch_size = len(X))
    pn.loc[:, timestamp, 'ys'] = x_scaler.mean_[-1] + x_scaler.scale_[-1] * o.target.y
    ##o.target.y = y[:len(o.features)]

    if timestamp % 100 == 0:
        elapsed_time = time.time() - timer
        timer = time.time()
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        print("#{}) {:.1f}s {:.3f} ±{:.3f}  y_mean {:.6f}".format(timestamp, elapsed_time, r_mean, r_std, o.target.y.mean()))
        rewards = []
        pn = pn[:, timestamps[-s_length]:, :]

    o, reward, done, info = env.step(o.target)
    rewards.append(reward)
    if done:
        print("Public score: {}".format(info["public_score"]))
        print("Total time: {:.0f}s".format(time.time() - total_time))
        break