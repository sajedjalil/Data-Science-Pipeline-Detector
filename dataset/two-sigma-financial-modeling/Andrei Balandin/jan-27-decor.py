import kagglegym
import numpy as np
import pandas as pd
import tflearn
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

env = kagglegym.make()
o = env.reset()
o_cols = x_cols = [c for c in o.train.columns if not c in ['id', 'timestamp', 'y', 'sample']]

# clip y
df = o.train
df = df[~(df.y > 0.093497) & ~(df.y < -0.086093)]
df = df[df.timestamp > df.timestamp.max() * 0.3]

# Scaling
def scale_test(df):
    df = df.fillna(0)
    x_scaler = StandardScaler()
    df.loc[:, x_cols] = x_scaler.fit_transform(df[x_cols])
    df.loc[:, x_cols] = df[x_cols].astype('float16')
    return df,x_scaler
df, x_scaler = scale_test(df)
print("Scaling done")

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
drop_cols = get_top_abs_correlations(df[x_cols], 15).reset_index().level_0
df = df.drop(drop_cols, axis = 1)
d_cols = x_cols = list(set(x_cols) - set(drop_cols))
print('Dropped correlated columns, {} columns left of {}'.format(len(d_cols), len(o_cols)))

pca_components = 70
pca = PCA(n_components=pca_components)
pca.fit(df[d_cols])
Xt = pd.DataFrame(pca.transform(df[d_cols]), dtype='float16',
    columns=["x{}".format(x) for x in np.arange(pca_components)])
x_cols = Xt.columns
df1 = df[['id', 'timestamp']].reset_index()
df2 = df[['y']].reset_index()
df = pd.concat([df1, Xt, df2], axis=1)
print("PCA done, {} components".format(Xt.shape[1]))
del df1
del df2
del Xt

net = tflearn.input_data([None, len(x_cols)])
net = tflearn.fully_connected(net, 40, activation='relu', regularizer='L2')
net = tflearn.batch_normalization(net)
net = tflearn.fully_connected(net, 40, activation='relu', regularizer='L2')
net = tflearn.batch_normalization(net)
net = tflearn.fully_connected(net, 40, activation='relu', regularizer='L2')
net = tflearn.batch_normalization(net)
net = tflearn.fully_connected(net, 10, activation='relu', regularizer='L2')
net = tflearn.batch_normalization(net)
net = tflearn.fully_connected(net, 4, activation='relu', regularizer='L2')
net = tflearn.fully_connected(net, 1, activation='tanh', regularizer='L2')
net = tflearn.reshape(net, [-1])
net = tflearn.regression(net, learning_rate=0.001, loss='mean_square')

# Training
model = tflearn.DNN(net)
#model.fit(df[x_cols].values, df['y'].values, validation_set=0.1, show_metric=True, batch_size=16000)
model.fit(df[x_cols].values, df['y'].values)

max_r,min_r = (-1,1)
min_r = 1
while True:
    timestamp = o.features["timestamp"][0]
    f = o.features[o_cols].fillna(0)
    f = pd.DataFrame(x_scaler.transform(f), columns = o_cols)[d_cols]
    Xt = pca.transform(f)
    p = model.predict(Xt)
    o.target.y = p
    o.target.y = np.clip(p, -0.086093, 0.093497)
    
    if timestamp % 100 == 0:
        print("Timestamp #{}, reward {:+.4f} .. {:+.4f}".format(timestamp, min_r, max_r))
        max_r,min_r = (-1,1)

    o, reward, done, info = env.step(o.target)
    max_r = reward if reward > max_r else max_r
    min_r = reward if reward < min_r else min_r
    if done:
        print("Public score: {}".format(info["public_score"]))
        break