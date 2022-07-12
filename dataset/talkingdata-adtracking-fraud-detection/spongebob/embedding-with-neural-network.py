import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
train_df = pd.read_csv('../input/train.csv', nrows=40000000)
test_df = pd.read_csv('../input/test.csv')
print("load done")
#dask_df = train_df
#df_pos = dask_df[(dask_df['is_attributed'] == 1)]
#df_neg = dask_df[(dask_df['is_attributed'] == 0)]
#df_pos = df_pos.sample(n=5000)
#print(len(df_pos))
#print(len(df_neg))
#df_neg = df_neg.sample(n=2000000)
#train_df = pd.concat([df_pos,df_neg]).sample(frac=1)
#del df_pos, df_neg
#gc.collect()
def remove_unkonwn_tag(col, train_df = train_df, test_df = test_df):
    test_df.loc[~test_df[col].isin(train_df[col]),col] = 9999999

def remove_lowfreq_tag(col, train_df = train_df,test_df = test_df, N=3):
    topN_address_list = train_df[col].value_counts()
    #print(topN_address_list)
    topN_address_list = topN_address_list[topN_address_list <= N]
    topN_address_list = topN_address_list.index
    remove_list = train_df.loc[train_df[col].isin(topN_address_list), col]
    print('remove:',len(remove_list))
    print('reserve',len(train_df) - len(remove_list))
    remove_list = 9999998
    test_df.loc[test_df[col].isin(topN_address_list), col] = 9999998

def preprocess(df):
    df['time_t'] = df.click_time.str[11:13] + df.click_time.str[14:16]

for i,j in zip(['app','ip','device','os','channel'],[5,4,4,5,5]):
    remove_lowfreq_tag(i,N=j)


for i in ['app','ip','device','os','channel']:
    remove_unkonwn_tag(i)

from sklearn.preprocessing import LabelEncoder
def process_lable_encoder(col, train_df = train_df, test_df = test_df):
    le = LabelEncoder()
    le.fit(np.hstack([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

coding_list = ['ip','app','device','os','channel','time_t']

preprocess(train_df)
preprocess(test_df)
print("load done")

for i in coding_list:
    process_lable_encoder(i)
    
print("label gen done")
MAX_IP = np.max(train_df.ip.max()) + 2 #39612 #277396
MAX_DEVICE = np.max(train_df.device.max()) + 2 #299 #3475
MAX_OS = np.max(train_df.os.max()) + 2 #161 #3475
MAX_APP = np.max(train_df.app.max()) + 2 #214 #3475
MAX_CHANNEL = np.max(train_df.channel.max()) + 2  #155
MAX_TIME = np.max(train_df.time_t.max()) + 2 #24*60+1
print("MAX gen done")

def get_keras_data(df):
    X = {
        'ip': np.array(df.ip),
	    'app': np.array(df.app),
        'device': np.array(df.device),
        'os': np.array(df.os),
        'channel': np.array(df.channel),
        'clicktime': np.array(df.time_t),
    }
    return X

from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

def get_model(lr=0.001, decay=0.0):
    ip = Input(shape=[1], name="ip")
    app = Input(shape=[1], name="app")
    device = Input(shape=[1], name="device")
    os = Input(shape=[1], name="os")
    channel = Input(shape=[1], name="channel")
    clicktime = Input(shape=[1], name="clicktime")

    emb_ip = Embedding(MAX_IP, 64)(ip)
    emb_device = Embedding(MAX_DEVICE, 16)(device)
    emb_os= Embedding(MAX_OS, 16)(os)
    emb_app = Embedding(MAX_APP, 16)(app)
    emb_channel = Embedding(MAX_CHANNEL, 8)(channel)
    emb_time = Embedding(MAX_TIME, 32)(clicktime)

    main = concatenate([Flatten()(emb_ip), 
                        Flatten()(emb_device), 
                        Flatten()(emb_os),
                        Flatten()(emb_app),
                        Flatten()(emb_channel), 
                        Flatten()(emb_time)])
    main = Dense(128,kernel_initializer='normal', activation="tanh")(main)
    main = Dropout(0.2)(main)
    main = Dense(64,kernel_initializer='normal', activation="tanh")(main)
    main = Dropout(0.2)(main)    
    main = Dense(32,kernel_initializer='normal', activation="relu")(main)
    output = Dense(1,activation="sigmoid") (main)
    #model
    model = Model([ip, app, device, os, channel, clicktime], output)
    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="binary_crossentropy", 
                  optimizer=optimizer)
    return model


from sklearn.model_selection import train_test_split


Y_train = train_df.is_attributed.values.reshape(-1, 1)

X_train, X_valid, y_train, y_valid = train_test_split(train_df[coding_list], Y_train, test_size = 0.1, random_state= 1984, stratify = Y_train)
X_train = get_keras_data(X_train[coding_list])
X_valid = get_keras_data(X_valid[coding_list])

print("Defining  model...")

# Model hyper parameters.
BATCH_SIZE = 1024*2
epochs = 1

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_df['ip']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.001, 0.0003
lr_decay = exp_decay(lr_init, lr_fin, steps)

model = get_model(lr=lr_init, decay=lr_decay)

print("Fitting  model to training examples...")
cw = {0: 1, 1: 3}
for i in range(1):
    model.fit(
            X_train, y_train, epochs=1, batch_size=BATCH_SIZE,
            validation_data=(X_valid, y_valid), verbose=1,class_weight=cw
    )
    y_val_pred = model.predict(X_valid)[:, 0]
    print('Valid AUC: {:.4f}'.format(roc_auc_score(y_valid, y_val_pred)))

X_test = get_keras_data(test_df[coding_list])
preds = model.predict(X_test, batch_size=BATCH_SIZE)
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
sub['is_attributed'] = preds
sub.to_csv('sub_nn.csv', index=False)
print(sub.head())
