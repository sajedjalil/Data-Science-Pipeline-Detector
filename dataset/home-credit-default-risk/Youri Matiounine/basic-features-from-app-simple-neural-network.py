import time
import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Load data
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('Loading data...')
rows_read = None
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=rows_read)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=rows_read)
print('Time elapsed %.0f sec'%(time.time()-start_time))
print('Pre-processing data...')

# Merge the datasets into a single one
target = app_train_df.pop('TARGET')
len_train = len(app_train_df)
merged_df = pd.concat([app_train_df, app_test_df])
meta_df = merged_df.pop('SK_ID_CURR')
del app_test_df, app_train_df
gc.collect()

# Encode categoricals: 1-hot
categorical_feats = merged_df.columns[merged_df.dtypes == 'object']
print('Using %d prediction variables'%(merged_df.shape[1]))
print('Encoding %d non-numeric columns...'%(merged_df.columns[merged_df.dtypes == 'object'].shape))
for feat in categorical_feats:
    merged_df[feat].fillna('MISSING', inplace=True) # populate missing labels
    encoder = LabelBinarizer() # works with text
    new_columns = encoder.fit_transform(merged_df[feat])
    i=0
    for u in merged_df[feat].unique():
        if i<new_columns.shape[1]:
            merged_df[feat+'_'+u]=new_columns[:,i]
            i+=1
    merged_df.drop(feat, axis=1, inplace=True)
print('Now using %d prediction variables'%(merged_df.shape[1]))
print('Time elapsed %.0f sec'%(time.time()-start_time))

# handle missing values
null_counts = merged_df.isnull().sum()
null_counts = null_counts[null_counts > 0]
null_ratios = null_counts / len(merged_df)

# Drop columns over x% null
null_thresh = .8
null_cols = null_ratios[null_ratios > null_thresh].index
merged_df.drop(null_cols, axis=1, inplace=True)
if null_cols.shape[0] > 0:
    print('Columns dropped for being over %.2f null:'%(null_thresh))
    for col in null_cols:
        print(col)

# Fill the rest with 0
merged_df.fillna(0, inplace=True)

# scale continuous features
# first, convert large ingegers into floats.
for feat in merged_df.columns:
    if (merged_df[feat].max() > 100) | (merged_df[feat].min() < -100):
        merged_df[feat]=merged_df[feat].astype(np.float64)
scaler = StandardScaler()
continuous_feats = merged_df.columns[merged_df.dtypes == 'float64']
print('Scaling %d features...'%(continuous_feats.shape))
s1 = merged_df.shape[0],1
for feat in continuous_feats:
    merged_df[feat] = scaler.fit_transform(merged_df[feat].values.reshape(s1))

# Re-separate into train and test
train_df = merged_df[:len_train]
test_df = merged_df[len_train:]
del merged_df
gc.collect()

print('Time elapsed %.0f sec'%(time.time()-start_time))
print('Starting training...')

# define train parameters
L2c = 4e-4                    # loss, with L2
lr0 = 0.02                    # starting learning rate
lr_decay = 0.90               # lr decay rate
iterations = 41               # full passes over data
ROWS = train_df.shape[0]      # rows in input data
VARS = train_df.shape[1]      # vars used in the model
NUMB = 10000                  # batch size
NN = int(ROWS/NUMB)           # number of batches

# define the model
y_ = tf.placeholder(tf.float32, [None, 1])
x  = tf.placeholder(tf.float32, [None, VARS])

# model: logistic + 1 hidden layer
W      = tf.Variable(tf.truncated_normal([VARS,1],mean=0.0,stddev=0.001),dtype=np.float32)
NUML1  = 64
W1     = tf.Variable(tf.truncated_normal([VARS,NUML1],mean=0.0,stddev=0.0001),dtype=np.float32)
W1f    = tf.Variable(tf.truncated_normal([NUML1,1],mean=0.0,stddev=0.0001),dtype=np.float32)
logit1 = tf.matmul( x, W ) + tf.matmul(tf.nn.relu(tf.matmul( x, W1 )), W1f)
y      = tf.nn.sigmoid( logit1 )

# loss/optimizer
loss0 = tf.reduce_mean( (y_-y)*(y_-y) )
loss1 = L2c * (tf.nn.l2_loss( W ) + tf.nn.l2_loss( W1 ) + tf.nn.l2_loss( W1f ))
loss  = loss0 + loss1
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr0, global_step, NN, lr_decay)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# main training loop
y0=target.values.astype(np.float32)
x0=train_df.values.astype(np.float32)
del train_df
gc.collect()
y0_1=np.where(y0[0:int(NN*0.8)*NUMB] == 1)[0] # reserve last 20% for testing
y0_0=np.where(y0[0:int(NN*0.8)*NUMB] == 0)[0] # reserve last 20% for testing
for i in range(iterations):
    for j in range(int(NN*0.8)): # reserve last 20% for testing
        pos_ratio = 0.5
        pos_idx = np.random.choice(y0_1, size=int(np.round(NUMB*pos_ratio)))
        neg_idx = np.random.choice(y0_0, size=int(np.round(NUMB*(1-pos_ratio))))
        idx = np.concatenate([pos_idx, neg_idx])
        fd = {y_: y0[idx].reshape(NUMB,1),x:  x0[idx,:]}
        _= sess.run( [train_step], feed_dict=fd )
    if i%10 == 0:
        # get area under the ROC curve
        fd   = {y_: y0.reshape(y0.shape[0],1),x: x0}
        y1   = sess.run( y, feed_dict=fd )
        lim  = int(NN*0.8) * NUMB
        auc1 = roc_auc_score(y0[0:lim],y1[0:lim,0])
        auc2 = roc_auc_score(y0[lim:y0.shape[0]],y1[lim:y0.shape[0],0])
        print('iteration %d, auc train/validatet %.5f/%.5f'%(i,auc1,auc2))

#Predict on test set and create submission
x0     = test_df.values.astype(np.float32)
fd     = {y_: np.zeros([x0.shape[0],1]),x: x0}
y_pred = sess.run( y, feed_dict=fd )
out_df = pd.DataFrame({'SK_ID_CURR': meta_df[len_train:], 'TARGET': y_pred[:,0]})
out_df.to_csv('submission.csv', index=False)
print('Time elapsed %.0f sec'%(time.time()-start_time))