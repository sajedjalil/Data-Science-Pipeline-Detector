import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import time
import warnings
import sys
from numba import njit
warnings.filterwarnings('ignore')

multAE2 = 0.97 # decay rate for the second AE
qg_th   = 0.50 # threshold for q group split
AE_TOT  = 0.644938480942317 # mean correct answer rate, excl repetitions and lectures


import riiideducation
env = riiideducation.make_env()
iter_test = env.iter_test()


# Load data
start_time = time.time()
print(' Loading data...')

# add content_id categorical embeddings, extracted from NN
emb = pd.read_csv('/kaggle/input/riiidsub2/emb.csv').reset_index()
emb.columns = ['content_id','cid_f1','cid_f2','cid_f3','cid_f4','cid_f5','cid_f6','cid_f7','cid_f8']
emb['content_id'] = emb['content_id'].astype('int16')
for col in ['cid_f1','cid_f2','cid_f3','cid_f4','cid_f5','cid_f6','cid_f7','cid_f8']:
    emb[col] = emb[col].astype('float32')



# Process questions data
q = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv', dtype={'question_id':'int16', 'bundle_id':'int16', 'correct_answer':'int8', 'part':'int8', 'tags':'str'})
q = q.rename(columns={'question_id': 'content_id'})
q['tags'][10033] = '255' # fix one bad value; treat 255 as "not present"
q['tags'] = q['tags'].str.split(' ')
q['tag0'] = q['tags'].apply(lambda x: x[0]).astype('int16')
q['tag1'] = q['tags'].apply(lambda x: x[1] if len(x) > 1 else 255).astype('int16')
q['tag2'] = q['tags'].apply(lambda x: x[2] if len(x) > 2 else 255).astype('int16')
q['tag3'] = q['tags'].apply(lambda x: x[3] if len(x) > 3 else 255).astype('int16')
q['tag4'] = q['tags'].apply(lambda x: x[4] if len(x) > 4 else 255).astype('int16')
q['tag5'] = q['tags'].apply(lambda x: x[5] if len(x) > 5 else 255).astype('int16')
q.drop(['correct_answer', 'tags'], axis=1, inplace=True)
# number of qs in current batch
df = q.groupby('bundle_id').size().reset_index()
df.columns = ['bundle_id', 'bundle_size']
q = q.merge(df, how='left', on='bundle_id')
for col in ['tag0', 'tag1']:
    q[col] = q[col].astype('int16')
q['bundle_size'] = q['bundle_size'].astype('int8')
q.drop('bundle_id', axis=1, inplace=True)



# Process train data
dtypes = {'row_id':'int64', 'timestamp':'int64', 'user_id':'int32', 'content_id':'int16',
'content_type_id':'int8', 'task_container_id':'int16', 'user_answer':'int8', 'answered_correctly':'int8',
'prior_question_elapsed_time':'float32', 'prior_question_had_explanation':'boolean'}
usecols = ['user_id', 'content_id', 'content_type_id',
       'task_container_id', 'user_answer', 'answered_correctly',
       'prior_question_elapsed_time', 'prior_question_had_explanation','timestamp']
df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', dtype=dtypes, usecols=usecols)
df.rename(columns={'prior_question_elapsed_time': 'time', 'prior_question_had_explanation': 'expl'}, inplace=True)
print(' Finished loading. Time elapsed %.0f sec'%(time.time()-start_time))

@njit
def cs_dev_user(data, user, mult, size):# return sum per user
    x = np.zeros(size)
    c = 0
    x[0] = data[0]
    for i in range(1, data.shape[0]):
        if user[i] != user[i-1]:# incr destination for new user
            c += 1
        x[c] = x[c] * mult + data[i]
    return x

@njit
def cs_dev_user_last(data, user, size):# return last item per user
    x = np.zeros(size)
    x[0] = data[0]
    c = 0
    for i in range(1, data.shape[0]):
        if user[i] != user[i-1]:# new user, save
            x[c] = data[i-1]
            c += 1
    return x

@njit
def cs(data, user, task, mult):# cumsum per user
    x = np.zeros(data.shape[0])
    x[0] = 0
    result = 0
    for i in range(1, data.shape[0]):
        result = result * mult + data[i-1]
        x[i] = result
        if user[i] != user[i-1]:# reset totals for new user
            x[i] = 0
            result = 0
        if task[i] == task[i-1]:# do not count questions in current batch
            x[i] = x[i-1]
    return x

@njit
def cs_dt(data, user, task):# time since the end of last batch
    x = np.zeros(data.shape[0])
    x[0] = -1000
    for i in range(1, data.shape[0]):
        x[i] = (data[i] - data[i-1]) / 1000
        if task[i] == task[i-1]:# do not count questions in current batch
            x[i] = x[i-1]
        if user[i] != user[i-1]:# reset for new user
            x[i] = -1000
    return x

df = df.merge(q[['content_id','bundle_size']], how='left', on='content_id', right_index=True)
df['dt'] = np.maximum(-1, (cs_dt(np.array(df['timestamp']), np.array(df['user_id']), np.array(df['task_container_id'])) / df['bundle_size'].fillna(1))).astype('float32')
print(' Finished dt. Time elapsed %.0f sec'%(time.time()-start_time))

@njit
def cs_dt2(data, user, task):# time since the end of previous batch
    x = np.zeros(data.shape[0])
    x[0] = -1000
    x[1] = -1000
    for i in range(2, data.shape[0]):
        x[i] = (data[i] - data[i-1]) / 1000
        # find Nth different time
        cc = 0
        for j in range(i-1, -1, -1):
            if data[j] != data[j+1]:
                cc += 1
                if cc >= 2:
                    x[i] = max(-1000, (data[i] - data[j]) / 1000)
                    break
        if task[i] == task[i-1]:# do not count questions in current batch
            x[i] = x[i-1]
        if user[i] != user[i-1] or user[i-1] != user[i-2]:# reset for new user
            x[i] = -1000
    return x

df['dt2'] = cs_dt2(np.array(df['timestamp']), np.array(df['user_id']), np.array(df['task_container_id'])).astype('float32')
df['dt2'] = np.maximum(-1, (df['dt2'] - np.maximum(0, df['dt'])) / df['bundle_size'].fillna(1)).astype('float32')
df.drop(['user_answer', 'bundle_size', 'content_type_id'], axis=1, inplace=True)
print(' Finished dt2. Time elapsed %.0f sec'%(time.time()-start_time))



# Drop lectures
df = df.loc[df['answered_correctly'] >= 0].reset_index(drop=True)
print(' Finished dropping lectures. Time elapsed %.0f sec'%(time.time()-start_time))



# Do this after dropping lectures: median dt by q
mdt = df.groupby('content_id')['dt'].median().reset_index()
mdt.columns = ['content_id','mdt']
mdt['mdt'] = np.minimum(300, np.maximum(5, mdt['mdt'])) # floor/cap
q = q.merge(mdt, how='left', on='content_id', right_index=True)
del mdt
gc.collect()
print(' Finished mdt. Time elapsed %.0f sec'%(time.time()-start_time))

# add ratio of dt to median dt by q
df = df.merge(q[['content_id','mdt']], how='left', on='content_id', right_index=True)
df['dt_q'] = np.maximum(-0.1, df['dt'] / df['mdt']).astype('float32')
df.drop('mdt', axis=1, inplace=True)

# median dt2 by q
mdt = df.groupby('content_id')['dt2'].median().reset_index()
mdt.columns = ['content_id','mdt2']
mdt['mdt2'] = np.minimum(300, np.maximum(5, mdt['mdt2'])) # floor/cap
q = q.merge(mdt, how='left', on='content_id', right_index=True)
del mdt
gc.collect()
print(' Finished mdt2. Time elapsed %.0f sec'%(time.time()-start_time))

# median time by q
mdt = df.groupby('content_id')['time'].median().fillna(0).reset_index()
mdt.columns = ['content_id','mdt3']
mdt['mdt3'] = np.minimum(300, np.maximum(5, mdt['mdt3'].fillna(0)/1000)) # floor/cap
q = q.merge(mdt, how='left', on='content_id', right_index=True)
del mdt
gc.collect()
print(' Finished mdt3. Time elapsed %.0f sec'%(time.time()-start_time))

# add ratio of time to median time by q
df = df.merge(q[['content_id','mdt3']], how='left', on='content_id', right_index=True)
df['dt3_q'] = np.maximum(-0.1, df['time'].fillna(0)/1000 / df['mdt3']).astype('float32')
df.drop('mdt3', axis=1, inplace=True)
print(' Finished dt_q. Time elapsed %.0f sec'%(time.time()-start_time))



# group by content_id***************************************************************
df['one']             = np.ones((df.shape[0],), dtype=np.int8)
df_content_id         = df[['content_id','answered_correctly']].groupby('content_id').agg(['mean','size']).reset_index()
df_content_id.columns = ['content_id', 'q_mean', 'q_cnt']
print(' Finished df_content_id. Time elapsed %.0f sec'%(time.time()-start_time))

# use optimized q_mean - from iterations in a separate script. It adjusts each q for ability of users who took them.
df_content_id0 = pd.read_csv('/kaggle/input/riiidsub2/df_content_id0.csv')
df_content_id['q_mean'] = df_content_id0['q_mean'].astype('float32')
del df_content_id0

# blend df_content_id values with AE_TOT - credibity weighting
w = 10 # give this much count to total
df_content_id['q_mean'] = np.round(df_content_id['q_mean'] * df_content_id['q_cnt'] + AE_TOT * w, 0) / (df_content_id['q_cnt'] + w)
df_content_id['q_cnt']  += w

# change data types
df_content_id['content_id'] = df_content_id['content_id'].astype('int16')
df_content_id['q_mean']     = df_content_id['q_mean'].astype('float32')
df_content_id['q_cnt']      = df_content_id['q_cnt'].astype('int32')
print(' Finished grouping by content_id. Time elapsed %.0f sec'%(time.time()-start_time))



# add expected by content_id to df_user_id; now i have q_mean
df = df.merge(df_content_id[['content_id', 'q_mean']], how='left', on='content_id', right_index=True)

# apply durational adjustment for dt: A/E, hardcoded
dur_adj_di={-1:0.901,0:0.307,1:0.407,2:0.473,3:0.471,4:0.458,5:0.505,6:0.601,
7:0.725,8:0.878,9:1.022,10:1.143,11:1.232,12:1.295,13:1.326,14:1.341,15:1.334,16:1.34,17:1.345,
18:1.347,19:1.341,20:1.346,21:1.341,22:1.317,23:1.276,24:1.242,25:1.215,26:1.192,27:1.168,28:1.149,
29:1.128,30:1.106,31:1.086,32:1.069,33:1.049,34:1.035,35:1.023,36:1.01,37:1.004,38:0.999,39:0.988,
40:0.983,41:0.981,42:0.969,43:0.964,44:0.959,45:0.955,46:0.948,47:0.942,48:0.942,49:0.94,50:0.905}
m = np.minimum(50, df['dt'].astype('int32')).map(dur_adj_di)
df['q_mean'] = (df['q_mean'] * m / (1 + df['q_mean'] * (m - 1))).astype('float32')
df['q_mean'] = np.maximum(0.00001, np.minimum(0.99999, df['q_mean'])) # cap/floor to 5 digits
del m
gc.collect()
print(' Finished adding expecteds. Time elapsed %.0f sec'%(time.time()-start_time))



# group by user_id*********************************************************************
df_user_id          = pd.DataFrame(df['user_id'].unique(), columns=['user_id'])
df_user_id['dtc']   = cs_dev_user(np.array(np.minimum(400, np.maximum(0, df['dt']))), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['dt3c']  = cs_dev_user(np.array(np.minimum(400, np.maximum(0, df['time'].fillna(0)/1000))), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['act1']  = cs_dev_user(np.array(df['answered_correctly']), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['exp1']  = cs_dev_user(np.array(df['q_mean']), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['cnt']   = cs_dev_user(np.array(df['one']), np.array(df['user_id']), 1, df_user_id.shape[0])

# add AE2 fields
df_user_id['act2']  = cs_dev_user(np.array(df['answered_correctly']), np.array(df['user_id']), multAE2, df_user_id.shape[0])
df_user_id['exp2']  = cs_dev_user(np.array(df['q_mean']), np.array(df['user_id']), multAE2, df_user_id.shape[0])
df_user_id['cnt2']  = cs_dev_user(np.array(df['one']), np.array(df['user_id']), multAE2, df_user_id.shape[0])
df_user_id['AE2mc'] = cs_dev_user(np.array(df['answered_correctly'] / df['q_mean']), np.array(df['user_id']), multAE2, df_user_id.shape[0])
df_user_id['AE2mic']= cs_dev_user(np.array((1 - df['answered_correctly']) / (1 - df['q_mean'])), np.array(df['user_id']), multAE2, df_user_id.shape[0])
print(' Finished user AE2. Time elapsed %.0f sec'%(time.time()-start_time))

# add partgX fields
df                      = df.merge(q[['content_id','part']], how='left', on='content_id', right_index=True)
d                       = np.array(df['answered_correctly'] * (df['part'] < 5))
df_user_id['act1pg2']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
d                       = np.array(df['q_mean'] * (df['part'] < 5))
df_user_id['exp1pg2']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
d                       = np.array(df['answered_correctly'] * (df['part'] >= 5))
df_user_id['act1pg1']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
d                       = np.array(df['q_mean'] * (df['part'] >= 5))
df_user_id['exp1pg1']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['cnt1pg2']   = cs_dev_user(np.array(df['part'] < 5), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['cnt1pg1']   = cs_dev_user(np.array(df['part'] >= 5), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['AE1mEpgc2'] = cs_dev_user(np.array((df['part'] < 5) * (df['answered_correctly'] * np.log(df['q_mean']) + (1 - df['answered_correctly']) * np.log(1 - df['q_mean']))), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['AE1mEpgc1'] = cs_dev_user(np.array((df['part'] >= 5) * (df['answered_correctly'] * np.log(df['q_mean']) + (1 - df['answered_correctly']) * np.log(1 - df['q_mean']))), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['AE1mipgc2'] = cs_dev_user(np.array((df['part'] < 5) * (1-df['answered_correctly']) / (1-df['q_mean'])), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['AE1mipgc1'] = cs_dev_user(np.array((df['part'] >= 5) * (1-df['answered_correctly']) / (1-df['q_mean'])), np.array(df['user_id']), 1, df_user_id.shape[0])

# p
for i in range(1, 8):
    df_user_id['cnt1p'+str(i)]   = cs_dev_user(np.array(df['part'] == i), np.array(df['user_id']), 1, df_user_id.shape[0])
    df_user_id['AE1mEpc'+str(i)] = cs_dev_user(np.array((df['part'] == i) * (df['answered_correctly'] * np.log(df['q_mean']) + (1 - df['answered_correctly']) * np.log(1 - df['q_mean']))), np.array(df['user_id']), 1, df_user_id.shape[0])
    d                            = np.maximum(0.01, df['answered_correctly']) * np.maximum(0.01, 1 - df['q_mean']) / np.maximum(0.01, df['q_mean']) / np.maximum(0.01, 1 - df['answered_correctly'])
    df_user_id['AE1mOpc'+str(i)] = cs_dev_user(np.array((df['part'] == i) * d), np.array(df['user_id']), 1, df_user_id.shape[0])

# qg
d                       = np.array(df['answered_correctly'] * (df['q_mean'] < qg_th))
df_user_id['act1qg2']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
d                       = np.array(df['q_mean'] * (df['q_mean'] < qg_th))
df_user_id['exp1qg2']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
d                       = np.array(df['answered_correctly'] * (df['q_mean'] >= qg_th))
df_user_id['act1qg1']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
d                       = np.array(df['q_mean'] * (df['q_mean'] >= qg_th))
df_user_id['exp1qg1']   = cs_dev_user(d, np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['cnt1qg2']   = cs_dev_user(np.array(df['q_mean'] < qg_th), np.array(df['user_id']), 1, df_user_id.shape[0])
df_user_id['cnt1qg1']   = cs_dev_user(np.array(df['q_mean'] >= qg_th), np.array(df['user_id']), 1, df_user_id.shape[0])
print(' Finished user p/qq/pgX. Time elapsed %.0f sec'%(time.time()-start_time))

# last_timestamps by user
df_user_id['t_p']  = cs_dev_user_last(np.array(df['timestamp']), np.array(df['user_id']), df_user_id.shape[0]).astype('int64')
# this is not exactly correct for trailing batches, but is close enough
df_user_id['t_p2'] = cs_dev_user_last(np.array(df['timestamp'].shift(periods=1).fillna(0)), np.array(df['user_id']), df_user_id.shape[0]).astype('int64')

# time of last incorrect answer
@njit
def cs_tt4(cond, time, user, task):
    x = np.zeros(time.shape[0], dtype=np.int64)
    x[0] = 0
    for i in range(1, time.shape[0]):
        for j in range(i-1, -1, -1):
            if cond[j] == 1 or user[i] != user[j]:
                if user[i] != user[j]:
                    j += 1
                break
        x[i] = time[j]
        if task[i] == task[i-1]:
            x[i] = x[i-1]
    return x

d = cs_tt4(np.array(df['answered_correctly']==0), np.array(df['timestamp'], dtype=np.int64), np.array(df['user_id']), np.array(df['task_container_id'])).astype('int64')
df_user_id['t_incor']  = cs_dev_user_last(d, np.array(df['user_id']), df_user_id.shape[0]).astype('int64')
del d
gc.collect()

# some other items
df = df.merge(q[['content_id','tag0']], how='left', on='content_id', right_index=True)

df_user_id['tag0'] = cs_dev_user_last(np.array(df['tag0']), np.array(df['user_id']), df_user_id.shape[0]).astype('int64')
df_user_id['part'] = cs_dev_user_last(np.array(df['part']), np.array(df['user_id']), df_user_id.shape[0]).astype('int64')

@njit
def cum_f5(data, user, task):
    size = data.shape[0]
    x = np.zeros(size)
    result = 1
    result_p = 1
    data_p = data[0]
    for i in range(1, size):
        # get outputs
        if data[i] == data_p:
            x[i] = result_p
        else:
            x[i] = 0
        # update result
        if data[i] == data[i-1]:
            result += 1
        else:
            result = 1
        # at the end of batch update stored values
        if i < size - 1 and task[i] != task[i+1]:
            data_p = data[i]
            result_p = result
        # reset totals for new user
        if user[i] != user[i-1]:
            x[i] = 0
            result = 1
            result_p = 1
            data_p = data[i]
    return x

# cum count of the same tag0
df['cc_tg0'] = cum_f5(np.array(df['tag0'], dtype=np.int16), np.array(df['user_id']), np.array(df['task_container_id'])).astype('int16')
df.drop('tag0', axis=1, inplace=True)
# cum count of the same part
df['cc_part'] = cum_f5(np.array(df['part'], dtype=np.int16), np.array(df['user_id']), np.array(df['task_container_id'])).astype('int16')
# starts from 1, not 0
df_user_id['cc_tg0']  = 1 + cs_dev_user_last(np.array(df['cc_tg0']), np.array(df['user_id']), df_user_id.shape[0]).astype('int16')
df_user_id['cc_part'] = 1 + cs_dev_user_last(np.array(df['cc_part']), np.array(df['user_id']), df_user_id.shape[0]).astype('int16')

@njit
def cs_dev_user_last15(x, data, user):# return last 15 items per user
    c = 0
    for i in range(1, data.shape[0]):
        if user[i] != user[i-1]:# new user, save
            for j in range(15):
                x[c, j] = data[i - 15 + j]
            c += 1
    return x

x = np.zeros([df_user_id.shape[0], 15], dtype=np.int64)
last_times      = cs_dev_user_last15(x, np.array(df['timestamp']), np.array(df['user_id']))
last_answers    = cs_dev_user_last15(x, np.array(df['answered_correctly']), np.array(df['user_id'])).astype('int8')
last_cid        = cs_dev_user_last15(x, np.array(df['content_id']), np.array(df['user_id'])).astype('int16')
del x
gc.collect()

# clean it up
for col in df_user_id.columns:
    if col in ['user_id', 'cnt', 'tag0', 'part', 'cc_tg0', 'cc_part', 'cnt1pg1', 'cnt1pg2', 'cnt1qg1', 'cnt1qg2']:
        df_user_id[col] = df_user_id[col].astype('int32')
    elif col in ['t_p', 't_p2']:
        df_user_id[col] = df_user_id[col].astype('int64')
    else:
        df_user_id[col] = df_user_id[col].fillna(0).astype('float32')
print(' Finished grouping by user_id. Time elapsed %.0f sec'%(time.time()-start_time))



def FE(df):# feature engeneering - construct df from di
    u_cnt    = np.zeros(df.shape[0], dtype=np.float32)
    u_dtm    = np.zeros(df.shape[0], dtype=np.float32)
    u_dt3m   = np.zeros(df.shape[0], dtype=np.float32)
    u_AE2    = np.zeros(df.shape[0], dtype=np.float32)
    u_AE2i   = np.zeros(df.shape[0], dtype=np.float32)
    u_AE2O   = np.zeros(df.shape[0], dtype=np.float32)
    u_AE2m   = np.zeros(df.shape[0], dtype=np.float32)
    u_AE2mi  = np.zeros(df.shape[0], dtype=np.float32)
    u_pgXcnt = np.zeros(df.shape[0], dtype=np.int32)
    u_AE1OpgX= np.zeros(df.shape[0], dtype=np.float32)
    u_AE1mEpgX=np.zeros(df.shape[0], dtype=np.float32)
    u_AE1mipgX=np.zeros(df.shape[0], dtype=np.float32)
    u_qgXcnt = np.zeros(df.shape[0], dtype=np.int32)
    u_AE1qgX = np.zeros(df.shape[0], dtype=np.float32)
    u_pXcnt  = np.zeros(df.shape[0], dtype=np.int32)
    u_AE1mEpX= np.zeros(df.shape[0], dtype=np.float32)
    u_AE1mOpX= np.zeros(df.shape[0], dtype=np.float32)
    u_cc_tg0 = np.zeros(df.shape[0], dtype=np.int32)
    u_cc_part= np.zeros(df.shape[0], dtype=np.int32)
    u_t_p    = np.zeros(df.shape[0], dtype=np.int64)
    u_t_incor= np.zeros(df.shape[0], dtype=np.int64)
    u_t_p2   = np.zeros(df.shape[0], dtype=np.int64)
    u_t_n_att= np.zeros((df.shape[0], 5), dtype=np.int32)
    u_y      = np.zeros((df.shape[0], 15), dtype=np.int8)
    u_cid    = np.zeros((df.shape[0], 15), dtype=np.int16)
    for i in range(df.shape[0]):
        q_mean      = df['q_mean'].iat[i]
        data_u      = u_stats.get_item(df['user_id'].iat[i], df['part'].iat[i], df['tag0'].iat[i])
        u_cnt[i]    = data_u['cnt']
        u_dtm[i]    = data_u['dtm']
        u_dt3m[i]   = data_u['dt3m']
        u_AE2[i]    = data_u['AE2']
        u_AE2i[i]   = data_u['AE2i']
        u_AE2O[i]   = data_u['AE2O']
        u_AE2m[i]   = data_u['AE2m']
        u_AE2mi[i]  = data_u['AE2mi']
        u_pgXcnt[i] = data_u['pgXcnt']
        u_AE1OpgX[i]= data_u['AE1OpgX']
        u_AE1mEpgX[i]=data_u['AE1mEpgX']
        u_AE1mipgX[i]=data_u['AE1mipgX']
        u_pXcnt[i]  = data_u['pXcnt']
        u_AE1mEpX[i]= data_u['AE1mEpX']
        u_AE1mOpX[i]= data_u['AE1mOpX']
    
        u_cc_tg0[i] = data_u['cc_tg0']
        u_cc_part[i]= data_u['cc_part']
        u_t_p[i]    = data_u['t_p']
        u_t_incor[i]= data_u['t_incor']
        if u_cnt[i] < 2:
            u_t_p2[i] = u_t_p[i]
        else:
            u_t_p2[i] = data_u['t_p2']
        j = 0
        for w in [3, 10, 20]:
            u_t_n_att[i, j] = (df['timestamp'].iat[i] - np.array(data_u['timestamp']) < w * 10000).sum()
            j += 1
        for w in [10, 20]:
            u_t_n_att[i, j] = ((np.array(data_u['answer']) == 0) * (df['timestamp'].iat[i] - np.array(data_u['timestamp']) < w * 10000)).sum()
            j += 1
        # prior answers and cid
        for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            u_y[i, w]   = data_u['answer'][15 - w]
            u_cid[i, w] = data_u['cid'][15 - w]

        # apply adj to q_mean and lookup again
        dt          = (df['timestamp'].iat[i] - u_t_p[i]) / 1000 / df['bundle_size'].iat[i]
        m           = dur_adj_di[np.minimum(50, int(dt))]
        q_mean      = q_mean * m / (1 + q_mean * (m - 1))
        data_u2     = u_stats.get_item_q(df['user_id'].iat[i], q_mean)
        u_qgXcnt[i] = data_u2['qgXcnt']
        u_AE1qgX[i] = data_u2['AE1qgX']
    df['dt']      = ((np.array(df['timestamp']) - u_t_p) / 1000 / df['bundle_size']).astype('float32')
    df['dt_inc']  = ((np.array(df['timestamp']) - u_t_incor) / 1000).astype('float32')
    # apply durational adjustment
    m             = np.minimum(50, df['dt'].astype('int32')).map(dur_adj_di)
    df['q_mean']  = (df['q_mean'] * m / (1 + df['q_mean'] * (m - 1))).astype('float32')
    df['q_mean']  = np.maximum(0.00001, np.minimum(0.99999, df['q_mean'])) # cap/floor

    df['u_cnt']   = u_cnt.astype('int32')
    df['AE2']     = np.nan_to_num(u_AE2, nan=1).astype('float32')
    df['AE2i']    = np.nan_to_num(u_AE2i, nan=1).astype('float32')
    df['AE2O']    = np.nan_to_num(u_AE2O, nan=1).astype('float32')
    df['AE2m']    = np.nan_to_num(u_AE2m, nan=1).astype('float32')
    df['AE2mi']   = np.nan_to_num(u_AE2mi, nan=1).astype('float32')
    df['pgXcnt']  = np.nan_to_num(u_pgXcnt, nan=0).astype('int32')
    df['AE1OpgX'] = np.nan_to_num(u_AE1OpgX, nan=1).astype('float32')
    df['AE1mEpgX']= np.nan_to_num(u_AE1mEpgX, nan=1).astype('float32')
    df['AE1mipgX']= np.nan_to_num(u_AE1mipgX, nan=1).astype('float32')

    df['pXcnt']   = np.nan_to_num(u_pXcnt, nan=0).astype('int32')
    df['AE1mEpX'] = np.nan_to_num(u_AE1mEpX, nan=1).astype('float32')
    df['AE1mOpX'] = np.nan_to_num(u_AE1mOpX, nan=1).astype('float32')

    df['qgXcnt']  = np.nan_to_num(u_qgXcnt, nan=0).astype('int32')
    df['AE1qgX']  = np.nan_to_num(u_AE1qgX, nan=1).astype('float32')
    
    df['cc_tg0']  = np.minimum(120, u_cc_tg0).astype('int8')
    df['cc_part'] = np.minimum(120, u_cc_part).astype('int8')
  
    df['dt2']     = ((np.array(df['timestamp']) - u_t_p2) / 1000).astype('float32')
    df['dt2']     = np.maximum(-1, (df['dt2'] - np.maximum(0, df['dt'])) / df['bundle_size']).astype('float32')
    df['dt2'].loc[df['u_cnt'] < 2] = -1
    j = 0
    for w in [3, 10, 20]:
        df['n_att'+str(w)] = u_t_n_att[:,j].astype('int16')
        j += 1
    for w in [10, 20]:
        df['n_att0_'+str(w)] = u_t_n_att[:,j].astype('int16')
        j += 1

    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        df['y'+str(j)] = u_y[:, j]
        df['cid'+str(j)] = u_cid[:, j]

    # dt: ratio to historical average
    cap1        = 400
    cap2        = 6
    idx         = df['u_cnt'] < 2
    df['dt_m']  = np.nan_to_num(u_dtm, nan=45).astype('float32')
    df['dt_m'].loc[idx] = 45 # replace with median
    df['dt_u']  = np.minimum(cap2, np.minimum(cap1, df['dt'])/df['dt_m']).astype('float32')
    df['dt_u'].loc[idx] = 1

    # add ratio of dt to median dt by q
    df['dt_q'] = np.maximum(-0.1, df['dt'] / df['mdt']).astype('float32')

    # time: ratio to historical average
    df['dt3_m']  = np.nan_to_num(u_dt3m, nan=45).astype('float32')
    df['dt3_m'].loc[idx] = 45 # replace with median
    df['dt3_u']  = np.minimum(cap2, np.minimum(cap1, df['time']/1000)/df['dt3_m']).astype('float32')
    df['dt3_u'].loc[idx] = 1

    # add ratio of time to median dt by q
    df['dt3_q'] = np.maximum(-0.1, df['time'] / 1000 / df['mdt3']).astype('float32')
        
    df['expl'] = df['expl'].fillna(0).astype('int8')
    df['dt2'] = np.maximum(5, df['dt2'])
    return df






# check if 'user_id' + 'content_id' occured before*****************************************************************
print(' Starting reps_di. Time elapsed %.0f sec'%(time.time()-start_time))
y = df['answered_correctly'] # target
reps_di = {}
keys = df['user_id'].astype('int64') * 20000 + df['content_id'].astype('int64')
size = df.shape[0]
del df
gc.collect()
for i in range(size):
    key   = keys.iat[i]
    data  = (reps_di.get(key) or 0)
    data += 1 + 199 * y.iat[i]
    reps_di[key] = data
del keys, y
gc.collect()
print(' Finished reps_di. Time elapsed %.0f sec'%(time.time()-start_time))





# define user stats object
u_di0 = {'act1': 0, 'exp1': 0, 'act2': 0, 'exp2': 0, 'cnt': 0, 'cnt2': 0
    , 'act1pg1': 0, 'exp1pg1': 0, 'act1pg2': 0, 'exp1pg2': 0, 'cnt1pg1': 0, 'cnt1pg2': 0
    , 'cc_tg0': 0, 'cc_part': 0, 'part': 99, 'tag0': 255, 't_p':1000, 't_p2':1000, 'dtc':0, 'dt3c':0
    , 'AE2mc':0, 'AE2mic':0, 'AE1mEpgc1':0, 'AE1mEpgc2':0, 'AE1mipgc1':0, 'AE1mipgc2':0
    , 'cnt1p1': 0, 'cnt1p2': 0, 'cnt1p3': 0, 'cnt1p4': 0, 'cnt1p5': 0, 'cnt1p6': 0, 'cnt1p7': 0
    , 'AE1mEpc1':0, 'AE1mEpc2':0, 'AE1mEpc3':0, 'AE1mEpc4':0, 'AE1mEpc5':0, 'AE1mEpc6':0, 'AE1mEpc7':0
    , 'AE1mOpc1':0, 'AE1mOpc2':0, 'AE1mOpc3':0, 'AE1mOpc4':0, 'AE1mOpc5':0, 'AE1mOpc6':0, 'AE1mOpc7':0
    , 'act1qg1': 0, 'exp1qg1': 0, 'act1qg2': 0, 'exp1qg2': 0, 'cnt1qg1': 0, 'cnt1qg2': 0, 't_incor': 0
    , 'timestamp':[-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000,-1000000]
    , 'answer':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    , 'cid':   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
class UStats:
    def __init__(self):
        self.stats = {}
        
    def get_item_q(self, item, q):
        d0 = (self.stats.get(item) or {'act1qg1': 0, 'exp1qg1': 0, 'act1qg2': 0, 'exp1qg2': 0, 'cnt1qg1': 0, 'cnt1qg2': 0})
        data = d0.copy() # copy, so that it is not changed inplace
        
        # qg
        pg = 1 + (q < qg_th)
        if data['exp1qg'+str(pg)] == 0:
            data['AE1qgX'] = 1
        else:
            data['AE1qgX'] = data['act1qg'+str(pg)] / data['exp1qg'+str(pg)]
        data['qgXcnt'] = data['cnt1qg'+str(pg)]    
        return data

    def get_item(self, item, part, tag0):
        d0 = (self.stats.get(item) or u_di0)
        data = d0.copy() # copy, so that it is not changed inplace

        # AE2
        if data['cnt'] == 0:
            data['dtm']  = 0
            data['dt3m'] = 0
            data['AE2O'] = 1
            data['AE2']  = 1
            data['AE2i'] = 1
            data['AE2m'] = 1
            data['AE2mi']= 1
        else:
            data['dtm']  = data['dtc'] / np.maximum(1, data['cnt'] - 1)
            data['dt3m'] = data['dt3c'] / np.maximum(1, data['cnt'] - 1)
            data['AE2']  = data['act2'] / data['exp2']
            data['AE2i'] = (data['cnt2'] - data['act2']) / (data['cnt2'] - data['exp2'])
            data['AE2O'] = np.maximum(0.01, data['act2']) / np.maximum(0.01, data['exp2']) / np.maximum(0.01, data['cnt2'] - data['act2']) * np.maximum(0.01, data['cnt2'] - data['exp2'])
            data['AE2m'] = data['AE2mc'] / data['cnt2']
            data['AE2mi']= data['AE2mic'] / data['cnt2']

        # pg
        pg = 1 + (part < 5)
        data['AE1OpgX']  = np.maximum(0.01, data['act1pg'+str(pg)]) / np.maximum(0.01, data['exp1pg'+str(pg)]) / np.maximum(0.01, data['cnt1pg'+str(pg)] - data['act1pg'+str(pg)]) * np.maximum(0.01, data['cnt1pg'+str(pg)] - data['exp1pg'+str(pg)])
        if data['exp1pg'+str(pg)] == 0:
            data['AE1pgX']   = 1
            data['AE1mEpgX'] = -0.56
            data['AE1mipgX'] = 1
        else:
            data['AE1pgX']   = data['act1pg'+str(pg)] / data['exp1pg'+str(pg)]
            data['AE1mEpgX'] = data['AE1mEpgc'+str(pg)] / data['cnt1pg'+str(pg)]
            data['AE1mipgX'] = data['AE1mipgc'+str(pg)] / data['cnt1pg'+str(pg)]
        data['pgXcnt']  = data['cnt1pg'+str(pg)]

        # p
        pg = int(part)
        if data['cnt1p'+str(pg)] == 0:
            data['AE1mEpX'] = -0.56
            data['AE1mOpX'] = 1
        else:
            data['AE1mEpX'] = data['AE1mEpc'+str(pg)] / data['cnt1p'+str(pg)]
            data['AE1mOpX'] = data['AE1mOpc'+str(pg)] / data['cnt1p'+str(pg)]
        data['pXcnt']  = data['cnt1p'+str(pg)]

        #cc_tg0, cc_part
        if part != data['part']:
            data['cc_part'] = 0
        if tag0 != data['tag0']:
            data['cc_tg0'] = 0
        return data

    # this function updates values with 1 record
    def add_item(self, item, act1, exp1, part, tag0, t_p, dt, time, q, cid):
        data = self.stats.get(item)
        if data is None:
            self.stats[item] = u_di0.copy()
        data = self.stats.get(item)
        data['dtc']  += np.minimum(400, np.maximum(0, dt))
        data['dt3c'] += np.minimum(400, np.maximum(0, time/1000))

        # AE1
        data['act1'] += act1
        data['exp1'] += exp1
        data['cnt']  += 1

        # AE2
        data['act2'] = data['act2'] * multAE2 + act1
        data['exp2'] = data['exp2'] * multAE2 + exp1
        data['cnt2'] = data['cnt2'] * multAE2 + 1
        data['AE2mc'] = data['AE2mc'] * multAE2 + act1 / exp1
        data['AE2mic'] = data['AE2mic'] * multAE2 + (1 - act1) / (1 - exp1)
        
        # pg
        E = act1 * np.log(exp1) + (1 - act1) * np.log(1 - exp1)
        pg = 1 + (part < 5)
        data['cnt1pg'+str(pg)]   += 1
        data['act1pg'+str(pg)]   += act1
        data['exp1pg'+str(pg)]   += exp1
        data['AE1mEpgc'+str(pg)] += E
        data['AE1mipgc'+str(pg)] += ( 1 - act1 ) / (1 - exp1)

        # p
        pg = int(part)
        data['cnt1p'+str(pg)]    += 1
        data['AE1mEpc'+str(pg)]  += E
        data['AE1mOpc'+str(pg)]  += np.maximum(0.01, act1 ) * np.maximum(0.01, 1 - exp1) / np.maximum(0.01, exp1) / np.maximum(0.01, 1 - act1)

        # qg
        pg = 1 + (q < qg_th)
        data['cnt1qg'+str(pg)]   += 1
        data['act1qg'+str(pg)]   += act1
        data['exp1qg'+str(pg)]   += exp1

        # cc_
        if part == data['part']:
            data['cc_part'] += 1
        else:
            data['cc_part'] = 1
        if tag0 == data['tag0']:
            data['cc_tg0'] += 1
        else:
            data['cc_tg0'] = 1
        data['part'] = part
        data['tag0'] = tag0

        # t_p/t_p2/t_incor
        if t_p > data['t_p']:
            data['t_p2'] = data['t_p']
        data['t_p']  = t_p # save last value, as is
        if act1 == 0: # incorrect answer, record its time
            data['t_incor'] = t_p

        # timestamp
        data['timestamp'] = list(data['timestamp'][1:])# drop first
        data['timestamp'].append(t_p) # append new

        # answer
        data['answer'] = list(data['answer'][1:])# drop first
        data['answer'].append(act1) # append new

        # cid
        data['cid'] = list(data['cid'][1:])# drop first
        data['cid'].append(cid + 1) # append new; make it start with 1 (for NN)
        
    # this function updates values with 1 record, for lectures only
    def add_item_l(self, item, t_p):# only update timestamp
        data = self.stats.get(item)
        if data is None:
            self.stats[item] = {'t_p':0}
        data = self.stats.get(item)
        if t_p > data['t_p']:
            data['t_p2'] = data['t_p']
        data['t_p']  = t_p # save last value, as is

# convert df_user_id to UStats************************************************************
u_stats = UStats()
for i in range(df_user_id.shape[0]):
    dil = {}
    dil['timestamp']    = list(last_times[i,:])
    dil['answer']       = list(last_answers[i,:])
    dil['cid']          = list(last_cid[i,:])
    for col in df_user_id.columns:
        if col == 'user_id':
            item = df_user_id[col].iat[i]
        else:
            dil[col] = df_user_id[col].iat[i]
        u_stats.stats[item] = dil

# prepare final q dataframe: combine it with other data of the same shape
for col in ['q_mean', 'q_cnt']:
    q[col] = df_content_id[col]
for col in ['cid_f1', 'cid_f2', 'cid_f3', 'cid_f4', 'cid_f5', 'cid_f6', 'cid_f7', 'cid_f8']:
    q[col] = emb[col]
del df_user_id, df_content_id, emb, last_times, last_answers, last_cid
gc.collect()


# prior test data - create placeholder
pr_test_df = pd.DataFrame()
cc = 0 # init counter





# LGB model **********************************
data_columns = ['expl', 'dt', 'dt2', 'dt_q', 'dt3_q', 'q_mean', 'part', 'cc_tg0',
       'cc_part', 'tag0', 'tag1', 'mdt2', 'mdt3', 'q_cnt', 'cid_f1', 'cid_f2',
       'cid_f3', 'cid_f4', 'cid_f5', 'cid_f6', 'cid_f7', 'cid_f8', 'n_att3',
       'n_att10', 'n_att20', 'n_att0_10', 'n_att0_20', 'dt_inc', 'AE1mEpgX',
       'AE1mEpX', 'AE1mipgX', 'AE1mOpX', 'AE1OpgX', 'AE1qgX', 'pXcnt',
       'qgXcnt', 'AE2', 'AE2i', 'AE2O', 'AE2m', 'AE2mi', 'dt_u', 'dt3_m',
       'dt3_u', 'rep0', 'rep1']

modelLGB = lgb.Booster(model_file='/kaggle/input/riiidsub2/model6.txt')






# define NN **********************************
import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, BatchNormalization, Dense
from tensorflow.keras.losses import BinaryCrossentropy

# excl tags from NN, keep the rest
data_columns2 = ['expl', 'dt', 'dt2', 'dt_q', 'dt3_q', 'q_mean', 'part', 'cc_tg0',
       'cc_part', 'mdt2', 'mdt3', 'q_cnt', 'n_att3', 'n_att10',
       'n_att20', 'n_att0_10', 'n_att0_20', 'dt_inc', 'AE1mEpgX', 'AE1mEpX',
       'AE1mipgX', 'AE1mOpX', 'AE1OpgX', 'AE1qgX', 'pXcnt', 'qgXcnt', 'AE2',
       'AE2i', 'AE2O', 'AE2m', 'AE2mi', 'dt_u', 'dt3_m', 'dt3_u', 'rep0',
       'rep1']
num_columns = len(data_columns2)
LL2 = 1e-8
NN_SIZE = 512

# main input - same as LGB
i1 = Input(shape=(num_columns,), dtype='float32')

# content_id embeddings
i2 = Input(shape=(11,), dtype='int16')
i2a = Embedding(input_dim=13524, output_dim=16, input_length=11, embeddings_regularizer=L2(l2=LL2), mask_zero=True)(i2)
i2b = Flatten()(i2a)

# prior answers
i3 = Input(shape=(10,), dtype='float32')

# tag embeddings
i4 = Input(shape=(6,), dtype='int16')
i4a = Embedding(input_dim=256, output_dim=4, input_length=6, embeddings_regularizer=L2(l2=LL2))(i4)
i4b = Flatten()(i4a)

# combine all inputs
d1 = Concatenate(axis=1)([i1, i2b, i3, i4b])
d2 = BatchNormalization()(d1)

# layer 1
d3 = Dense(NN_SIZE, activation='relu', kernel_regularizer=L2(l2=LL2))(d2)
d4 = BatchNormalization()(d3)

# layer 2
d5 = Dense(NN_SIZE, activation='relu', kernel_regularizer=L2(l2=LL2))(d4)
d6 = BatchNormalization()(d5)

# layer 3
d7 = Dense(NN_SIZE, activation='relu', kernel_regularizer=L2(l2=LL2))(d6)
d8 = BatchNormalization()(d7)

# layer 4
d9 = Dense(NN_SIZE, activation='relu', kernel_regularizer=L2(l2=LL2))(d8)
d10 = BatchNormalization()(d9)

# last layer
o = Dense(1, activation='sigmoid', kernel_regularizer=L2(l2=LL2))(d10)

modelNN = tf.keras.Model(inputs=[i1, i2, i3, i4], outputs=o)
modelNN.compile(optimizer=tf.optimizers.Adam(), loss = BinaryCrossentropy(from_logits=True))
modelNN.load_weights('/kaggle/input/riiidsub2/checkpoint')







# submission
for (test_df, sample_prediction_df) in iter_test:


    # populate prior group responces
    cc += 1 # incr counter
    if cc > 1: # skip first iter, it has no priors
        a = test_df['prior_group_answers_correct'].iat[0]
        a = a[1:-1].split(',') # exclude brackets
        for i in range(len(a)):
            a1 = int(a[i])
            # update totals by user, and reps_di
            if pr_test_df['content_type_id'].iat[i] == 0:# skip lectures
                c_key = pr_test_df['content_id'].iat[i]                
                e1 = pr_test_df['q_mean'].iat[i]
                
                # update user stats
                u_key = pr_test_df['user_id'].iat[i]          
                u_stats.add_item(u_key, a1, e1, pr_test_df['part'].iat[i]
                    , pr_test_df['tag0'].iat[i], pr_test_df['timestamp'].iat[i], pr_test_df['dt'].iat[i]
                    , pr_test_df['time'].iat[i], pr_test_df['q_mean'].iat[i], pr_test_df['content_id'].iat[i])

                # update repeats
                key = u_key.astype('int64') * 20000 + c_key.astype('int64')
                data = (reps_di.get(key) or 0)
                data += 1 + 199 * a1
                reps_di[key] = data
            else: # lectures only: update u stats for lectures
                u_stats.add_item_l(pr_test_df['user_id'].iat[i], pr_test_df['timestamp'].iat[i])


    # add q columns
    test_df = test_df.merge(q, how='left', on='content_id', right_index=True)
    test_df['part'] = test_df['part'].fillna(1).astype('int8')
    test_df['tag0'] = test_df['tag0'].fillna(38).astype('int16')
    test_df['bundle_size'] = test_df['bundle_size'].fillna(1).astype('int8')

    # save it
    pr_test_df = test_df.copy()
    
    # drop lectures
    idx = test_df['content_type_id'] == 0
    df = test_df.loc[idx].reset_index(drop=True).copy() # need to drop lectures before FE!
    df.rename(columns={'prior_question_elapsed_time': 'time', 'prior_question_had_explanation': 'expl'}, inplace=True)

    # repeats
    dd = np.zeros(df.shape[0], dtype=np.int16)
    keys = df['user_id'].astype('int64') * 20000 + df['content_id'].astype('int64')
    for i in range(df.shape[0]):
        key   = keys.iat[i]
        data  = (reps_di.get(key) or 0)
        dd[i] = data # save result
    df['rep0'] = (dd % 200).astype('int8')
    df['rep1'] = (dd // 200).astype('int8')

    # FE
    df = FE(df)

    # need these for prior value updates
    pr_test_df['dt'] = 0
    pr_test_df['dt'].loc[idx] = df['dt'].fillna(0).astype('float32').values
    pr_test_df['time'] = 0
    pr_test_df['time'].loc[idx] = df['time'].fillna(0).astype('float32').values
    pr_test_df['q_mean'] = 0
    pr_test_df['q_mean'].loc[idx] = df['q_mean'].fillna(AE_TOT).astype('float32').values


    
    # predict LGB*****************************************************************************
    pred1 = modelLGB.predict(np.array(df[data_columns].fillna(0), dtype=np.float32), num_iteration=modelLGB.best_iteration)



    # predict NN*****************************************************************************
    df2 = df.copy()
    # undo logit on q
    df2['q_mean'] = np.log(df2['q_mean'] / (1 - df2['q_mean']))
    # log-transform counts and some other cols with large values
    for col in ['q_cnt', 'pXcnt', 'qgXcnt', 'cc_part', 'cc_tg0', 'dt', 'dt2', 'dt_inc', 'rep0', 'rep1']:
        df2[col] = np.log(1 + np.maximum(-0.5, df2[col])).astype('float32')
    df['content_id'] = df['content_id'] + 1 # incr this - delayed ones are already incremented
    # predict
    pred2 = np.array(modelNN([np.array(df2[data_columns2].fillna(0), dtype=np.float32)
                            , np.array(df[['content_id', 'cid1', 'cid2', 'cid3', 'cid4', 'cid5', 'cid6', 'cid7', 'cid8', 'cid9', 'cid10']].fillna(0), dtype=np.int16)
                            , np.array(df[['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10']].fillna(1), dtype=np.int8)
                            , np.array(df[['tag0', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5']].fillna(255), dtype=np.int16)]
                            , training=False)
                     ).reshape(df2.shape[0])


    # blend
    pred = pred1 * 0.4 + 0.6 * pred2
    

    # submit
    test_df['answered_correctly'] = 0.5
    test_df['answered_correctly'].loc[idx] = pred



    # submit
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
print('finished! Time elapsed %.1f sec'%(time.time()-start_time))
