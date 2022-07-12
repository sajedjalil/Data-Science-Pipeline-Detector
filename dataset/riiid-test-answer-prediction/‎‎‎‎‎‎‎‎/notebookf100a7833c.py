import sys
if 0: #local
    sys.path.insert(0, '/root/share1/kaggle/2020/riiid/data/riiid-test-answer-prediction')
    data_dir = '/root/share1/kaggle/2020/riiid/data/riiid-test-answer-prediction'
else: #kaggle kernel
    data_dir = '../input/myriiid'

import riiideducation
env = riiideducation.make_env()

#from common import *
import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
from collections import defaultdict


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x


def compute_kaggle_auc(probability, truth):
    fpr, tpr, thresholds = metrics.roc_curve(truth, probability)
    auc = metrics.auc(fpr, tpr)
    return auc


'''
df_test = pd.read_csv(data_dir + '/example_test.csv')
df_test.columns
Index(['row_id', 'group_num', 'timestamp', 'user_id', 'content_id',
       'content_type_id', 'task_container_id', 'prior_question_elapsed_time',
       'prior_question_had_explanation', 'prior_group_answers_correct',
       'prior_group_responses'],
      dtype='object')



df_train.columns
Index(['timestamp', 'user_id', 'content_id', 'content_type_id',
       'task_container_id', 'user_answer', 'answered_correctly',
       'prior_question_elapsed_time', 'prior_question_had_explanation'],
      dtype='object')
'''

########################################################################################################
'''
https://www.kaggle.com/hengck23/notebookcdb764afc6?scriptVersionId=44475365
average prediction baseline : LB 0.741
(local cv  auc 0.7619972542330822)
'''


# https://www.kaggle.com/kneroma/riid-user-and-content-mean-predictor


# training
df_train = read_pickle_from_file(data_dir + '/train.pkl')
df_train = df_train[['user_id', 'content_id', 'answered_correctly']]
print(df_train.shape)
df_train = df_train[df_train['answered_correctly'] != -1]
print(df_train.shape)

user_model = df_train.groupby('user_id', as_index=True).agg(
             sum  =('answered_correctly', 'sum'),
             count=('answered_correctly', 'count'),
)

content_model = df_train.groupby('content_id', as_index=True).agg(
             sum  =('answered_correctly', 'sum'),
             count=('answered_correctly', 'count'),
)
# write_pickle_to_file('/root/share1/kaggle/2020/riiid/result/user_model.pickle',user_model)
# write_pickle_to_file('/root/share1/kaggle/2020/riiid/result/content_model.pickle',content_model)
# exit(0)


# debug only
#
# gb = df_train.groupby('user_id', as_index=False)['answered_correctly'].mean()
# d_user_model = gb.set_index('user_id').to_dict()['answered_correctly']

def predict_user_score(user_id):
    #print(user_id)
    query = user_model[user_model.index==user_id]
    if query.empty :
        p_user = 0.5
    else:
        sum, count = user_model.loc[user_id]
        p_user = sum/count
    return p_user


def predict_content_score(content_id):
    query = content_model[content_model.index==content_id]
    if query.empty :
        p_content = 0.5
    else:
        sum, count = content_model.loc[content_id]
        p_content = sum/count
    return p_content


def update_model(test_df, prior_test_df):
    if prior_test_df is None: return

    mask = (prior_test_df.content_type_id==0).values
    if sum(mask)==0: return

    #---

    prior_group_responses = eval(test_df['prior_group_responses'].values[0])
    prior_group_answers_correct = eval(test_df['prior_group_answers_correct'].values[0])
    if len(prior_group_answers_correct) == 0: return

    prior_group_answers_correct = np.array(prior_group_answers_correct)[mask]
    if len(prior_group_answers_correct) == 0: return

    #---
    prior_user_id    = prior_test_df['user_id'].values[mask]
    prior_content_id = prior_test_df['content_id'].values[mask]

    #<todo> inefficient and dirty code !!!!!
    for user_id, answer_correct in zip(prior_user_id, prior_group_answers_correct):
        query = user_model[user_model.index==user_id]
        if query.empty :
            #print('adding user_id ...',user_id)
            user_model.loc[user_id] = [answer_correct, 1] # sum  count
        else:
            #print('updating user_id ...',user_id)
            user_model.loc[user_id, 'sum'] += answer_correct
            user_model.loc[user_id, 'count'] += 1

    #---
    for content_id, answer_correct in zip(prior_content_id, prior_group_answers_correct):
        query = content_model[content_model.index==content_id]
        if query.empty :
            #print('adding content_id ...',content_id)
            content_model.loc[content_id] = [answer_correct, 1] # sum  count
        else:
            #print('updating content_id ...',content_id)
            content_model.loc[content_id, 'sum'] += answer_correct
            content_model.loc[content_id, 'count'] += 1

    pass


#----------------------------------------

#dummy validation
if 0:
    user_id = df_train.user_id.unique()
    np.random.shuffle(user_id)
    user_id = user_id[:100]

    df_valid = df_train[df_train['user_id'].isin(user_id)].reset_index(drop=True)
    print(df_valid.shape)

    probability =[]
    truth =[]
    for i in range(len(df_valid)):
        r = df_valid.iloc[i]
        t = r['answered_correctly']
        p = 0.5*predict_user_score(r['user_id']) + \
            0.5*predict_content_score(r['content_id'])

        probability.append(p)
        truth.append(t)
        print('\r',i,p, end='',flush=True)
    print('')

    probability = np.array(probability)
    truth = np.array(truth)
    auc = compute_kaggle_auc(probability, truth)
    print('auc', auc)
    #exit(0)
    '''
    auc 0.7619972542330822
    lb = 0.741
    '''



#----------------------------------------
# test submission
# https://www.kaggle.com/sohier/competition-api-detailed-introduction


def do_predict_df(test_df):
    d = test_df[test_df['content_type_id'] == 0]
    predict_df = d[['row_id']]

    s0 = d['user_id'].map(predict_user_score)
    s1 = d['content_id'].map(predict_content_score)
    predict_df['answered_correctly'] = (s0+s1)/2
    return predict_df

#----
'''
Index(['row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id',
       'task_container_id', 'prior_question_elapsed_time',
       'prior_question_had_explanation', 'prior_group_answers_correct',
       'prior_group_responses'],
      dtype='object')

'''


prior_test_df = None
iter_test = env.iter_test()
for t, (test_df, sample_prediction_df) in enumerate(iter_test):
    predict_df = do_predict_df(test_df)
    env.predict(predict_df)

    print(sample_prediction_df)
    print(predict_df)
    print('loop at %d :'%(t), test_df.shape, sample_prediction_df.shape, predict_df.shape)
    #---

    update_model(test_df, prior_test_df)
    prior_test_df = test_df

print(predict_df)
print('iter_test sucessful!')

