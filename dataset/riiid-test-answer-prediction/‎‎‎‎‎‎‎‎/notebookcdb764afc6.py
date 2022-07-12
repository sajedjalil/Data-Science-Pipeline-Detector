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

# https://www.kaggle.com/kneroma/riid-user-and-content-mean-predictor


# training
if 0:
    df_train = read_pickle_from_file(data_dir + '/train.pkl')
    df_train = df_train[['user_id', 'content_id', 'answered_correctly']]
    print(df_train.shape)
    df_train = df_train[df_train['answered_correctly'] != -1]
    print(df_train.shape)

    gb = df_train.groupby('user_id', as_index=False)['answered_correctly'].mean()
    user_model = gb.set_index('user_id').to_dict()['answered_correctly']

    gb = df_train.groupby('content_id', as_index=False)['answered_correctly'].mean()
    content_model = gb.set_index('content_id').to_dict()['answered_correctly']
else:
    user_model = read_pickle_from_file('../input/myriiid/user_model.pickle')
    content_model = read_pickle_from_file('../input/myriiid/content_model.pickle')
 

user_model = defaultdict(lambda : 0.5, user_model)
content_model = defaultdict(lambda : 0.5, content_model)
    


def do_predict(user_id,content_id):
    p_user = user_model.get(user_id, 0.5)
    p_content = content_model.get(content_id, 0.5)
    probability = (p_user+p_content)/2
    return probability


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
        p = do_predict(r['user_id'],r['content_id'])
        probability.append(p)
        truth.append(t)
        print('\r',i,p, end='',flush=True)
    print('')

    probability = np.array(probability)
    truth = np.array(truth)
    auc = compute_kaggle_auc(probability, truth)
    print('auc', auc)

#exit(0)



#----------------------------------------
# test submission
# https://www.kaggle.com/sohier/competition-api-detailed-introduction


def do_predict_df(test_df, sample_prediction_df):
    if len(sample_prediction_df)==0: return sample_prediction_df
    
    d = test_df[test_df['content_type_id'] == 0]
    s0 = d['user_id'].map(user_model) 
    s1 = d['content_id'].map(content_model) 
    sample_prediction_df['answered_correctly'] = (s0+s1)/2
    return sample_prediction_df


iter_test = env.iter_test()
for t, (test_df, sample_prediction_df) in enumerate(iter_test):

    predict_df = do_predict_df(test_df, sample_prediction_df)
    env.predict(predict_df)
    print('loop at %d :'%(t), test_df.shape, sample_prediction_df.shape, predict_df.shape)

print(predict_df)
print('iter_test sucessful!')

