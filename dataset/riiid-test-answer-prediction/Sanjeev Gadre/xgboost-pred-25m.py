# %% [code]
# %% [code]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 08:59:37 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import riiideducation
import os

env = riiideducation.make_env()
iter_test = env.iter_test()

#%% Helper Variables
DATAPATH = '/kaggle/input/riiid-feature-engg/'

#%% Helper Functions
# OneHotEncoder to encode the questions part number
part_enc = OneHotEncoder(categories = [np.arange(1, 8, 1)], dtype = 'int', sparse = False)

#%% Get Scoring data
userscores = np.genfromtxt(DATAPATH + 'userscores.csv', delimiter = ',')
ques = np.genfromtxt(DATAPATH + 'ques.csv', delimiter = ',')
model = xgb.Booster(model_file = DATAPATH + 'xgbmodel.bin')

# mean userscores to use when a new user is encountered
mean_userscores = np.mean(userscores[:, 1:8], axis = 0).reshape(1, -1)
'''
train_cols = ['answered_correctly', 'prior_question_had_explanation', 'score_1',
              'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7',
              'part', 'correct_attempt_prob']
'''

#%% Process Test data
''' 
test_cols = ['row_id', 'timestamp', 'user_id', 'content_id', 'content_type_id',
             'task_container_id', 'prior_question_elapsed_time',
             'prior_question_had_explanation', 'prior_group_answers_correct',
             'prior_group_responses']
'''

for (test, sample_submission) in iter_test:
    # Identify row index that are questions
    mask = test['content_type_id'] == 0
    # Use copy of test for predictions
    test_c = test.copy()
    # Drop unwanted rows
    test_c = test_c.drop(index = test_c.loc[test_c['content_type_id'] == 1, :].index)
    # Drop unwanted columns
    test_c = test_c.drop(columns = ['timestamp', 'content_type_id', 'task_container_id',
                                    'prior_question_elapsed_time', 'prior_group_answers_correct',
                                    'prior_group_responses'])
    # test_c_cols : ['row_id', 'user_id', 'content_id', 'prior_question_had_explanation']
    # Eliminating nans
    test_c.loc[test_c['prior_question_had_explanation'].isna(), 'prior_question_had_explanation'] = False
    test_c['prior_question_had_explanation'] = test_c['prior_question_had_explanation'].astype('int')
    # Convert to numpy array
    test_c = test_c.to_numpy()
    '''
    Add new columns required --> ['score_1', score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 
                                  'part', 'correct_attempt_prob'] i.e. 9 new columns
    '''
    
    test_c = np.concatenate((test_c, np.zeros((test_c.shape[0], 9))), axis = 1)
    '''
    test_c_cols : ['row_id', 'user_id', 'content_id', 'prior_question_had_explanation',
                   'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7', 'part',
                   'correct_attempt_prob']
    '''
    # To the test_c dataset add the appropriate part, correct_attempt_prob and userscores
    for i in range(len(test_c)):
        # part and correct_attempt_prob
        test_c[i, 11:] = ques[np.where(ques[:, 0] == test_c[i, 2])[0], [1, 3]]
        # userscores
        # is the user a "new" user
        if np.isin(test_c[i, 1], userscores[:, 0], assume_unique = True):
            test_c[i, 4:11] = userscores[np.where(userscores[:, 0] == test_c[i, 1])[0], 1:8]
        else:
            test_c[i, 4:11] = mean_userscores

    # OneHotEncode the part number
    encoded_part = part_enc.fit_transform(test_c[:, 11].reshape(-1, 1))
    # Drop the user_id, content_id and part_number columns and add the encoded part columns
    test_c = np.delete(test_c, [1, 2, 11], 1)
    test_c = np.concatenate((test_c, encoded_part), axis = 1)
    '''
    test_c_cols : ['row_id', 'prior_question_had_explanation',
                   'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'score_6', 'score_7',
                   'correct_attempt_prob', 'part_1', 'part_2', 'part_3', 'part_4', 'part_5',
                   'part_6', 'part_7']'''
    
    # Make predictions
    dtest_c = xgb.DMatrix(test_c[:, 1:])
    probs = model.predict(dtest_c)[:, 1]
    
    # Update probabilities to the test set
    sample_submission.loc[mask, 'answered_correctly'] = probs
    # Make the prediction
    env.predict(sample_submission.loc[mask, :])




