#!/usr/bin/env python

"""This script summarize elapsed time by each device for each user in sessions.csv, and 
   merge the information with train_users.csv and test_users.csv DataFrame (join by user id)
"""

import pandas as pd


def calculate_pct(device_type_time, total_time):
      return device_type_time/total_time if total_time > 0 else None
      

def get_user_device_type_time(fpath='../input/sessions.csv'):

    sessions = pd.read_csv(fpath)
    # sum up secs_elapsed time on each device_type for each user
    device_type_time = pd.pivot_table(sessions, index=['user_id'], columns=['device_type'], values='secs_elapsed', aggfunc=sum, fill_value=0)
    device_type_time.reset_index(inplace=True)
    # sum up elapsed time on all the devices for each user
    device_type_time['total_elapsed_time'] = device_type_time.sum(axis=1)
    
    # add attributes for usage percentage of each device type
    device_columns = device_type_time.columns[1:-2]  # exclude first column: user_id and last column: total_elapsed_time
    for column in device_columns:
        device_type_time[column+'_pct'] = device_type_time.apply(lambda row: calculate_pct(row[column], row['total_elapsed_time']), axis=1)
    
    
    print(device_type_time[device_type_time.total_elapsed_time > 0].head())

    return device_type_time


def merge_user_and_session_data(user_df, user_device_type_time_df=None):

    if not isinstance(user_device_type_time_df, pd.DataFrame):
        user_device_type_time_df = get_user_device_type_time()

    users_combined_df = pd.merge(user_df, user_device_type_time_df, left_on='id', right_on='user_id', how='left')
    return users_combined_df


def main():

    train_users = pd.read_csv('../input/train_users.csv')
    predict_users = pd.read_csv('../input/test_users.csv')

    user_device_type_time_df = get_user_device_type_time()
    print(type(user_device_type_time_df))
    print(user_device_type_time_df[user_device_type_time_df.total_elapsed_time > 0].head())

    train_users_combined = merge_user_and_session_data(train_users, user_device_type_time_df)
    predict_users_combined = merge_user_and_session_data(predict_users, user_device_type_time_df)
    

    print(train_users_combined[train_users_combined.total_elapsed_time > 0].head())
    print(predict_users_combined[predict_users_combined.total_elapsed_time > 0].head())
    
    # how many users have session data
    print(train_users_combined[train_users_combined.total_elapsed_time > 0].shape[0], train_users_combined.shape[0])
    print(predict_users_combined[predict_users_combined.total_elapsed_time > 0].shape[0], predict_users_combined.shape[0])


main()

