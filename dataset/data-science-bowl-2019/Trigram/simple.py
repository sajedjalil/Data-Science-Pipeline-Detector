#Nalchik city

# All credit goes to Ilyas and his notebook 'simple lgb model'. Please upvote that kernel.
from tqdm import tqdm
from collections import Counter
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import json
import pandas as pd
import numpy as np
import warnings
import random
import time
import os
import multiprocessing
from multiprocessing import Lock, Process, Queue, current_process
import scipy as sp
from functools import partial
from numba import jit
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

def read_data():
    start = time.time()
    print("Start read data")

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    #train = pd.read_csv('../input/data-science-bowl-2019/train.csv', nrows=1200000)
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    #test = pd.read_csv('../input/data-science-bowl-2019/test.csv', nrows=30000)

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    print("read data done, time - ", time.time() - start)
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    start = time.time()

    print("Start encoding data")
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    train['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['world']))
    test['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['world']))
    all_type_world = list(set(train["type_world"].unique()).union(test["type_world"].unique()))

    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
        set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    print("End encoding data, time - ", time.time() - start)


    event_data = {}
    event_data["train_labels"] = train_labels
    event_data["win_code"] = win_code
    event_data["list_of_user_activities"] = list_of_user_activities
    event_data["list_of_event_code"] = list_of_event_code
    event_data["activities_labels"] = activities_labels
    event_data["assess_titles"] = assess_titles
    event_data["list_of_event_id"] = list_of_event_id
    event_data["all_title_event_code"] = all_title_event_code
    event_data["activities_map"] = activities_map
    event_data["all_type_world"] = all_type_world

    return train, test, event_data

def get_all_features(feature_dict, ac_data):
    if len(ac_data['durations']) > 0:
        feature_dict['installation_duration_mean'] = np.mean(ac_data['durations'])
        feature_dict['installation_duration_sum'] = np.sum(ac_data['durations'])
    else:
        feature_dict['installation_duration_mean'] = 0
        feature_dict['installation_duration_sum'] = 0

    return feature_dict


def get_data(user_sample, event_data, test_set):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_assesment = {}

    last_activity = 0

    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy': 0,
                            'Mushroom Sorter (Assessment)_4020_accuracy': 0,
                            'Bird Measurer (Assessment)_4020_accuracy': 0,
                            'Chest Sorter (Assessment)_4020_accuracy': 0}

    game_time_dict = {'Clip_gametime': 0, 'Game_gametime': 0,
                      'Activity_gametime': 0, 'Assessment_gametime': 0}

    last_session_time_sec = 0
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0

    # Newly added features
    accumulated_game_miss = 0
    Cauldron_Filler_4025 = 0
    mean_game_round = 0
    mean_game_duration = 0
    mean_game_level = 0
    Assessment_mean_event_count = 0
    Game_mean_event_count = 0
    Activity_mean_event_count = 0
    chest_assessment_uncorrect_sum = 0

    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    durations_game = []
    durations_activity = []
    last_accuracy_title = {'acc_' + title: -1 for title in event_data["assess_titles"]}
    last_game_time_title = {'lgt_' + title: 0 for title in event_data["assess_titles"]}
    ac_game_time_title = {'agt_' + title: 0 for title in event_data["assess_titles"]}
    ac_true_attempts_title = {'ata_' + title: 0 for title in event_data["assess_titles"]}
    ac_false_attempts_title = {'afa_' + title: 0 for title in event_data["assess_titles"]}
    event_code_count: dict[str, int] = {ev: 0 for ev in event_data["list_of_event_code"]}
    event_code_proc_count = {str(ev) + "_proc" : 0. for ev in event_data["list_of_event_code"]}
    event_id_count: dict[str, int] = {eve: 0 for eve in event_data["list_of_event_id"]}
    title_count: dict[str, int] = {eve: 0 for eve in event_data["activities_labels"].values()}
    title_event_code_count: dict[str, int] = {t_eve: 0 for t_eve in event_data["all_title_event_code"]}
    type_world_count: dict[str, int] = {w_eve: 0 for w_eve in event_data["all_type_world"]}
    session_count = 0

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = event_data["activities_labels"][session_title]

        if session_type == "Activity":
            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1]) / 2.0

        if session_type == "Game":
            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1]) / 2.0

            game_s = session[session.event_code == 2030]
            misses_cnt = cnt_miss(game_s)
            accumulated_game_miss += misses_cnt

            try:
                game_round = json.loads(session['event_data'].iloc[-1])["round"]
                mean_game_round = (mean_game_round + game_round) / 2.0
            except:
                pass

            try:
                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]
                mean_game_duration = (mean_game_duration + game_duration) / 2.0
            except:
                pass

            try:
                game_level = json.loads(session['event_data'].iloc[-1])["level"]
                mean_game_level = (mean_game_level + game_level) / 2.0
            except:
                pass

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {event_data["win_code"][session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(title_count.copy())
            features.update(game_time_dict.copy())
            features.update(event_id_count.copy())
            features.update(title_event_code_count.copy())
            features.update(assess_4020_acc_dict.copy())
            features.update(type_world_count.copy())
            features.update(last_game_time_title.copy())
            features.update(ac_game_time_title.copy())
            features.update(ac_true_attempts_title.copy())
            features.update(ac_false_attempts_title.copy())

            features.update(event_code_proc_count.copy())
            features['installation_session_count'] = session_count
            features['accumulated_game_miss'] = accumulated_game_miss
            features['mean_game_round'] = mean_game_round
            features['mean_game_duration'] = mean_game_duration
            features['mean_game_level'] = mean_game_level
            features['Assessment_mean_event_count'] = Assessment_mean_event_count
            features['Game_mean_event_count'] = Game_mean_event_count
            features['Activity_mean_event_count'] = Activity_mean_event_count
            features['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum

            variety_features = [('var_event_code', event_code_count),
                                ('var_event_id', event_id_count),
                                ('var_title', title_count),
                                ('var_title_event_code', title_event_code_count),
                                ('var_type_world', type_world_count)]

            for name, dict_counts in variety_features:
                arr = np.array(list(dict_counts.values()))
                features[name] = np.count_nonzero(arr)

            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts

            # ----------------------------------------------
            ac_true_attempts_title['ata_' + session_title_text] += true_attempts
            ac_false_attempts_title['afa_' + session_title_text] += false_attempts

            last_game_time_title['lgt_' + session_title_text] = session['game_time'].iloc[-1]
            ac_game_time_title['agt_' + session_title_text] += session['game_time'].iloc[-1]
            # ----------------------------------------------

            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
                features['duration_std'] = 0
                features['last_duration'] = 0
                features['duration_max'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
                features['last_duration'] = durations[-1]
                features['duration_max'] = np.max(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

            if durations_game == []:
                features['duration_game_mean'] = 0
                features['duration_game_std'] = 0
                features['game_last_duration'] = 0
                features['game_max_duration'] = 0
            else:
                features['duration_game_mean'] = np.mean(durations_game)
                features['duration_game_std'] = np.std(durations_game)
                features['game_last_duration'] = durations_game[-1]
                features['game_max_duration'] = np.max(durations_game)

            if durations_activity == []:
                features['duration_activity_mean'] = 0
                features['duration_activity_std'] = 0
                features['game_activity_duration'] = 0
                features['game_activity_max'] = 0
            else:
                features['duration_activity_mean'] = np.mean(durations_activity)
                features['duration_activity_std'] = np.std(durations_activity)
                features['game_activity_duration'] = durations_activity[-1]
                features['game_activity_max'] = np.max(durations_activity)

            # the accuracy is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            # --------------------------
            features['Cauldron_Filler_4025'] = Cauldron_Filler_4025 / counter if counter > 0 else 0

            Assess_4025 = session[(session.event_code == 4025) & (session.title == 'Cauldron Filler (Assessment)')]
            true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()
            false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()

            cau_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (
                                                                                                      true_attempts_ + false_attempts_) != 0 else 0
            Cauldron_Filler_4025 += cau_assess_accuracy_

            chest_assessment_uncorrect_sum += len(session[session.event_id == "df4fe8b6"])

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1]) / 2.0
            # ----------------------------
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                last_assesment = features.copy()

            if true_attempts + false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        if session_type == 'Game':
            durations_game.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

        if session_type == 'Activity':
            durations_activity.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)

        session_count += 1

        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(session[col])
            for k in num_of_session_count.keys():
                x = k
                if col == 'title':
                    x = event_data["activities_labels"][k]
                counter[x] += num_of_session_count[k]
            return counter

        def update_proc(count: dict):
            res = {}
            for k, val in count.items():
                res[str(k) + "_proc"] = (float(val) * 100.0) / accumulated_actions
            return res

        event_code_count = update_counters(event_code_count, "event_code")


        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')
        type_world_count = update_counters(type_world_count, 'type_world')

        assess_4020_acc_dict = get_4020_acc(session, assess_4020_acc_dict, event_data)
        game_time_dict[session_type + '_gametime'] = (game_time_dict[session_type + '_gametime'] + (
                    session['game_time'].iloc[-1] / 1000.0)) / 2.0

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        event_code_proc_count = update_proc(event_code_count)

        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type

            # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return last_assesment, all_assessments
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def cnt_miss(df):
    cnt = 0
    for e in range(len(df)):
        x = df['event_data'].iloc[e]
        y = json.loads(x)['misses']
        cnt += y
    return cnt

def get_4020_acc(df, counter_dict, event_data):
    for e in ['Cauldron Filler (Assessment)', 'Bird Measurer (Assessment)',
              'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)']:
        Assess_4020 = df[(df.event_code == 4020) & (df.title == event_data["activities_map"][e])]
        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()
        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()

        measure_assess_accuracy_ = true_attempts_ / (true_attempts_ + false_attempts_) if (
                                                                                                      true_attempts_ + false_attempts_) != 0 else 0
        counter_dict[e + "_4020_accuracy"] += (counter_dict[e + "_4020_accuracy"] + measure_assess_accuracy_) / 2.0

    return counter_dict

def get_users_data(users_list, return_dict,  event_data, test_set):
    if test_set:
        for user in users_list:
            return_dict.append(get_data(user, event_data, test_set))
    else:
        answer = []
        for user in users_list:
            answer += get_data(user, event_data, test_set)
        return_dict += answer

def get_data_parrallel(users_list, event_data, test_set):
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    threads_number = event_data["process_numbers"]
    data_len = len(users_list)
    processes = []
    cur_start = 0
    cur_stop = 0
    for index in range(threads_number):
        cur_stop += (data_len-1) // threads_number

        if index != (threads_number - 1):
            p = Process(target=get_users_data, args=(users_list[cur_start:cur_stop], return_dict, event_data, test_set))
        else:
            p = Process(target=get_users_data, args=(users_list[cur_start:], return_dict, event_data, test_set))

        processes.append(p)
        cur_start = cur_stop

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    return list(return_dict)

def get_train_and_test(train, test, event_data):
    start = time.time()
    print("Start get_train_and_test")

    compiled_train = []
    compiled_test = []

    user_train_list = []
    user_test_list = []

    stride_size = event_data["strides"]
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):
        user_train_list.append(user_sample)
        if (i + 1) % stride_size == 0:
            compiled_train += get_data_parrallel(user_train_list, event_data, False)
            del user_train_list
            user_train_list = []

    if len(user_train_list) > 0:
        compiled_train += get_data_parrallel(user_train_list, event_data, False)
        del user_train_list

    for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby('installation_id', sort=False)), total=1000):
        user_test_list.append(user_sample)
        if (i + 1) % stride_size == 0:
            compiled_test += get_data_parrallel(user_test_list, event_data, True)
            del user_test_list
            user_test_list = []

    if len(user_test_list) > 0:
        compiled_test += get_data_parrallel(user_test_list, event_data, True)
        del user_test_list

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = [x[0] for x in compiled_test]

    reduce_train_from_test = []
    for i in [x[1] for x in compiled_test]:
        reduce_train_from_test += i

    reduce_test = pd.DataFrame(reduce_test)
    reduce_train_from_test = pd.DataFrame(reduce_train_from_test)
    print("End get_train_and_test, time - ", time.time() - start)
    return reduce_train, reduce_test, reduce_train_from_test

def get_train_and_test_single_proc(train, test, event_data):
    compiled_train = []
    compiled_test = []
    compiled_test_his = []
    for ins_id, user_sample in tqdm(train.groupby('installation_id', sort=False), total=17000):
        compiled_train += get_data(user_sample, event_data, False)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):
        test_data = get_data(user_sample, event_data, True)
        compiled_test.append(test_data[0])
        compiled_test_his += test_data[1]


    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    reduce_test_his = pd.DataFrame(compiled_test_his)

    return reduce_train, reduce_test, reduce_test_his


# thank to Bruno
def eval_qwk_lgb_regr(y_pred, train_t):
    """
    Fast cappa eval function for lgb.
    """
    dist = Counter(train_t['accuracy_group'])
    for k in dist:
        dist[k] /= len(train_t)

    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    #bound = [1.122, 1.739, 2.225]
    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred)))

    return y_pred

def predict(sample_submission, y_pred):
    sample_submission['accuracy_group'] = y_pred
    sample_submission['accuracy_group'] = sample_submission['accuracy_group'].astype(int)
    sample_submission.to_csv('submission.csv', index=False)
    print(sample_submission['accuracy_group'].value_counts(normalize=True))


def get_random_assessment(reduce_train):
    used_idx = []
    for iid in tqdm(set(reduce_train['installation_id'])):
        list_ = list(reduce_train[reduce_train['installation_id']==iid].index)
        cur = random.choices(list_, k = 1)[0]
        used_idx.append(cur)
    reduce_train_t = reduce_train.loc[used_idx]
    return reduce_train_t, used_idx


# function to exclude columns from the train and test set if the mean is different, also adjust test column by a factor to simulate the same distribution
def exclude(reduce_train, reduce_test, features):
    to_exclude = []
    ajusted_test = reduce_test.copy()
    for feature in features:
        if feature not in ['accuracy_group', 'installation_id', 'session_title']:
            data = reduce_train[feature]
            train_mean = data.mean()
            data = ajusted_test[feature]
            test_mean = data.mean()
            try:
                ajust_factor = train_mean / test_mean
                if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                    to_exclude.append(feature)
                    print(feature)
                else:
                    ajusted_test[feature] *= ajust_factor
            except:
                to_exclude.append(feature)
                print(feature)
    return to_exclude, ajusted_test


def remove_correlated_features(reduce_train, features):
    counter = 0
    to_remove = []
    for feat_a in features:
        for feat_b in features:
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]
                if c > 0.995:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    return to_remove

def main():
    in_kaggle = False
    random.seed(42)
    start_program = time.time()

    event_data = {}
    if in_kaggle:
        event_data["strides"] = 300
        event_data["process_numbers"] = 4
    else:
        event_data["strides"] = 300
        event_data["process_numbers"] = 3

    # read data
    train, test, train_labels, specs, sample_submission = read_data()
    # get usefull dict with maping encode
    train, test, event_data_update = encode_title(train, test, train_labels)
    event_data.update(event_data_update)

    #reduce_train, reduce_test, reduce_train_from_test = get_train_and_test_single_proc(train, test, event_data)
    reduce_train, reduce_test, reduce_train_from_test = get_train_and_test(train, test, event_data)
    dels = [train, test]
    del dels

    reduce_train.to_csv('reduce_train.csv', index=False, sep=";")
    reduce_test.to_csv('reduce_test.csv', index=False, sep=";")
    reduce_train_from_test.to_csv('reduce_train_from_test.csv', index=False, sep=";")


    reduce_train = pd.read_csv('reduce_train.csv', sep=";")
    reduce_test = pd.read_csv('reduce_test.csv', sep=";")
    reduce_train_from_test =  pd.read_csv('reduce_train_from_test.csv', sep=";")
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')


    reduce_train.sort_values("installation_id", axis=0, ascending=True, inplace=True, na_position='last')
    reduce_test.sort_values("installation_id", axis=0, ascending=True, inplace=True, na_position='last')

    reduce_train = pd.concat([reduce_train, reduce_train_from_test], ignore_index=True)

    old_features = list(reduce_train.columns[0:99]) + list(reduce_train.columns[886:])
    el_features = ['accuracy_group', 'accuracy', 'installation_id']
    old_features = [col for col in old_features if col not in el_features]
    event_id_features = list(reduce_train.columns[99:483])
    title_event_code_cross = list(reduce_train.columns[483:886])
    features = old_features + event_id_features + title_event_code_cross

    to_remove = remove_correlated_features(reduce_train, features)
    features = [col for col in features if col not in to_remove]

    features = [col for col in features if col not in ['Heavy, Heavier, Heaviest_2000', 'Heavy, Heavier, Heaviest']]
    features.append('installation_id')
    print('Training with {} features'.format(len(features)))

    to_exclude, ajusted_test = exclude(reduce_train, reduce_test, features)
    features = [col for col in features if col not in to_exclude]

    my_model = MyModel(reduce_train, features, kmodels=6, kfold=6)
    train_pred = my_model.predict(reduce_train)

    optR = OptimizedRounder()
    coefficients = [0.5, 1.5, 2.5]
    y = reduce_train['accuracy_group'].values
    print("Train cappa = ", qwk(y, train_pred))
    opt_preds = optR.predict(train_pred, coefficients)
    print("Train cappa = ", qwk(y, opt_preds))
    optR.fit(train_pred, y)
    coefficients = optR.coefficients()
    print("New coefs = ", coefficients)
    opt_preds = optR.predict(train_pred, coefficients)
    print("New train cappa rounding= ", qwk(y, opt_preds))
    train_rounding_origin = eval_qwk_lgb_regr(train_pred, reduce_train)
    print("Train cappa origin ", qwk(y, train_rounding_origin))

    y_final = my_model.predict(ajusted_test)
    y_final = optR.predict(y_final, coefficients)
    predict(sample_submission, y_final)
    print("Programm full time:", time.time() - start_program)

@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

class MyModel:
    def __init__(self, train, features, kmodels=5, kfold = 5):
        self.bin_models= []
        self.models = []
        self.features = features
        self.kfold = kfold


        params = {
            'num_boost_round': 1000,
            'boosting_type': 'gbdt', #'dart', 'dart', 'gbdt'
            'metric':'regression',
            'objective': 'regression', #soft_kappa_obj, #'regression', #'regression',#regression',quantile fair huber poisson
            'n_jobs': -1,
            'seed': 42,
            'num_leaves': 32,
            'learning_rate': 0.08,
            'max_depth': 14,
            'lambda_l1': 2.0,
            'lambda_l2': 1.0,
            'bagging_fraction': 0.90,
            'bagging_freq': 1,
            'feature_fraction': 0.90,
            'early_stopping_rounds': 300,
            'verbose': 0,
        }

        oof_rmse_scores = []
        oof_cohen_scores = []

        for model_number in range(kmodels):
            kf = GroupKFold(n_splits=kfold)
            target = 'accuracy_group'
            oof_pred = np.zeros(len(train))
            ind = []

            for fold, (tr_ind, val_ind) in enumerate(kf.split(train, groups=train['installation_id'])):

                print('Fold:', fold + 1)
                x_train, x_val = train[features].iloc[tr_ind], train[features].iloc[val_ind]
                y_train, y_val = train[target][tr_ind], train[target][val_ind]
                x_train.drop('installation_id', inplace=True, axis=1)

                x_val, idx_val = get_random_assessment(x_val)
                ind.extend(idx_val)
                x_val.drop('installation_id', inplace=True, axis=1)
                y_val = y_val.loc[idx_val]

                train_set = lgb.Dataset(x_train, y_train, categorical_feature=['session_title'])
                val_set = lgb.Dataset(x_val, y_val, categorical_feature=['session_title'])

                model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=200,
                                  feval=eval_qwk_lgb_regr_metric,)
                                  #fobj=soft_kappa_obj,)
                                  #fobj=soft_kappa_obj,)
                ###

                self.models.append(model)
                reg_pred = model.predict(x_val)
                oof_pred[idx_val] = reg_pred

            oof_rmse_score = np.sqrt(mean_squared_error(train[target][ind], oof_pred[ind]))
            oof_cohen_score = cohen_kappa_score(train[target][ind],
                                            eval_qwk_lgb_regr(oof_pred[ind], train), weights='quadratic')

            print('Our oof rmse score is:', oof_rmse_score)
            print('Our oof cohen kappa score is:', oof_cohen_score)
            oof_rmse_scores.append(oof_rmse_score)
            oof_cohen_scores.append(oof_cohen_score)

        print('Our mean rmse score is: ', sum(oof_rmse_scores) / len(oof_rmse_scores))
        print('Our mean cohen kappa score is: ', sum(oof_cohen_scores) / len(oof_cohen_scores))



    def predict(self, test):
        current_features = [x for x in self.features if x not in ['installation_id']]
        y_pred = np.zeros(len(test))
        for model in self.models:
            y_pred += np.array(model.predict(test[current_features]), dtype=float)

        return y_pred / len(self.models)

class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1.10, 1.72, 2.25]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead', options={
            'maxiter': 5000})

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

def soft_kappa_obj(y, p):
    y = np.asarray(y)
    p = np.asarray(p.label)
    norm = p.dot(p) + y.dot(y)

    grad = -2 * y / norm + 4 * p * np.dot(y, p) / (norm ** 2)
    hess = 8 * p * y / (norm ** 2) + 4 * np.dot(y, p) / (norm ** 2) - (16 * p ** 2 * np.dot(y, p)) / (norm ** 3)
    return grad, hess

def eval_qwk_lgb_regr_metric(y_pred, true):
    y_true=true.label

    dist = Counter(y_true)
    for k in dist:
        dist[k] /= len(y_true)

    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred)))

    return 'cappa', qwk(y_true, y_pred), True


if __name__ == '__main__':
    main()


