import numpy as np
import pandas as pd
from sklearn import preprocessing

def sessions_stats(group):
    group.fillna(0, inplace=True)

    if group.count() == 0:
        return {'sessions_total_duration': group.max() - group.min(),
                'average_action_duration': 0,
                'actions_total_count': 0}
    else:
        return {'sessions_total_duration': group.max() - group.min(),
                'average_action_duration': (group.max() - group.min()) / group.count(),
                'actions_total_count': group.count()}

def main():
    # two scalers we use
    sessions_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    action_scaler = lambda x: np.sqrt(x/3600)

    df_sessions = pd.read_csv('../input/sessions.csv')

    # gather sessions total & average durations from 'secs_elapsed' column,
    # which will be droped lately, apply external function
    df_sstats = df_sessions['secs_elapsed'].groupby(
        df_sessions['user_id'] ).apply(sessions_stats).unstack()

    # scale features for better distribution
    df_sstats['actions_total_count'] = df_sstats['actions_total_count'].apply(action_scaler)
    df_sstats['average_action_duration'] = df_sstats['average_action_duration'].apply(action_scaler)

    # normalization of new features + convert them to int
    # to avoid possible later scientific notation
    normalize_feats = ['actions_total_count',
        'average_action_duration', 'sessions_total_duration']
    for f in normalize_feats:
        df_sstats[f] = sessions_scaler.fit_transform(df_sstats[f].reshape(-1, 1)).astype(int)

    # take rest of the columns for OHE
    df_sactions = df_sessions.groupby(
        ['user_id', 'action_detail', 'action_type'], as_index=False ).count()
    # Drop 'secs_elapsed', already aggregated to something usefull,
    # 'device_type' already in 'train_users_2', 'action' redundant
    df_sactions.drop( ['secs_elapsed', 'action', 'device_type'],
        axis=1, inplace=True)

    # one-hot-encoding sessions features
    ohe_features = ['action_detail', 'action_type']
    for f in ohe_features:
        df_dummy = pd.get_dummies(df_sactions[f], prefix=f)
        df_sactions.drop([f], axis=1, inplace = True)
        df_sactions = pd.concat((df_sactions, df_dummy.astype(int)), axis=1)

    # merge OHE to single row
    df_sactions = df_sactions.groupby(['user_id']).sum().reset_index()

    # join them all into single DataFrame
    df_joined = df_sactions.join(df_sstats, on=['user_id'], how='left')
    df_joined.rename(columns={'user_id': 'id'}, inplace=True)

    pd.DataFrame(df_joined).to_csv('./sessions_done.csv',
        sep=',', header=True, index=False)

if __name__ == '__main__':
    main()
