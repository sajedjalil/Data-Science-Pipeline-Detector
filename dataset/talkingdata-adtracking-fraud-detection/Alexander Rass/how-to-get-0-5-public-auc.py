import pandas as pd

dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }
train_grp = pd.read_csv(r'../input/train.csv', parse_dates=['click_time']
                 , dtype=dtypes).groupby(['app'])['is_attributed'].mean().reset_index()
test = pd.read_csv(r'../input/test.csv', parse_dates=['click_time'])
df = pd.merge(test, train_grp, how = 'left', on = ['app'])
df['hour'] = pd.DatetimeIndex(df['click_time']).hour
df.loc[df['hour']==4, ['is_attributed']] = 0
df.fillna(0)[['click_id','is_attributed']].to_csv('public.csv', index=False)