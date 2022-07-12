import pandas as pd

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }

pd.merge(pd.read_csv('../input/test.csv', dtype=dtypes),
pd.read_csv('../input/train.csv', dtype=dtypes).groupby(['app','channel'])['is_attributed'].mean().reset_index(),
on=['app','channel'], how='left').fillna(0)[['click_id','is_attributed']].to_csv('submean_app.csv', index=False)