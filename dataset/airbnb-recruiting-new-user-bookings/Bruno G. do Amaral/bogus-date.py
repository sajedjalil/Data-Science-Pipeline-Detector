import numpy as np
import pandas as pd

train = pd.read_csv('../input/train_users_2.csv', usecols=['date_account_created', 'timestamp_first_active'])
train['date_account_created'] = train['date_account_created'].apply(lambda x: x[0:4] + x[5:7] + x[8:10])
train['timestamp_first_active'] = train['timestamp_first_active'].apply(lambda x: str(x)[:8])

different_dates = (train['date_account_created'] != train['timestamp_first_active']).sum()

print("There were %d different dates out of %d samples (%.4f%%)" % (different_dates, train.shape[0], different_dates * 100.0 / train.shape[0]))