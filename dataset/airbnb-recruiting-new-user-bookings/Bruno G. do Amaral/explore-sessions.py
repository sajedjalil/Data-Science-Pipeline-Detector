import pandas as pd


print('Reading file...')

sessions = pd.read_csv('../input/sessions.csv')

print(sessions.shape)