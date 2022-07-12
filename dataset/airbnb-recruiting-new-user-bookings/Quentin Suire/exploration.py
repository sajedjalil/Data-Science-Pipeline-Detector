import pandas as pd
import numpy as np
files = [
    'countries',
    'age_gender_bkts',
    'test_users',
    'train_users',
    'sessions'
]
data = {}
for f in files:
    data[f] = pd.read_csv('../input/' + f + '.csv')
# Dummy prediction: predict US when it is possible, otherwise NDF
def predict(row):
    country = 'US'
    if str(row['date_first_booking']) == 'nan':
        country = 'NDF'
    return { 'id': row['id'], 'country': country }

results = pd.DataFrame(columns=('id', 'country'))
for index, row in data['test_users'].iterrows():
    prediction = predict(row)
    results.loc[index] = [prediction['id'], prediction['country']]

#submission.to_csv('submission.csv', index=False)

