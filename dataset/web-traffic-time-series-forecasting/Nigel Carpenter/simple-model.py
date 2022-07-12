import numpy as np
import pandas as pd

print('Reading data...')
key_1 = pd.read_csv('../input/key_1.csv')
train_1 = pd.read_csv('../input/train_1.csv')
ss_1 = pd.read_csv('../input/sample_submission_1.csv')

print('Preprocessing...')
train_1.fillna(0, inplace=True)

print('Processing...')
ids = key_1.Id.values
pages = key_1.Page.values

print('key_1...')
d_pages = {}
for id, page in zip(ids, pages):
    d_pages[id] = page[:-11]

print('train_1...')
pages = train_1.Page.values
# visits = train_1['2016-12-31'].values # Version 1 score: 60.6
# visits = np.round(np.mean(train_1.drop('Page', axis=1).values, axis=1)) # Version 2 score: 64.8
# visits = np.round(np.mean(train_1.drop('Page', axis=1).values[:, -14:], axis=1)) # Version 3 score: 52.5
visits = np.round(np.median(train_1.drop('Page', axis=1).values[:, -14:], axis=1)) 

d_visits = {}
for page, visits_number in zip(pages, visits):
    d_visits[page] = visits_number

print('Modifying sample submission...')
ss_ids = ss_1.Id.values
ss_visits = ss_1.Visits.values

for i, ss_id in enumerate(ss_ids):
    ss_visits[i] = d_visits[d_pages[ss_id]]

print('Saving submission...')
subm = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})
subm.to_csv('submission.csv', index=False)