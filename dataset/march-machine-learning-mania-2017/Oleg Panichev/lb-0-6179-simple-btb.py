__author__ = 'opanichev'

import numpy as np 
import pandas as pd 
import random
from random import shuffle
from sklearn.metrics import log_loss

random.seed(1001)

def predict(team1, team2, win_proba):   
    p1 = np.array([win_proba[t] if t in win_proba else 0 for t in team1])
    p2 = np.array([win_proba[t] if t in win_proba else 0 for t in team2])
    pred = (p1 - p2) + 0.5
    pred = [v if (v <= 1) else 1 for v in pred]
    pred = [v if (v >= 0) else 0 for v in pred]
    return pred

years = [2013, 2014, 2015, 2016]
loss = np.zeros(len(years))
results = []

# print('Reading the data...')
dsc = pd.read_csv('../input/RegularSeasonCompactResults.csv')
dtc = pd.read_csv('../input/TourneyCompactResults.csv')
data = pd.concat([dsc, dtc])

# print('Preparing submission...')
s = pd.read_csv('../input/sample_submission.csv')
s['year'] = [int(x.split('_')[0]) for x in s['id'].values]

for i, year in enumerate(years):
    print('*** Year ' + str(year) + ' ***')

    # print('Splitting the data on train and validation sets...')
    validation = data[data.Season == year]
    train = data[data.Season < year]
    print('train.shape = ' + str(train.shape))
    print('validation.shape = ' + str(validation.shape))

    # print('Calculating amounts of each team won and lost...')
    win_count = train.groupby('Wteam').Wscore.agg(['count']).reset_index()
    win_count.rename(columns={'Wteam': 'Team', 'count': 'win_count'}, inplace=True)
    lose_count = train.groupby('Lteam').Lscore.agg(['count']).reset_index()
    lose_count.rename(columns={'Lteam': 'Team', 'count': 'lose_count'}, inplace=True)

    # print('Extract probability of winning for each team...')
    win_proba = win_count.merge(lose_count, on='Team')
    win_proba['proba'] = win_proba['win_count'].values/ \
        (1.0*(win_proba['win_count'].values + win_proba['lose_count'].values))
    win_proba.set_index('Team', inplace=True)
    win_proba.drop(['win_count', 'lose_count'], axis=1, inplace=True)
    win_proba = win_proba.to_dict()
    win_proba = win_proba['proba']

    # print('Validating...')
    team1 = validation['Wteam'].values
    team2 = validation['Lteam'].values
    idx = random.sample(range(0, len(team1)), int(round(len(team1)/2.0)))
    tmp = team1[idx]
    team1[idx] = team2[idx]
    team2[idx] = tmp
    y_true = np.ones(validation.shape[0])
    y_true[idx] = np.zeros(len(idx))
    pred = predict(team1, team2, win_proba)
    loss[i] = log_loss(y_true, pred)
    print('Validation logloss = ' + str(loss[i]))

    # print('Preparing submission...')
    ss = s[s.year == year]
    team1 = [int(x.split('_')[1]) for x in ss['id'].values]
    team2 = [int(x.split('_')[2]) for x in ss['id'].values]
    ss['pred'] = predict(team1, team2, win_proba)
    ss.drop(['year'], axis=1, inplace=True)
    results.append(ss)

print('Average logloss = ' + str(np.mean(loss)))
print('Saving to file...')
results = pd.concat(results)
results.to_csv('subm.csv', index=False)
print('Done!')

