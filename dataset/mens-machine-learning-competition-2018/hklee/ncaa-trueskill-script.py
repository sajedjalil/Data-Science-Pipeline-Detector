# TrueSkill is a rating system based on Bayesian inference, estimating each players skill as a gaussian like Elo rating.
# See trueskill.org for more.

import pandas as pd, numpy as np
from trueskill import TrueSkill, Rating, rate_1vs1

ts = TrueSkill(draw_probability=0.01) # 0.01 is arbitary small number
beta = 25 / 6  # default value

def win_probability(p1, p2):
    delta_mu = p1.mu - p2.mu
    sum_sigma = p1.sigma * p1.sigma + p2.sigma * p2.sigma
    denom = np.sqrt(2 * (beta * beta) + sum_sigma)
    return ts.cdf(delta_mu / denom)
    
submit = pd.read_csv('../input/SampleSubmissionStage1.csv')
submit[['Season', 'Team1', 'Team2']] = submit.apply(lambda r:pd.Series([int(t) for t in r.ID.split('_')]), axis=1)

df_tour = pd.read_csv('../input/RegularSeasonCompactResults.csv')
teamIds = np.unique(np.concatenate([df_tour.WTeamID.values, df_tour.LTeamID.values]))
ratings = { tid:ts.Rating() for tid in teamIds }

def feed_season_results(season):
    print("season = {}".format(season))
    df1 = df_tour[df_tour.Season == season]
    for r in df1.itertuples():
        ratings[r.WTeamID], ratings[r.LTeamID] = rate_1vs1(ratings[r.WTeamID], ratings[r.LTeamID])

def update_pred(season):
    beta = np.std([r.mu for r in ratings.values()]) 
    print("beta = {}".format(beta))
    submit.loc[submit.Season==season, 'Pred'] = submit[submit.Season==season].apply(lambda r:win_probability(ratings[r.Team1], ratings[r.Team2]), axis=1)

for season in sorted(df_tour.Season.unique())[:-4]: # exclude last 4 years
    feed_season_results(season)

update_pred(2014)
feed_season_results(2014)
update_pred(2015)
feed_season_results(2015)
update_pred(2016)
feed_season_results(2016)
update_pred(2017)

submit.drop(['Season', 'Team1', 'Team2'], axis=1, inplace=True)
submit.to_csv('trueskill_estimation.csv', index=None)

