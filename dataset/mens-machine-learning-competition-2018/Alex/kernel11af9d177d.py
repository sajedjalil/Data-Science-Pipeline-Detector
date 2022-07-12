# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_dir = '../input/'
# df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
# df_tour = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')

df_reg_season = pd.read_csv(data_dir + 'RegularSeasonCompactResults.csv')
df_teams = pd.read_csv(data_dir + 'Teams.csv')
df_massey = pd.read_csv(data_dir + 'MasseyOrdinals.csv')
df_tourney_results = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')

# print(df_massey)

df_massey['Season'] = df_massey['Season'].map(str) + '-' + df_massey['TeamID'].map(str)
df_massey.set_index('Season', inplace=True)
df_massey.drop(['TeamID'], axis=1, inplace=True)


ranking_systems = {};
for s in df_massey['SystemName'].unique():
    ranking_systems[s] = {}
    ranking_systems[s]['total'] = 0
    ranking_systems[s]['games'] = 0

for index, row in df_tourney_results.iterrows():
    try:
        df_grouped = df_massey.loc[str(row['Season'])+'-'+str(row['WTeamID'])].groupby('SystemName').last()
        for i, group_row in df_grouped.iterrows():
                ranking_systems[i]['games'] += 1
                ranking_systems[i]['total'] += group_row['OrdinalRank']
                
    except KeyError:
        if row['Season'] > 2002:
            print('error')



df_ranking_systems = pd.DataFrame(data=ranking_systems).transpose()
df_ranking_systems['score'] = df_ranking_systems['games']/df_ranking_systems['total']
df_ranking_systems = df_ranking_systems.loc[df_ranking_systems['games'] > 200]
df_ranking_systems = df_ranking_systems.sort_values(by=['score'])
# print(df_ranking_systems)

df_massey = pd.read_csv(data_dir + 'MasseyOrdinals.csv')
df_current_massey = df_massey.loc[df_massey['Season'] == 2018]
df_current_massey = df_current_massey.loc[df_current_massey['RankingDayNum'] == 133]
print(df_current_massey)

curr_massey_lst = df_current_massey['SystemName'].unique()

df_ranking_systems = df_ranking_systems[df_ranking_systems.index.isin(curr_massey_lst)]


df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
df_seeds = df_seeds.loc[df_seeds['Season'] == 2018]
ncaa_2018_teamids = df_seeds['TeamID']
# print(df_seeds)
# print(ncaa_2018_teamids)

rankings_lst = ['WLK','ARG','RPI','COL','DOL','WOL','BIH','SAG']
df_ranking_systems = df_ranking_systems[df_ranking_systems.index.isin(rankings_lst)]

# getting the current rankings of all the ncaa teams in the tourney
df_current_massey = df_current_massey[df_current_massey['TeamID'].isin(ncaa_2018_teamids)]
df_current_massey = df_current_massey[df_current_massey['SystemName'].isin(rankings_lst)]
df_current_massey = df_current_massey.sort_values(by=['TeamID'])

teams = {}
for index, row in df_current_massey.iterrows():
    try:
        teams[row['TeamID']]['rank'] += df_ranking_systems.at[row['SystemName'],'score']*row['OrdinalRank']
    except:
        teams[row['TeamID']] = {}
        teams[row['TeamID']]['rank'] = df_ranking_systems.at[row['SystemName'],'score']*row['OrdinalRank']

#print(teams)

df_teams = pd.read_csv(data_dir + 'Teams.csv')
#print(df_teams.loc[df_teams['TeamID'].isin(ncaa_2018_teamids)])

def get_id(team_name):
    try:
        return df_teams.loc[df_teams['TeamName'] == team_name]['TeamID'].values[0]
    except:
        return "invalid name"


def probability(team1, team2):
    try:
        
        team1_rank = teams[team1]['rank']+10
        team2_rank = teams[team2]['rank']+10
        
        team1_prob = round(team2_rank/(team1_rank+team2_rank),2)
        team2_prob = round(team1_rank/(team1_rank+team2_rank),2)
        return team1_prob
        # print(team1 + ' probability to win: ' + str(team1_prob) + '%')
        # print(team2 + ' probability to win: ' + str(team2_prob) + '%')
    except:
        print('invalid team names')
        
ids = []
probs = []
df_seeds = df_seeds.sort_values(by=['TeamID'])
#print(df_seeds)

for row in df_seeds.itertuples():
    for row2 in df_seeds.itertuples():
        if (row2[3] > row[3]):
            output_id = str(row[1]) + '_' + str(row[3]) + '_' + str(row2[3])
            
            prob = probability(row[3], row2[3])
            ids.append(output_id)
            probs.append(prob)
            
#print(ids)
#print(probs)

df_final = pd.DataFrame(ids)
df_final.columns = ['ID']
df_probs = pd.DataFrame(probs)
df_probs.columns = ['Pred']
df_final = df_final.join(df_probs)
df_final.to_csv('submit.csv', index = False)
#print(df_final)