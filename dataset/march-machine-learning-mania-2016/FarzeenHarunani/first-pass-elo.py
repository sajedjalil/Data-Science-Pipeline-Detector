
import pandas as pd
import numpy as np

data = pd.read_csv('../input/RegularSeasonCompactResults.csv')
teams = pd.read_csv('../input/Teams.csv'); 
sample = pd.read_csv('../input/SampleSubmission.csv')

# Initialize Elo Variables
homeAdvantage = 100
k = 24
cval = 200
year = 1985 
ratings = {}
for t in range(0, len(teams.index)): 
	ratings[teams.Team_Id[t]] = 1500	
	
# Calculate Elo Ratings
for index, row in data.iterrows(): 
	if data.Season[index] != year:
		year = data.Season[index]
		temp = dict(ratings)
		for key,value in temp.items(): 
			ratings[key] = .75 * value + .25 * 1500
	Ra = ratings[data.Wteam[index]]
	Rb = ratings[data.Lteam[index]]
	if data.Wloc[index] == 'A': 
		Rb += homeAdvantage
	elif data.Wloc[index] == 'H': 
		Ra += homeAdvantage
	Ea = 1/(1+(10**((Rb-Ra)/cval))) 
	Eb = 1/(1+(10**((Ra-Rb)/cval))) 
	ratings[data.Wteam[index]] = Ra + k*(1-Ea)
	ratings[data.Lteam[index]] = Rb + k*(0-Eb)

games = pd.concat([sample['Id'],sample['Id'].str.split('_', expand=True)], axis=1)
games.rename(columns={0: 'season', 1: 'team1',2: 'team2'}, inplace=True)
games['season'] = pd.to_numeric(games['season'])
games['team1'] = pd.to_numeric(games['team1'])
games['team2'] = pd.to_numeric(games['team2'])
games['Pred'] = 0
for i in range(0,len(games.index)): 
	m = ratings[games.team2[i]] - ratings[games.team1[i]]
	games.ix[i,'Pred'] = 1/(1+(10**(m/200)))

games[['Id','Pred']].to_csv('eloResults.csv',index=False)