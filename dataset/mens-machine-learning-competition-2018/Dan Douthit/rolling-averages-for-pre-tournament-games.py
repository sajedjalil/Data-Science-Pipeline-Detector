import numpy as np
import pandas as pd
import os
import sys

# Constant definitions
numAvgGames = 10
# Columns for main DataFrame and temporary DataFrame for inner loop
rollingAvgCols = ['Season', 'TeamID', 'ScoreAvg', 'AstAvg', 'TOAvg', 'StlAvg', 'RbdsAvg', 'FGPAvg', '3PPAvg', 'FTPAvg']
dummy_cols = ['Season', 'TeamID', 'DayNum', 'Score', 'Ast', 'TO', 'Stl', 'Rbds', 'FGP', '3PP', 'FTP']


df_season_data = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
df_tourney_data = pd.read_csv('../input/NCAATourneyDetailedResults.csv')

# Drop the columns from the regular season detailed stats that I won't be using
df_season_data.drop(labels=['WLoc', 'NumOT', 'WPF', 'LPF', 'WBlk', 'LBlk'], inplace=True, axis=1)

# Combine offensive and defensive rebound numbers and field goals and free throws into percentages
# Drop the individual rebound, field goal, and free throw columns after combining them
df_season_data['WRbds'] = df_season_data['WOR'] + df_season_data['WDR']
df_season_data['LRbds'] = df_season_data['LOR'] + df_season_data['LDR']
df_season_data['WFGP'] = df_season_data['WFGM'] / df_season_data['WFGA'] * 100
df_season_data['W3PP'] = df_season_data['WFGM3'] / df_season_data['WFGA3'] * 100
df_season_data['WFTP'] = df_season_data['WFTM'] / df_season_data['WFTA'] * 100
df_season_data['LFGP'] = df_season_data['LFGM'] / df_season_data['LFGA'] * 100
df_season_data['L3PP'] = df_season_data['LFGM3'] / df_season_data['LFGA3'] * 100
df_season_data['LFTP'] = df_season_data['LFTM'] / df_season_data['LFTA'] * 100
df_season_data.drop(labels=['WOR', 'WDR', 'LOR', 'LDR'], inplace=True, axis=1)
df_season_data.drop(labels=['WFGM', 'WFGA', 'LFGM', 'LFGA', 'WFGM3', 'WFGA3', 'LFGM3', 'LFGA3', 'WFTM', 'WFTA', 'LFTM', 'LFTA'], inplace=True, axis=1)

# generate list of tournament teams and seasons for outer loops
# and find dimensions of our final output DataFrame, this will optimize the for loop later
teamList = np.unique(df_tourney_data[['WTeamID', 'LTeamID']].values)
seasonList = df_season_data.Season.unique()
numTeams = len(teamList)
numSeasons = len(seasonList)

# create empty DataFrame with dimensions calculated above
# Note: not perfectly optimized since I'm calculating rows for each tourney team each season
# 		even if they didn't make the tournament in a given year
df_rolling_avg = pd.DataFrame(index=np.arange(0, numTeams*numSeasons), columns=rollingAvgCols)

# index counter for df_rolling_avg
j = 0
############################################################################################
# Start main loop here
############################################################################################
for seasonID in seasonList:
	for teamID in teamList:

		df_team_season = df_season_data[(df_season_data.Season == seasonID) & ((df_season_data.WTeamID == teamID) | (df_season_data.LTeamID == teamID))]
		# some teams weren't Division I teams for entire 2003-2018 span, so skip if there is no data returned above
		if len(df_team_season) == 0:
			continue

		# reset the indices of df_team_season after filtering
		df_team_season = df_team_season.reset_index(drop=True)

		df_dummy = pd.DataFrame(index=np.arange(0, len(df_team_season)), columns=dummy_cols)

		for ii, row in df_team_season.iterrows():
			if row.WTeamID == teamID:
				df_dummy.loc[ii] = [row.Season, row.WTeamID, row.DayNum, row.WScore, row.WAst, row.WTO, row.WStl, row.WRbds, row.WFGP, row.W3PP, row.WFTP]
			else:
				df_dummy.loc[ii] = [row.Season, row.LTeamID, row.DayNum, row.LScore, row.LAst, row.LTO, row.LStl, row.LRbds, row.LFGP, row.L3PP, row.LFTP]

		# add columns to dummy DataFrame with rolling averages so we can store the last rolling average
		df_dummy['ScoreAvg'] = df_dummy['Score'].rolling(numAvgGames).mean()
		df_dummy['AstAvg'] = df_dummy['Ast'].rolling(numAvgGames).mean()
		df_dummy['TOAvg'] = df_dummy['TO'].rolling(numAvgGames).mean()
		df_dummy['StlAvg'] = df_dummy['Stl'].rolling(numAvgGames).mean()
		df_dummy['RbdsAvg'] = df_dummy['Rbds'].rolling(numAvgGames).mean()
		df_dummy['FGPAvg'] = df_dummy['FGP'].rolling(numAvgGames).mean()
		df_dummy['3PPAvg'] = df_dummy['3PP'].rolling(numAvgGames).mean()
		df_dummy['FTPAvg'] = df_dummy['FTP'].rolling(numAvgGames).mean()

		# Drop the single game data and recast dummy data as the averages for last 10 games of season
		df_dummy.drop(labels=['DayNum', 'Score', 'Ast', 'TO', 'Stl', 'Rbds', 'FGP', '3PP', 'FTP'], inplace=True, axis=1)
		df_dummy = df_dummy.tail(1)
		df_rolling_avg.loc[j] = df_dummy.iloc[0].values
		
		# increment counter
		j +=1

# Drop rows that are NaN
df_rolling_avg = df_rolling_avg[pd.notnull(df_rolling_avg['Season'])]


# Make sure that Season and TeamID columns are integers
df_rolling_avg[['Season']] = df_rolling_avg[['Season']].astype('int')
df_rolling_avg[['TeamID']] = df_rolling_avg[['TeamID']].astype('int')

# Finally write the data out to .csv file
df_rolling_avg.to_csv('rolling_average_data.csv', index=False)
