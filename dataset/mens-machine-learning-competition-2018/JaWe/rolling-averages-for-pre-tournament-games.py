import pandas as pd

# Constant definitions
numAvgGames = 10

df_season_data = pd.read_csv('data/RegularSeasonDetailedResults.csv')
df_tourney_data = pd.read_csv('data/NCAATourneyDetailedResults.csv')

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

# Start to get the Columns Winners and Losers in one column 'Team' by splitting the data in 2 dummy dataframes

df_winners = df_season_data[['Season', 'DayNum', 'WTeamID', 'WScore', 'WAst', 'WTO', 'WStl', 'WRbds', 'WFGP', 'W3PP', 'WFTP']]
df_losers = df_season_data[['Season', 'DayNum', 'LTeamID', 'LScore', 'LAst', 'LTO', 'LStl', 'LRbds', 'LFGP', 'L3PP', 'LFTP']]

df_winners.rename(columns={
    'WTeamID': 'TeamID', 
    'WScore': 'Score', 
    'WAst': 'Ast',
    'WTO': 'TO',
    'WStl': 'Stl',
    'WRbds': 'Rbds',
    'WFGP': 'FGP',
    'W3PP': '3PP',
    'WFTP': 'FTP'
}, inplace=True)
df_losers.rename(columns={
    'LTeamID': 'TeamID', 
    'LScore': 'Score', 
    'LAst': 'Ast',
    'LTO': 'TO',
    'LStl': 'Stl',
    'LRbds': 'Rbds',
    'LFGP': 'FGP',
    'L3PP': '3PP',
    'LFTP': 'FTP'
}, inplace=True)

# Concat the winners and losers
df_scoring = pd.concat((df_winners, df_losers))
df_scoring.head(5)

# Use Pandas' 'nlargest' on the column 'DayNum' with the earlier defined 'numAvgGames'
df_last_10 = df_scoring.groupby(['Season', 'TeamID'], as_index = False).apply(lambda x: x.nlargest(numAvgGames, columns=['DayNum']))

# Do the final groupby with the aggregation of the wanted stats
df_rolling_avg = df_last_10.groupby(['Season', 'TeamID'], as_index = False).agg(
    {'Score': 'mean', 
     'Ast': 'mean', 
     'TO': 'mean',
     'Stl': 'mean',
     'Rbds': 'mean',
     'FGP': 'mean',
     '3PP': 'mean',
     'FTP': 'mean'
    })

df_rolling_avg.head(10)