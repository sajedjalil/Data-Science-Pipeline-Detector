# importing packages
import numpy as np
import pandas as pd

# file path
path = '../input/'

# importing files
video_footage_control = pd.read_csv(path+'video_footage-control.csv')
video_footage_injury = pd.read_csv(path+'video_footage-injury.csv')
play_information = pd.read_csv(path+'play_information.csv')
play_player_role_data = pd.read_csv(path+'play_player_role_data.csv')
game_data = pd.read_csv(path+'game_data.csv')
player_punt_data = pd.read_csv(path+'player_punt_data.csv')
video_review = pd.read_csv(path+'video_review.csv')

# Key Variables:
# GameKey, PlayID, GSISD

#################
# Cleaning Data #
#################

### cleaning video_footage_control

# quick value sumary
video_footage_control.head()
video_footage_control.info(verbose=True)

# loop to provide unique values for every column in df
for i in range(0,10):
    print(video_footage_control.iloc[:,i].value_counts(dropna=False))

# notes: video_footage_control
# no null values, all values look okay to move forward with
# added the column control_injury to mark data as 'control' data

### cleaning video_footage_injury

# quick value sumary
video_footage_injury.head()
video_footage_injury.info(verbose=True)

# loop to provide unique values for every column in df
for i in range(0,10):
    print(video_footage_injury.iloc[:,i].value_counts(dropna=False))

# adding column control_injury
video_footage_injury['control_injury'] = np.nan
video_footage_injury['control_injury'] = video_footage_injury['control_injury'].replace(np.nan,'injury')

# notes: video_footage_injury
# no null values, all values look okay to move forward with
# added the column control_injury to mark data as 'injury' data

### cleaning play_information
# quick value sumary
play_information.head()
play_information.info(verbose=True)

# loop to provide unique values for every column in df
for i in range(0,14):
    print(play_information.iloc[:,i].value_counts(dropna=False))

# notes: play_information
# no null values, all values look okay to move forward with

### cleaning play_player_role_data
# quick value sumary
play_player_role_data.head()
play_player_role_data.info(verbose=True)

# loop to provide unique values for every column in df
for i in range(0,5):
    print(play_player_role_data.iloc[:,i].value_counts(dropna=False))

# notes: play_player_role_data
# no null values, all values look okay to move forward with

### cleaning game_data
# quick value sumary
game_data.head()
game_data.info(verbose=True)

# columns with null values:
# StadiumType, Turf, GameWeather, Temperature, OutdoorWeather

# loop to provide unique values for every column in df
for i in range(0,18):
    print(game_data.iloc[:,i].value_counts(dropna=False))

# colums with messy data:
# Game_Date, Stadium, Stadium Type, Turf, GameWeather, Temperature

# cleaning game_data['Game_Date']
# remove ending 0s, YYYY-MM-DD
game_data['Game_Date'] = game_data['Game_Date'].astype('str')
game_data['Game_Date'] = game_data['Game_Date'].str.replace(' 00:00:00.000','')

# cleaning game_data['Stadium']
stadium_prefix = [
    'AT&T',
    'Arrowhead',
    'Bank of America',
    'Camping World',
    'CenturyLink'
    'Estadio Azteca',
    'EverBank',
    'FedEx',
    'First Energy',
    'Ford',
    'Georgia Dome',
    'Gillette',
    'Hard Rock',
    'Heinz',
    'Lambeau',
    'Levis',
    'Lincoln Financial',
    'Los Angeles Memorial',
    'Lucas Oil',
    'M&T',
    'Mercedes'
    'MetLife',
    'NRG',
    'New Era',
    'Nissan',
    'Oakland',
    'Paul Brown',
    'Qualcomm',
    'Ralph Wilson',
    'Raymond James',
    'Soldier',
    'Sports Authority',
    'StubHub',
    'Tom Benson Hall of Fame',
    'Twickenham',
    'US Bank',
    'University of Phoenix',
    'Wembley'
    ]

# replace messy values
game_data['Stadium'] = game_data['Stadium'].str.replace('-', ' ')
game_data['Stadium'] = game_data['Stadium'].str.replace('.', '')
game_data['Stadium'] = game_data['Stadium'].str.replace(' & ', '&')
game_data['Stadium'] = game_data['Stadium'].str.replace('  ', ' ')
game_data['Stadium'] = game_data['Stadium'].str.replace('Phoeinx', 'Phoenix')
game_data['Stadium'] = game_data['Stadium'].str.replace('CenturyLink Field', 'CenturyLink')
game_data['Stadium'] = game_data['Stadium'].str.replace('FirstEnergy', 'First Energy')
game_data['Stadium'] = game_data['Stadium'].str.replace('Raymon', 'Raymond')
game_data['Stadium'] = game_data['Stadium'].str.replace('Raymondd', 'Raymond')
game_data['Stadium'] = game_data['Stadium'].str.replace('Dome', '')
game_data['Stadium'] = game_data['Stadium'].str.replace('Stadium', '')
game_data['Stadium'] = game_data['Stadium'].str.replace('Superdome', '')
game_data['Stadium'] = game_data['Stadium'].str.replace('Solider', 'Soldier')
game_data['Stadium'] = game_data['Stadium'].str.replace('Solidier', 'Soldier')
game_data['Stadium'] = game_data['Stadium'].str.replace('MetLife ', 'MetLife')

# for loops that replace values that contain 'prefix' and replace with 'prefix' or preferred value
game_data_replace = game_data['Stadium']
for prefix in stadium_prefix:
    game_data_replace.loc[game_data_replace.str.contains(prefix, case=False)] = prefix
game_data['Stadium'] = game_data_replace

# check series cleaning results
print(game_data['Stadium'].value_counts(dropna=False).sort_index())

# cleaning game_data['StadiumType']
stadiumtype_prefix = [
    'Dome',
    'Indoor',
    'Outdoor',
    'Heinz'
    ]

stadium_indoor = [
    'AT&T',
    'Ford',
    'Lucas Oil',
    'NRG',
    'US Bank',
    'University of Phoenix',
    'Wembley'
    ]

stadium_outdoor = [
    'Arrowhead',
    'Bank of America',
    'Camping World',
    'EverBank',
    'FedEx',
    'First Energy',
    'Gillette',
    'Hard Rock',
    'Heinz',
    'Lambeau',
    'Levis',
    'Lincoln Financial',
    'Los Angeles Memorial',
    'M&T',
    'MetLife'
    'New Era',
    'Nissan',
    'Oakland',
    'Paul Brown',
    'Qualcomm',
    'Ralph Wilson',
    'Raymond James',
    'Soldier',
    'Sports Authority',
    'StubHub',
    'Tom Benson Hall of Fame',
    'Twickenham'
    ]

# replace messy values
game_data['StadiumType'] = game_data['StadiumType'].replace(np.nan, game_data['Stadium'])
game_data['StadiumType'] = game_data['StadiumType'].str.replace('-', ' ')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('.', '')
game_data['StadiumType'] = game_data['StadiumType'].str.replace(' & ', '&')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('  ', ' ')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Turf', 'Tom Benson Hall of Fame')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Dome', 'Indoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Domed', 'Indoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('MetLife', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('New Era', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Oudoor', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Open', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Ourdoor', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Outddors', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Outdor', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Outside', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retr Roof  Closed', 'Indoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retr Roof  Open', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retr Roof Closed', 'Indoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retr Roof Open', 'Outdoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retr roof  closed', 'Indoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retr roof closed', 'Indoor')
game_data['StadiumType'] = game_data['StadiumType'].str.replace('Retractable Roof', 'Indoor')

# for loops that replace values that contain 'prefix' and replace with 'prefix' or preferred value
game_data_replace = game_data['StadiumType']
for prefix in stadiumtype_prefix:
    game_data_replace.loc[game_data_replace.str.contains(prefix, case=False)] = prefix
game_data['StadiumType'] = game_data_replace

for stadium in stadium_indoor:
    game_data['StadiumType'] = game_data['StadiumType'].replace([stadium],'Indoor')

for stadium in stadium_outdoor:
    game_data['StadiumType'] = game_data['StadiumType'].replace([stadium],'Outdoor')

# check series cleaning results
print(game_data['StadiumType'].value_counts(dropna=False).sort_index())

# cleaning game_data['Turf']
turf_turf = [
    'turf',
    'Artifical',
    'Artificial',
    'Synthetic',
    'S5-M'
    ]

turf_grass = [
    'grass',
    'Natural',
    'Hard Rock'
    ]

# replace messy values
game_data['Turf'] = game_data['Turf'].replace(np.nan, game_data['Stadium'])

# for loops that replace values that contain 'prefix' and replace with 'prefix' or preferred value
game_data_replace = game_data['Turf']
for prefix in turf_turf:
    game_data_replace.loc[game_data_replace.str.contains(prefix, case=False)] = 'Turf'
for prefix in turf_grass:
    game_data_replace.loc[game_data_replace.str.contains(prefix, case=False)] = 'Grass'
game_data['Turf'] = game_data_replace

# check series cleaning results
print(game_data['Turf'].value_counts(dropna=False).sort_index())

# cleaning game_data['GameWeather']
weather_rain = [
    'rain'
    ]

weather_no_rain = [
    'Clear',
    'Cloudy',
    'Indoor',
    'Controlled',
    'Fair',
    'Indoors',
    'Sunny',
    ]

# replacing GameWeather value with 'Indoor' for all indoor stadiums
game_data['GameWeather'] = (
    game_data['GameWeather']
    .where(game_data['StadiumType']=='Indoor', 'Indoor')
    )

# replace messy values
game_data['GameWeather'] = game_data['GameWeather'].replace(np.nan, 'NaN.str')

# for loops that replace values that contain 'prefix' and replace with 'prefix' or preferred value
game_data_replace = game_data['GameWeather']
for prefix in weather_rain:
    game_data_replace.loc[game_data_replace.str.contains(prefix, case=False)] = 'Rain'
for prefix in weather_no_rain:
    game_data_replace.loc[game_data_replace.str.contains(prefix, case=False)] = 'No_Rain'
game_data['GameWeather'] = game_data_replace

# check series cleaning results
print(game_data['GameWeather'].value_counts(dropna=False).sort_index())

# cleaning game_data['Temperature']
# going to kepp variable as continous

# replacing NaN with 999
game_data['Temperature'] = game_data['Temperature'].replace(np.nan, 999)

# check series cleaning results
print(game_data['Temperature'].value_counts(dropna=False).sort_index())

# dropping game_data['OutdoorWeather'] due to being duplicate of game_data['GameWeather']
game_data = game_data.drop(['OutdoorWeather'], axis=1)

# dataset game_data clean

### cleaning player_punt_data
player_punt_data.head()
player_punt_data.info(verbose=True)

# dataset player_punt_data clean

### cleaning video_review
video_review.head()
video_review.info(verbose=True)

# replace all NaN values with 'NaN.str'
video_review = video_review.replace(np.nan, 'NaN.str')

# dataset player_punt_data clean

######################
# Combining Datasets #
######################

## Merging play_player_role_data and player_punt_data
# provides typical role for player and punt specific role

# creating unique_id for play_player_role_data
# unique_id = GameKey_GSISID_PlayID

# turning unique_id inputs into strings
play_player_role_data['GameKey'] = play_player_role_data['GameKey'].astype(str)
play_player_role_data['GSISID'] = play_player_role_data['GSISID'].astype(str)
play_player_role_data['PlayID'] = play_player_role_data['PlayID'].astype(str)

# creating unique_id
play_player_role_data['unique_id'] = (
    play_player_role_data['GameKey']
    + '_'
    + play_player_role_data['GSISID']
    + '_'
    + play_player_role_data['PlayID']
    )

# renaming player_punt_data['Position'] to player_punt_data['Punt_Position']
player_punt_data = player_punt_data.rename(columns={'Position': 'Punt_Position'})

# turning unique_id inputs into strings
player_punt_data['GSISID'] = player_punt_data['GSISID'].astype(str)

# merging play_player_role_data and player_punt_data on GSISD
# describes player's typical role and their punt role
nfl_final = pd.merge(play_player_role_data, player_punt_data, on='GSISID', how='outer')

## Merging nfl_final with game_data
# provides game info for each unique_id

# turning unique_id inputs into strings
game_data['GameKey'] = game_data['GameKey'].astype(str)

# merging nfl_final and game_data on GameKey
nfl_final = pd.merge(nfl_final, game_data, on='GameKey', how='outer')

# cleaning columns
# changing 'Season_Year_x' to Season_Year; dropping Season_Year_y
nfl_final = nfl_final.rename(columns={'Season_Year_x': 'Season_Year'})
nfl_final = nfl_final.drop(['Season_Year_y'], axis=1)

## Merging nfl_final with play_information

# turning unique_id inputs into strings
play_information['GameKey'] = play_information['GameKey'].astype(str)
play_information['PlayID'] = play_information['PlayID'].astype(str)

# creating GameKey_PlayID for both datframes to merge
nfl_final['GameKey_PlayID'] = nfl_final['GameKey'] + '_' + nfl_final['PlayID']
play_information['GameKey_PlayID'] = play_information['GameKey'] + '_' + play_information['PlayID']

# merging nfl_final and play_information on GameKey_PlayID
nfl_final = pd.merge(nfl_final, play_information, on='GameKey_PlayID', how='outer')

# dropping nfl_final['GameKey_PlayID'] to free up space
nfl_final = nfl_final.drop(['GameKey_PlayID'], axis=1)

# cleaning columns
# changing 'ColumnName_x' to Season_Year; dropping ColumnName_y
nfl_final = nfl_final.rename(columns={'Season_Year_x': 'Season_Year'})
nfl_final = nfl_final.rename(columns={'GameKey_x': 'GameKey'})
nfl_final = nfl_final.rename(columns={'PlayID_x': 'PlayID'})
nfl_final = nfl_final.rename(columns={'Season_Type_x': 'Season_Type'})
nfl_final = nfl_final.rename(columns={'Week_x': 'Week'})
nfl_final = nfl_final.rename(columns={'Game_Date_x': 'Game_Date'})

nfl_final = nfl_final.drop(['Season_Year_y'], axis=1)
nfl_final = nfl_final.drop(['Season_Type_y'], axis=1)
nfl_final = nfl_final.drop(['GameKey_y'], axis=1)
nfl_final = nfl_final.drop(['Game_Date_y'], axis=1)
nfl_final = nfl_final.drop(['Week_y'], axis=1)
nfl_final = nfl_final.drop(['PlayID_y'], axis=1)

## cleaning nfl_final before adding concussion info
# dropping duplicate rows with duplicate unique_id
nfl_final = nfl_final.drop_duplicates(subset ='unique_id')

# dropping rows without unique_id
nfl_final = nfl_final.dropna(subset=['unique_id'])

# replacing NaN values with 'NaN.str'
nfl_final = nfl_final.replace(np.nan, 'NaN.str')

## Merging nfl_final with video_review
# video_review are all the concussed players

# replacing NaN values with 'NaN.str'
video_review = video_review.replace(np.nan, 'NaN.str')

# turning unique_id inputs into strings
video_review['GameKey'] = video_review['GameKey'].astype(str)
video_review['GSISID'] = video_review['GSISID'].astype(str)
video_review['PlayID'] = video_review['PlayID'].astype(str)

# creating column video_review['concussion_status']
video_review['concussion_status'] = 'concussion'

# creating unique_id
video_review['unique_id'] = (
    video_review['GameKey']
    + '_'
    + video_review['GSISID']
    + '_'
    + video_review['PlayID']
    )

# merging nfl_final and video_review on unique_id
nfl_final = pd.merge(nfl_final, video_review, on='unique_id', how='outer')

# replacing NaN with 'no_concussion'
# all remaining NaN are rows that were not included in concussion population
nfl_final = nfl_final.replace(np.nan, 'no_concussion')

# cleaning columns
# changing 'ColumnName_x' to Season_Year; dropping ColumnName_y
nfl_final = nfl_final.rename(columns={'Season_Year_x': 'Season_Year'})
nfl_final = nfl_final.rename(columns={'GameKey_x': 'GameKey'})
nfl_final = nfl_final.rename(columns={'PlayID_x': 'PlayID'})
nfl_final = nfl_final.rename(columns={'GSISID_x': 'GSISID'})

nfl_final = nfl_final.drop(['Season_Year_y'], axis=1)
nfl_final = nfl_final.drop(['GameKey_y'], axis=1)
nfl_final = nfl_final.drop(['PlayID_y'], axis=1)
nfl_final = nfl_final.drop(['GSISID_y'], axis=1)

## nfl_final dataframe info
print(nfl_final.head())

# number of null values in nfl_final
nfl_final.isnull().sum().sum()

# column names in nfl_final
list(nfl_final)

# colums with missing values in nfl_final
null_columns=nfl_final.columns[nfl_final.isnull().any()]
nfl_final[null_columns].isnull().sum()

############################
# Exporting Clean Datasets #
############################

#exporting nfl_final as csv
nfl_final.to_csv('nfl_final.csv', index=False)

# exporting series unique_id as a single dataframe to subset NGS data
nfl_unique_id = pd.DataFrame(nfl_final['unique_id'])
nfl_unique_id.to_csv('nfl_unique_id.csv', index=False)