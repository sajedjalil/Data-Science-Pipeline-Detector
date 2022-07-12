'''
This script prepares data and creates basic features for the NFL 1st and Future 2020
challenge. It also reduces the memory footprint of datasets and
saves them to a memory-efficient format.

'''

#%%
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

import skmem #my utility script


#%%
def convert_days(df, daycols):
    day_buckets = np.array([1, 7, 28, 42])
    days_matrix = df[daycols] * day_buckets
    days_column = days_matrix.max(axis=1)
    return days_column

def convert_keys(df):
    df = df.fillna({'PlayKey': '0-0-0'})
    try:
        id_array = df.GameID.str.split('-', expand=True).to_numpy()
    except:
        id_array = df.PlayKey.str.split('-', expand=True).to_numpy()
    df['PlayerKey'] = id_array[:,0].astype(int)
    df['GameID'] = id_array[:,1].astype(int)
    df['PlayKey'] = df.PlayKey.str.extract(r'([0-9]+$)').astype(int)
    return df

def get_rest(group_):
    days_rest = group_.PlayerDay - group_.PlayerDay.shift()
    return pd.Series(days_rest, name='DaysRest').fillna(0).astype(np.int16)

key_cols = ['PlayerKey', 'GameID', 'PlayKey']


#%%#################
## Plays & Injuries
#

playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
print(playlist.info())
playlist = convert_keys(playlist)
playlist = playlist.fillna({'PlayType': 'unspecified'})


# add new features
playlist['PlayCount'] = playlist.groupby('PlayerKey').cumcount() + 1
playlist['OnSynthetic'] = np.where(playlist.FieldType == "Synthetic", 1, 0)
playlist['PlaysSynthetic'] = playlist.groupby('PlayerKey').OnSynthetic\
                                     .cumsum()
playlist['PlaysNatural'] = playlist.PlayCount - playlist.PlaysSynthetic
playlist['PctPlaysSynthetic'] = playlist.PlaysSynthetic/playlist.PlayCount
playlist['PctPlaysNatural'] = playlist.PlaysNatural/playlist.PlayCount
playlist['PlaysInGame'] = playlist.groupby(key_cols[0:2]).PlayKey\
                                   .transform(max)

# get rest_days
game_ids = playlist.drop_duplicates(key_cols[0:2]).GameID.to_numpy()
games_temp = playlist.drop_duplicates(key_cols[0:2])\
                     .reset_index(drop=True)\
                     .groupby('PlayerKey')\
                     .apply(get_rest)\
                     .reset_index(level=0)\
                     .assign(GameID=game_ids)
playlist = playlist.merge(games_temp, how='left', on=key_cols[0:2])


# Format column
playlist['RosterPosition'] = playlist.RosterPosition.str\
                                        .replace(' ', '_', regex=False)

# Condense stadium types
def condense_stadiums(type_list, cat_, df=playlist):
    idx = playlist.StadiumType.str.lower().str.contains('|'.join(type_list))
    df.loc[idx, 'StadiumType'] = cat_
    return df

unknown = ['cloudy', 'nan']
closed = ['dome', 'ind', 'closed']
open_ = ['out', 'open', 'heinz', 'oudoor', 'ourdoor', 'bowl']

type_list = [unknown, closed, open_]
cats_ = ['unknown', 'closed', 'open']

playlist['StadiumType'] = playlist.StadiumType.astype(str)
for t,c in zip(type_list, cats_):
    playlist = condense_stadiums(t, c)

playlist.loc[playlist.StadiumType == "Retractable Roof", 'StadiumType'] = "unknown"
print(playlist.StadiumType.unique())



# Fix temps
playlist.loc[playlist.Temperature == -999, 'Temperature'] = np.nan
playlist = playlist.fillna({'Temperature': playlist.Temperature.mean()})
playlist['Temperature'] = playlist.Temperature.round(0).astype(int)


# Condense weather
playlist['Weather'] = playlist.Weather.astype(str)
playlist = playlist.fillna({'Weather': 'unknown'})
precips = ['rain', 'shower', 'snow']
precip_idx = playlist.Weather.str.lower()\
                           .str.contains('|'.join(precips))
playlist['Weather'] = "dry"
playlist.loc[precip_idx, 'Weather'] = "wet"
print(playlist.Weather.unique())


# Dummify and percentify
playlist = pd.get_dummies(playlist, columns=['StadiumType', 'Weather'])
dummycols = ['Weather_wet', 'Weather_dry',
             'StadiumType_closed', 'StadiumType_open', 'StadiumType_unknown'
             ]
dummies = playlist.groupby('PlayerKey')[dummycols].transform('sum')
playlist['PctWetWeather'] = dummies.Weather_wet / (playlist.PlayCount)
playlist['PctOpenStadium'] = dummies.StadiumType_open / (playlist.PlayCount)
playlist = playlist.drop(columns=dummycols)


# Check positions
print(playlist.groupby('PlayerKey')['RosterPosition', 'Position', 'PositionGroup']\
        .agg('nunique').hist()) #RosterPosition is most consistent
playlist = playlist.drop(columns=['Position', 'PositionGroup'])


#%%
# Add injury data
injuries = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
print(injuries.info())

# Reformat days and keys
daycols = injuries.columns[injuries.columns.str.startswith('DM')]
injuries['DaysMissed'] = convert_days(injuries, daycols)
injuries = convert_keys(injuries).drop(columns=daycols)

# Replace unknown injury plays with last play of game
last_plays = playlist[['PlayerKey', 'GameID', 'PlaysInGame']]\
                     .drop_duplicates(key_cols[0:2], keep='last')
injuries = injuries.merge(last_plays, how='left', on=key_cols[0:2])
injuries.loc[injuries.PlayKey == 0, 'PlayKey'] = injuries.PlaysInGame
injuries = injuries.drop(columns=['Surface', 'PlaysInGame'])
injuries['Missed1Day'] = np.where(injuries.DaysMissed >=1, 1, 0)
injuries['Missed7Days'] = np.where(injuries.DaysMissed >=7, 1, 0)

print(injuries.Missed1Day.sum()) # this check is good


# merge and add features
playlist = playlist.merge(injuries, how='left', on=key_cols)
playlist['InjuredPlay'] = np.where(playlist.BodyPart.isnull(), 0,
                                            playlist.PlayCount)
playlist['MaxPlayCount'] = playlist.groupby('PlayerKey').PlayCount\
                                       .transform(max)
playlist['MaxPlayInjured'] = playlist.groupby('PlayerKey').InjuredPlay\
                                         .transform(max)
playlist['FinalPlay'] = np.where(playlist.MaxPlayInjured == 0,
                                         playlist.MaxPlayCount,
                                         playlist.MaxPlayInjured)

# cleanup
playlist = playlist.fillna({'BodyPart': 'none',
                                'DaysMissed': 0,
                                'Missed1Day': 0,
                                'Missed7Days': 0
                                })

playlist[['DaysMissed', 'Missed1Day', 'Missed7Days']] = \
    playlist[['DaysMissed', 'Missed1Day', 'Missed7Days']].astype(int)


playlist.Missed1Day.sum() # this check is good

# Redcue memory
mr = skmem.MemReducer()
playlist = mr.fit_transform(playlist)
playlist.sort_values(key_cols).to_parquet('PlayListLabeled.parq')




#%%###############
## NGS Tracks
#

# Use pandas chunker
csize = 4_000_000 #set this to fit your situation
chunker = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv', chunksize=csize)

tracks = []
mr = skmem.MemReducer()
for chunk in tqdm(chunker, total = int(80_000_000/csize)):
    chunk = convert_keys(chunk)
    chunk['event'] = chunk.event.fillna('none')
    floaters = chunk.select_dtypes('float').columns.tolist()
    chunk = mr.fit_transform(chunk, float_cols=floaters)
    tracks.append(chunk)

tracks = pd.concat(tracks)

#%%
col_order = [9,10,0,1,2,3,4,6,8,5,7]
tracks = tracks[[tracks.columns[idx] for idx in col_order]]

tracks['event'] = tracks.event.astype('category')

# two plays stopped on snap - safe to delete 2 rows with null dir, o
tracks[(tracks.PlayerKey == 39715) &\
       (tracks.GameID == 18) &\
       (tracks.PlayKey == 48)]
tracks[(tracks.PlayerKey == 43489) &\
       (tracks.GameID == 26) &\
       (tracks.PlayKey == 53)]
tracks = tracks[~tracks.dir.isnull()].copy()


#%% Create core features
tracks['VelocityIn'] = tracks.dis/0.1 #s in data is too smooth
tracks['dir_diff'] = (tracks.dir-tracks.dir.shift())

tracks['AccelLateral'] = np.abs(tracks.VelocityIn.shift(-1)\
                                      * np.sin(np.deg2rad(tracks.dir_diff))
                                      ).rolling(3).mean() / 0.1
tracks['AccelLong'] = np.abs(tracks.VelocityIn.shift(-1)\
                                   * np.cos(np.deg2rad(tracks.dir_diff))\
                                   - tracks.VelocityIn
                                   ).rolling(3).mean() / 0.1


#%% Make filter for active part of plays
start_events = ['ball_snap', 'snap_direct', 'punt', 'kickoff', 'onside_kick']
end_events = ['tackle', 'out_of_bounds', 'touchdown',
              'pass_outcome_incomplete', 'pass_outcome_touchdown', 
              'fair_catch']
bookends = start_events + end_events
tracks['sig_event'] = np.where(tracks.event.isin(bookends), 1, 0)
tracks['segment'] = tracks.groupby(key_cols)['sig_event'].cumsum()


# Check effectiveness and cut non-active segments
print(tracks.groupby(key_cols).segment.max().value_counts(normalize=True))
tracks = tracks[tracks.segment == 1]

print(tracks.shape)
tracks.reset_index(drop=True).to_parquet('PlayerTrackData.parq')
