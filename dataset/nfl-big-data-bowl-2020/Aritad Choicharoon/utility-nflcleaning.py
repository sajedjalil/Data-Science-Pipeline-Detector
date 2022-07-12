# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
from scipy import stats
from plotnine import *
import nltk
import datetime 


data = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')

### https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt - Probably even older source

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def CalcOutliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return

data = reduce_mem_usage(data)

### Player Level Team

data['Team'] = np.where(data['Team']=='away',data.VisitorTeamAbbr,data.HomeTeamAbbr)
data['OffenseTeam'] = np.where(data['Team']==data['PossessionTeam'],1,0)
data['DefenseTeam'] = np.where(data['Team']==data['PossessionTeam'],0,1)

### Time Until End

data['SecondsUntilEnd'] = data.GameClock.apply(lambda x:(int(x[0:2])*60)+int(x[3:5]))
data['TempSecondsUntilEnd'] = data['SecondsUntilEnd'] 
data['SecondsUntilEnd'] = data['SecondsUntilEnd']+(15*60)*(4-data.Quarter)
data['SecondsUntilEnd'] = np.where(data.Quarter>4, data['TempSecondsUntilEnd'],data['SecondsUntilEnd'])
data['ExtraTime'] = np.where(data.Quarter>4, 1,0)

### Time Until HalfTime

data['SecondsUntilHalfTime'] = data['SecondsUntilEnd'] 
data['SecondsUntilHalfTime'] = np.where(data.Quarter<3,data['SecondsUntilHalfTime']-(60*30),data['SecondsUntilHalfTime'])

### Time Until ChangeOver

data['SecondsUntilQuarter'] = data.GameClock.apply(lambda x:int(x[0:2])*60+int(x[3:5]))
data = data.drop(columns=['GameClock'])

data['PlayerHeightInches'] = data.PlayerHeight.apply(lambda x:(int(x[0])*60)+int(x[2]))

data['Age'] = data.PlayerBirthDate.apply(lambda x: 2019 - int(x.split('/')[-1]))

def windspeedclean(x,y):
    if((x == 'E')|(x == 'SE')|(x == 'SSW')):
        return (float(y))
    elif(x == 'Calm'):
        return (0.0)
    else:
        return (float(str(x).lower().replace('mph','').split('-')[0].split(' ')[0]))

data['WindSpeed'] = data.apply(lambda x: windspeedclean(x['WindSpeed'], x['WindDirection']),axis=1)

Positional = pd.DataFrame(data.groupby(['PlayId','Position']).GameId.count()).reset_index()
positionplay = Positional.pivot(index='PlayId', columns='Position', values='GameId').fillna(0).reset_index()

data = pd.merge(data, positionplay, on='PlayId',how='left')

outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field','Outdor', 'Ourdoor', 'Outside', 'Outddors', 'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed','Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

indoor_open = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

dome_closed = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

dome_open = ['Domed, Open', 'Domed, open']
    

def convert_stadium(stadium):
    if(stadium in outdoor):
        return "outdoor"
    elif(stadium in indoor_closed):
        return "indoor_closed"
    elif(stadium in indoor_open):
        return "indoor_open"
    elif(stadium in dome_closed):
        return "dome_closed"
    elif(stadium in dome_open):
        return "dome_open"
    else:
        return 'unknown'

data['StadiumType'] = data['StadiumType'].apply(lambda x:convert_stadium(x))
    
def clean_turf(inputturf):
    inputturf = inputturf.upper()
    if(inputturf == "ARTIFICAL"):
        return "ARTIFICAL"
    elif(inputturf == 'FIELDTURF'):
        return "FIELD TURF"
    elif(inputturf == 'FIELDTURF360'):
        return "FIELD TURF 360"
    elif(inputturf == 'NATURAL' or inputturf == 'NATURAL GRASS' or inputturf == 'NATURALL GRASS'):
        return "GRASS"
    elif(inputturf == 'UBU SPORTS SPEED S5-M'):
        return "UBU SPEED SERIES-S5-M"
    else:
        return inputturf
    
data['Turf'] = data['Turf'].apply(lambda x:clean_turf(x))
    
def clean_wd(inputwind):
    if(inputwind == "N" or inputwind == "FROM S"):
        return "north"
    elif(inputwind == 'S' or inputwind == 'FROM N'):
        return "south"
    elif(inputwind == 'W' or inputwind == 'FROM E'):
        return "west"
    elif(inputwind == 'E' or inputwind == 'FROM W'):
        return "east"
    elif(inputwind == 'FROM SW' or inputwind == 'FROM SSW' or inputwind == 'FROM WSW'):
        return "north east"
    elif(inputwind == 'FROM SE' or inputwind == 'FROM SSE' or inputwind == 'FROM ESE'):
        return "north west"
    elif(inputwind == 'FROM NW' or inputwind == 'FROM NNW' or inputwind == 'FROM WNW'):
        return "south east"
    elif(inputwind == 'FROM NE' or inputwind == 'FROM NNE' or inputwind == 'FROM ENE'):
        return "south west"
    elif(inputwind == 'NW' or inputwind == 'NORTHWEST'):
        return "north west"
    elif(inputwind == 'NE' or inputwind == 'NORTH EAST'):
        return "north east"
    elif(inputwind == 'SW' or inputwind == 'SOUTHWEST'):
        return "south west"
    elif(inputwind == 'NE' or inputwind == 'SOUTHEAST'):
        return "south east"
    else:
        return 'unknown'

data['WindDirection'] = data['WindDirection'].apply(lambda x:clean_wd(x))

rain = ['Rainy', 'Rain Chance 40%', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']

overcast = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',\
              'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',\
              'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',\
              'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',\
              'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',\
              'Partly Cloudy', 'Cloudy']

clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',\
           'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',\
           'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',\
           'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',\
           'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',\
           'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny']

snow = ['Heavy lake effect snow', 'Snow']

none = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']

def convert_weather(inputweather):
    if(inputweather in rain):
        return "rain"
    elif(inputweather in overcast):
        return "overcast"
    elif(inputweather in clear):
        return "clear"
    elif(inputweather in snow):
        return "snow"
    elif(inputweather in none):
        return "indoors"
    else:
        return 'unknown'
    
    
data['GameWeather'] = data['GameWeather'].apply(lambda x:convert_weather(x))

data = reduce_mem_usage(data)

print(data.columns)

data.columns = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation',
       'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'Season', 'YardLine',
       'Quarter', 'PossessionTeam', 'Down', 'Distance', 'FieldPosition',
       'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'NflIdRusher',
       'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox',
       'DefensePersonnel', 'PlayDirection', 'TimeHandoff', 'TimeSnap', 'Yards',
       'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'PlayerCollegeName',
       'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium',
       'Location', 'StadiumType', 'Turf', 'GameWeather', 'Temperature',
       'Humidity', 'WindSpeed', 'WindDirection', 'OffenseTeam', 'DefenseTeam',
       'SecondsUntilEnd', 'TempSecondsUntilEnd', 'ExtraTime',
       'SecondsUntilHalfTime', 'SecondsUntilQuarter', 'PlayerHeightInches',
       'Age', 'C', 'CB', 'DB', 'DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB',
       'LB', 'MLB', 'NT', 'OG', 'OLB', 'OT', 'QB', 'RB', 'S_Position', 'SAF', 'SS',
       'T', 'TE', 'WR']

PlayerLevel = pd.DataFrame(data.groupby(['GameId','PlayId','X','Y','S','A','Dis','Orientation',\
                                        'Dir','NflId','DisplayName','JerseyNumber','PlayerHeightInches',\
                                         'PlayerWeight','PlayerBirthDate','PlayerCollegeName',\
                                         'Position','Age']).Yards.count()).reset_index()

PlayLevel = pd.DataFrame(data.groupby(['GameId','PlayId','Season','YardLine','Quarter','PossessionTeam',\
                         'Down','Distance','FieldPosition','HomeScoreBeforePlay','VisitorScoreBeforePlay',\
                                      'OffenseFormation','DefendersInTheBox',\
                         'PlayDirection','Yards','Week','Stadium',\
                         'StadiumType','Turf','GameWeather','Temperature','Humidity',\
                          'WindDirection','SecondsUntilEnd','ExtraTime',\
       'SecondsUntilHalfTime', 'SecondsUntilQuarter','WindSpeed','C', 'CB', 'DB',\
                                       'DE', 'DL', 'DT', 'FB', 'FS',\
       'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG', 'OLB', 'OT', 'QB', 'RB',\
       'S_Position', 'SAF', 'SS', 'T', 'TE', 'WR']).X.count()).reset_index()

PlayLevel['LogYards'] = np.log(np.abs(PlayLevel['Yards']))*(PlayLevel['Yards']/np.abs(PlayLevel['Yards']))


# data.head()
# data.to_csv('train_clean.csv',index=False)
# PlayLevel.to_csv('play.csv',index=False)
# PlayerLevel.to_csv('player.csv',index=False)