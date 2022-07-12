# test by xlinsit
#import what we need
from kaggle.competitions import nflrush
env = nflrush.make_env()

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping



def pre_process(train_df):
    
    # fill some NAs
    train_df.loc[train_df.S.isnull(),'S'] = 2.6
    train_df.loc[train_df.A.isnull(),'A'] = 1.6    
    
    #initial tracking processing from https://www.kaggle.com/statsbymichaellopez/nfl-tracking-wrangling-voronoi-and-sonars
    
    #add a moving left flag and get rusher
    train_df['ToLeft'] = train_df['PlayDirection'] == 'left'
    train_df['IsBallCarrier'] = train_df['NflId'] == train_df['NflIdRusher']

    #correct some naming differences
    train_df.loc[train_df['VisitorTeamAbbr'] == "ARI", 'VisitorTeamAbbr'] = 'ARZ'
    train_df.loc[train_df['HomeTeamAbbr'] == "ARI", 'HomeTeamAbbr'] = 'ARZ'

    train_df.loc[train_df['VisitorTeamAbbr'] == "BAL", 'VisitorTeamAbbr'] = 'BLT'
    train_df.loc[train_df['HomeTeamAbbr'] == "BAL", 'HomeTeamAbbr'] = 'BLT'

    train_df.loc[train_df['VisitorTeamAbbr'] == "CLE", 'VisitorTeamAbbr'] = 'CLV'
    train_df.loc[train_df['HomeTeamAbbr'] == "CLE", 'HomeTeamAbbr'] = 'CLV'

    train_df.loc[train_df['VisitorTeamAbbr'] == "HOU", 'VisitorTeamAbbr'] = 'HST'
    train_df.loc[train_df['HomeTeamAbbr'] == "HOU", 'HomeTeamAbbr'] = 'HST'

    #work out who is on offense
    train_df['TeamOnOffense'] = np.where(train_df['PossessionTeam'] == train_df['HomeTeamAbbr'], 'home', 'away')
    train_df['IsOnOffense'] = train_df['TeamOnOffense'] == train_df['Team']

    #get the standardized yards
    train_df['YardsFromOwnGoal'] = np.where(train_df['FieldPosition'] == train_df['PossessionTeam'], train_df['YardLine'], 50 + (50 - train_df['YardLine']))
    train_df['YardsFromOwnGoal'] = np.where(train_df['YardLine'] == 50, 50, train_df['YardsFromOwnGoal'])


    #get standardized X,Y coordinates
    train_df['X_std'] = np.where(train_df['ToLeft'], 120 - train_df['X'], train_df['X']) - 10
    train_df['Y_std'] = np.where(train_df['ToLeft'], 160 / 3  - train_df['Y'], train_df['Y'])

    #get standardized direction
    train_df['Dir_std_1'] = np.where( (train_df['Dir'] < 90) & train_df['ToLeft'], train_df['Dir'] + 360, train_df['Dir'])
    train_df['Dir_std_1'] = np.where( (train_df['Dir'] > 270) & ~train_df['ToLeft'], train_df['Dir'] - 360, train_df['Dir_std_1'])

    train_df['Dir_std_2'] = np.where(train_df['ToLeft'], train_df['Dir_std_1'] - 180, train_df['Dir_std_1'])
    
    train_df['Dir_std_2'] = train_df['Dir_std_2']*np.pi/180
    
    #fill any na's in the standardized direction
    train_df.loc[train_df.Dir_std_2.isnull(),'Dir_std_2'] = np.pi/2
    
    #Dir_std_2 goes from -90 to 270    
    def x_angle_mult(row):
        if row.Dir_std_2 <= 0 and row.Dir_std_2 >= -np.pi/2:
            return -1 * np.sin( abs(row.Dir_std_2) )
        if row.Dir_std_2 > 0 and row.Dir_std_2 <= np.pi/2:
            return np.sin(row.Dir_std_2)
        if row.Dir_std_2 > np.pi/2 and row.Dir_std_2 <= np.pi:
            return np.sin(np.pi - row.Dir_std_2)
        if row.Dir_std_2 > np.pi and row.Dir_std_2 <= 1.5*np.pi:
            return -1 * np.sin(row.Dir_std_2 - np.pi)
        return 1
    
    def y_angle_mult(row):
        if row.Dir_std_2 <= 0 and row.Dir_std_2 >= -np.pi/2:
            return np.cos( abs(row.Dir_std_2) )
        if row.Dir_std_2 > 0 and row.Dir_std_2 <= np.pi/2:
            return np.cos(row.Dir_std_2)
        if row.Dir_std_2 > np.pi/2 and row.Dir_std_2 <= np.pi:
            return -1 * np.cos(np.pi - row.Dir_std_2)
        if row.Dir_std_2 > np.pi and row.Dir_std_2 <= 1.5*np.pi:
            return -1 * np.cos(row.Dir_std_2 - np.pi)
        return 1
    
    train_df['X_Angle_Mult'] = train_df.apply(x_angle_mult, axis = 1)
    train_df['Y_Angle_Mult'] = train_df.apply(y_angle_mult, axis = 1)

    train_df['X_Speed'] = train_df['S'] * train_df['X_Angle_Mult']
    train_df['Y_Speed'] = train_df['S'] * train_df['Y_Angle_Mult']
    
    train_df['X_Acc'] = train_df['A'] * train_df['X_Angle_Mult']
    train_df['Y_Acc'] = train_df['A'] * train_df['Y_Angle_Mult']
    
    train_df['X_std_new'] = train_df['X_Speed'] + train_df['X_std']
    train_df['Y_std_new'] = train_df['Y_Speed'] + train_df['Y_std']
    
    train_df['X_std_half_new'] = 0.5*train_df['X_Speed'] + train_df['X_std']
    train_df['Y_std_half_new'] = 0.5*train_df['Y_Speed'] + train_df['Y_std']


    # drop columns we don't need
    drop_cols = ['GameId','Team','X','Y','Dis','Orientation','Dir','DisplayName','JerseyNumber','Season','YardLine','Quarter'
                 ,'GameClock','PossessionTeam','Down','FieldPosition','HomeScoreBeforePlay','VisitorScoreBeforePlay','NflIdRusher',
                 'OffensePersonnel','PlayDirection','TimeHandoff','TimeSnap','PlayerBirthDate','PlayerCollegeName','HomeTeamAbbr','VisitorTeamAbbr',
                'Week','Stadium','Location','StadiumType','Turf','ToLeft','WindDirection','WindSpeed','TeamOnOffense','Dir_std_1']

    train_df.drop(drop_cols, axis = 1, inplace = True)

    # work on the play_info df
    play_info_cols = ['PlayId','Distance', 'DefendersInTheBox','YardsFromOwnGoal']

    play_train_df = train_df[play_info_cols]

    #drop dupes
    play_train_df = play_train_df.drop_duplicates()  


    # fill some na's with average value
    play_train_df.loc[play_train_df.DefendersInTheBox.isnull(), 'DefendersInTheBox'] = 7.0
    play_train_df.loc[play_train_df.Distance.isnull(), 'Distance'] = 8.3 
    play_train_df.loc[play_train_df.YardsFromOwnGoal.isnull(), 'YardsFromOwnGoal'] = 50  

    #get rusher to game_info
    rushers = train_df.loc[train_df.IsBallCarrier,['PlayId','X_std','Y_std','X_std_new','Y_std_new','X_std_half_new','Y_std_half_new','S','A','X_Speed','Y_Speed','X_Acc','Y_Acc','Dir_std_2']]
    rushers.columns = ['PlayId','X_std_rush','Y_std_rush','X_std_new_rush','Y_std_new_rush','X_std_half_new_rush','Y_std_half_new_rush','S_Rush','A_Rush','X_Speed_Rush','Y_Speed_Rush',
                       'X_Acc_Rush','Y_Acc_Rush','Dir_std_2_Rush']
    
   
    play_train_df = play_train_df.merge(rushers,how = 'left', on = 'PlayId' )
    play_train_df['DistToLOS_Rusher'] = play_train_df['X_std_rush'] - play_train_df['YardsFromOwnGoal']

    train_df = train_df.merge(rushers[['PlayId','X_std_rush','Y_std_rush','X_std_new_rush','Y_std_new_rush','X_std_half_new_rush','Y_std_half_new_rush','Dir_std_2_Rush']], how = 'left', on = 'PlayId')

    train_df['X_rel_rush'] = train_df['X_std_rush'] - train_df['X_std']
    train_df['Y_rel_rush'] = train_df['Y_std_rush'] - train_df['Y_std']
    
    train_df['X_rel_rush_new'] = train_df['X_std_new_rush'] - train_df['X_std_new']
    train_df['Y_rel_rush_new'] = train_df['Y_std_new_rush'] - train_df['Y_std_new']
    
    train_df['X_rel_rush_half_new'] = train_df['X_std_half_new_rush'] - train_df['X_std_half_new']
    train_df['Y_rel_rush_half_new'] = train_df['Y_std_half_new_rush'] - train_df['Y_std_half_new']

    train_df['DistToRusher'] = np.sqrt( train_df['X_rel_rush']**2 + train_df['Y_rel_rush']**2 )
    train_df['DistToRusherDef'] = train_df.apply(lambda row: row.DistToRusher if not row.IsOnOffense else np.NaN, axis = 1)
    
    train_df['DistToRusherNew'] = np.sqrt( train_df['X_rel_rush_new']**2 + train_df['Y_rel_rush_new']**2 )
    train_df['DistToRusherDefNew'] = train_df.apply(lambda row: row.DistToRusherNew if not row.IsOnOffense else np.NaN, axis = 1)
    
    train_df['DistToRusherHalfNew'] = np.sqrt( train_df['X_rel_rush_half_new']**2 + train_df['Y_rel_rush_half_new']**2 )
    train_df['DistToRusherDefHalfNew'] = train_df.apply(lambda row: row.DistToRusherHalfNew if not row.IsOnOffense else np.NaN, axis = 1)
    

    
    ##
    mean_def_dist   = train_df[['PlayId', 'DistToRusherDef']].groupby('PlayId').agg(DistToRusherDefMean = ('DistToRusherDef',np.nanmean))
    min_def_dist    = train_df[['PlayId', 'DistToRusherDef']].groupby('PlayId').agg(DistToRusherDefMin = ('DistToRusherDef',np.nanmin))
    
    mean_def_dist_new   = train_df[['PlayId', 'DistToRusherDefNew']].groupby('PlayId').agg(DistToRusherDefMeanNew = ('DistToRusherDefNew',np.nanmean))
    min_def_dist_new    = train_df[['PlayId', 'DistToRusherDefNew']].groupby('PlayId').agg(DistToRusherDefMinNew = ('DistToRusherDefNew',np.nanmin))
    
    mean_def_dist_half_new   = train_df[['PlayId', 'DistToRusherDefHalfNew']].groupby('PlayId').agg(DistToRusherDefMeanHalfNew = ('DistToRusherDefHalfNew',np.nanmean))
    min_def_dist_half_new    = train_df[['PlayId', 'DistToRusherDefHalfNew']].groupby('PlayId').agg(DistToRusherDefMinHalfNew = ('DistToRusherDefHalfNew',np.nanmin))
    
    play_train_df = play_train_df.merge(mean_def_dist, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(min_def_dist, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(mean_def_dist_new, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(min_def_dist_new, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(mean_def_dist_half_new, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(min_def_dist_half_new, how = 'left', on = 'PlayId')
    
    train_df.loc[train_df.DistToRusherDef.isnull(),'DistToRusherDef'] = 0
    train_df.loc[train_df.DistToRusherDefNew.isnull(),'DistToRusherDefNew'] = 0
    train_df.loc[train_df.DistToRusherDefHalfNew.isnull(),'DistToRusherDefHalfNew'] = 0
     
    
    def add_boxes(row, xcol, ycol):        
           
        if (row[ycol] >= 2.5) and (row[ycol] <= 7.5) and (row[xcol] <= 0) and (row[xcol] >= -5):
            return 1
        if (row[ycol] >= 2.5) and (row[ycol]<= 7.5) and (row[xcol] <= -5) and (row[xcol] >= -10):
            return 2
        if (row[ycol] >= 2.5) and (row[ycol] <= 7.5) and (row[xcol] <= -10) and (row[xcol] >= -15):
            return 3
        if ( abs(row[ycol]) <= 2.5) and (row[xcol] <= 0) and (row[xcol] >= -5):
            return 4
        if ( abs(row[ycol]) <= 2.5)  and (row[xcol] <= -5) and (row[xcol] >= -10):
            return 5
        if ( abs(row[ycol]) <= 2.5)  and (row[xcol] <= -10) and (row[xcol] >= -15):
            return 6
        if (row[ycol] <= -2.5) and (row[ycol] >= -7.5) and (row[xcol] <= 0) and (row[xcol]>= -5):
            return 7
        if (row[ycol] <= -2.5) and (row[ycol] >= -7.5) and (row[xcol] <= -5) and (row[xcol] >= -10):
            return 8
        if (row[ycol] <= -2.5) and (row[ycol] >= -7.5) and (row[xcol] <= -10) and (row[xcol] >= -15):
            return 9
        return np.NaN
        
        
    train_df['Box'] = train_df.apply(add_boxes, axis = 1,xcol = 'X_rel_rush', ycol = 'Y_rel_rush')
    train_df['BoxNew'] = train_df.apply(add_boxes, axis = 1,xcol = 'X_rel_rush_new', ycol = 'Y_rel_rush_new')
    train_df['BoxHalfNew'] = train_df.apply(add_boxes, axis = 1,xcol = 'X_rel_rush_half_new', ycol = 'Y_rel_rush_half_new')
    
    #defenders in box
    defs_in_b1 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 1),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b2 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 2),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b3 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 3),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b4 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 4),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b5 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 5),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b6 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 6),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b7 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 7),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b8 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 8),['PlayId','Box']].groupby('PlayId').count()
    defs_in_b9 = train_df.loc[(~train_df.IsOnOffense) & (train_df.Box == 9),['PlayId','Box']].groupby('PlayId').count()
    
    defs_in_bcols = ['DefsInb1','DefsInb2','DefsInb3','DefsInb4','DefsInb5','DefsInb6','DefsInb7','DefsInb8','DefsInb9']
    
    defs_in_b1.columns = [defs_in_bcols[0]]
    defs_in_b2.columns = [defs_in_bcols[1]]
    defs_in_b3.columns = [defs_in_bcols[2]]
    defs_in_b4.columns = [defs_in_bcols[3]]
    defs_in_b5.columns = [defs_in_bcols[4]]
    defs_in_b6.columns = [defs_in_bcols[5]]
    defs_in_b7.columns = [defs_in_bcols[6]]
    defs_in_b8.columns = [defs_in_bcols[7]]
    defs_in_b9.columns = [defs_in_bcols[8]]
    
    play_train_df = play_train_df.merge(defs_in_b1, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_b2, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b3, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b4, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b5, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b6, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_b7, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b8, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b9, how = 'left', on = 'PlayId')
    
    
    #defenders in box HALF NEW
    defs_in_b1HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 1),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b2HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 2),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b3HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 3),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b4HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 4),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b5HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 5),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b6HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 6),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b7HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 7),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b8HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 8),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    defs_in_b9HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxHalfNew == 9),['PlayId','BoxHalfNew']].groupby('PlayId').count()
    
    defs_in_bcolsHN = ['DefsInb1HN','DefsInb2HN','DefsInb3HN','DefsInb4HN','DefsInb5HN','DefsInb6HN','DefsInb7HN','DefsInb8HN','DefsInb9HN']
    
    defs_in_b1HN.columns = [defs_in_bcolsHN[0]]
    defs_in_b2HN.columns = [defs_in_bcolsHN[1]]
    defs_in_b3HN.columns = [defs_in_bcolsHN[2]]
    defs_in_b4HN.columns = [defs_in_bcolsHN[3]]
    defs_in_b5HN.columns = [defs_in_bcolsHN[4]]
    defs_in_b6HN.columns = [defs_in_bcolsHN[5]]
    defs_in_b7HN.columns = [defs_in_bcolsHN[6]]
    defs_in_b8HN.columns = [defs_in_bcolsHN[7]]
    defs_in_b9HN.columns = [defs_in_bcolsHN[8]]
    
    play_train_df = play_train_df.merge(defs_in_b1HN, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_b2HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b3HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b4HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b5HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b6HN, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_b7HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b8HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b9HN, how = 'left', on = 'PlayId')
    
    #defenders in box NEW
    defs_in_b1N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 1),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b2N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 2),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b3N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 3),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b4N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 4),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b5N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 5),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b6N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 6),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b7N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 7),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b8N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 8),['PlayId','BoxNew']].groupby('PlayId').count()
    defs_in_b9N = train_df.loc[(~train_df.IsOnOffense) & (train_df.BoxNew == 9),['PlayId','BoxNew']].groupby('PlayId').count()
    
    defs_in_bcolsN = ['DefsInb1N','DefsInb2N','DefsInb3N','DefsInb4N','DefsInb5N','DefsInb6N','DefsInb7N','DefsInb8N','DefsInb9N']
    
    defs_in_b1N.columns = [defs_in_bcolsN[0]]
    defs_in_b2N.columns = [defs_in_bcolsN[1]]
    defs_in_b3N.columns = [defs_in_bcolsN[2]]
    defs_in_b4N.columns = [defs_in_bcolsN[3]]
    defs_in_b5N.columns = [defs_in_bcolsN[4]]
    defs_in_b6N.columns = [defs_in_bcolsN[5]]
    defs_in_b7N.columns = [defs_in_bcolsN[6]]
    defs_in_b8N.columns = [defs_in_bcolsN[7]]
    defs_in_b9N.columns = [defs_in_bcolsN[8]]
    
    play_train_df = play_train_df.merge(defs_in_b1N, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_b2N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b3N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b4N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b5N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b6N, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_b7N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b8N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_b9N, how = 'left', on = 'PlayId')    
    

    # how many defenders within distances
    defs_in_1 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 1),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_2 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 2),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_3 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 3),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_4 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 4),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_5 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 5),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_6 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 6),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_7 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 7),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_8 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 8),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_9 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 9),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()
    defs_in_10 = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDef < 10),['PlayId','DistToRusherDef'] ].groupby('PlayId').count()

    defs_in_cols = ['DefsIn1','DefsIn2','DefsIn3','DefsIn4','DefsIn5','DefsIn6','DefsIn7','DefsIn8','DefsIn9','DefsIn10']

    defs_in_1.columns = [defs_in_cols[0]]
    defs_in_2.columns = [defs_in_cols[1]]
    defs_in_3.columns = [defs_in_cols[2]]
    defs_in_4.columns = [defs_in_cols[3]]
    defs_in_5.columns = [defs_in_cols[4]]
    defs_in_6.columns = [defs_in_cols[5]]
    defs_in_7.columns = [defs_in_cols[6]]
    defs_in_8.columns = [defs_in_cols[7]]
    defs_in_9.columns = [defs_in_cols[8]]
    defs_in_10.columns = [defs_in_cols[9]]

    play_train_df = play_train_df.merge(defs_in_1, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_2, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_3, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_4, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_5, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_6, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_7, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_8, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_9, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_10, how = 'left', on = 'PlayId')
    
    
    # how many defenders within distances HALF NEW
    defs_in_1HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 1),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_2HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 2),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_3HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 3),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_4HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 4),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_5HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 5),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_6HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 6),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_7HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 7),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_8HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 8),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_9HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 9),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()
    defs_in_10HN = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefHalfNew < 10),['PlayId','DistToRusherDefHalfNew'] ].groupby('PlayId').count()

    defs_in_colsHN = ['DefsIn1HN','DefsIn2HN','DefsIn3HN','DefsIn4HN','DefsIn5HN','DefsIn6HN','DefsIn7HN','DefsIn8HN','DefsIn9HN','DefsIn10HN']

    defs_in_1HN.columns = [defs_in_colsHN[0]]
    defs_in_2HN.columns = [defs_in_colsHN[1]]
    defs_in_3HN.columns = [defs_in_colsHN[2]]
    defs_in_4HN.columns = [defs_in_colsHN[3]]
    defs_in_5HN.columns = [defs_in_colsHN[4]]
    defs_in_6HN.columns = [defs_in_colsHN[5]]
    defs_in_7HN.columns = [defs_in_colsHN[6]]
    defs_in_8HN.columns = [defs_in_colsHN[7]]
    defs_in_9HN.columns = [defs_in_colsHN[8]]
    defs_in_10HN.columns = [defs_in_colsHN[9]]

    play_train_df = play_train_df.merge(defs_in_1HN, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_2HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_3HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_4HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_5HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_6HN, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_7HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_8HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_9HN, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_10HN, how = 'left', on = 'PlayId')
    
    
    # how many defenders within distances NEW
    defs_in_1N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 1),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_2N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 2),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_3N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 3),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_4N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 4),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_5N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 5),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_6N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 6),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_7N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 7),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_8N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 8),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_9N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 9),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()
    defs_in_10N = train_df.loc[(~train_df.IsOnOffense) & (train_df.DistToRusherDefNew < 10),['PlayId','DistToRusherDefNew'] ].groupby('PlayId').count()

    defs_in_colsN = ['DefsIn1N','DefsIn2N','DefsIn3N','DefsIn4N','DefsIn5N','DefsIn6N','DefsIn7N','DefsIn8N','DefsIn9N','DefsIn10N']

    defs_in_1N.columns = [defs_in_colsN[0]]
    defs_in_2N.columns = [defs_in_colsN[1]]
    defs_in_3N.columns = [defs_in_colsN[2]]
    defs_in_4N.columns = [defs_in_colsN[3]]
    defs_in_5N.columns = [defs_in_colsN[4]]
    defs_in_6N.columns = [defs_in_colsN[5]]
    defs_in_7N.columns = [defs_in_colsN[6]]
    defs_in_8N.columns = [defs_in_colsN[7]]
    defs_in_9N.columns = [defs_in_colsN[8]]
    defs_in_10N.columns = [defs_in_colsN[9]]

    play_train_df = play_train_df.merge(defs_in_1N, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_2N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_3N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_4N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_5N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_6N, how = 'left', on = 'PlayId') 
    play_train_df = play_train_df.merge(defs_in_7N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_8N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_9N, how = 'left', on = 'PlayId')
    play_train_df = play_train_df.merge(defs_in_10N, how = 'left', on = 'PlayId')    
    

    all_count_cols = defs_in_cols + defs_in_colsN + defs_in_bcols + defs_in_bcolsN + defs_in_bcolsHN + defs_in_colsHN
    for col in all_count_cols:
        play_train_df.loc[play_train_df[col].isnull(),col] = 0.0
        
    #check nulls
    play_train_df.isnull().sum()

    play_train_df.drop('PlayId', axis = 1, inplace = True)
    
    return play_train_df


#####
#####
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

#get our output and drop it from the training data
output = train_df[['PlayId','Yards']]
output.drop_duplicates(inplace = True)
output = output['Yards']

train_df.drop('Yards', axis = 1, inplace = True)

yards_lower_bound = 99 #negative 99
output_len = 100 + yards_lower_bound

y = np.zeros((output.shape[0], output_len))
for idx, target in enumerate(list(output)):
    y[idx][yards_lower_bound + target] = 1

#do the preprocessing
play_train_df = pre_process(train_df)

#standard scale the true numerical variables
play_train_num_cols = ['Distance','DefendersInTheBox', 'YardsFromOwnGoal',
       'X_std_rush', 'Y_std_rush', 'X_std_new_rush',
       'Y_std_new_rush', 'X_std_half_new_rush', 'Y_std_half_new_rush',
       'S_Rush', 'A_Rush', 'X_Speed_Rush', 'Y_Speed_Rush', 'X_Acc_Rush',
       'Y_Acc_Rush','DistToLOS_Rusher',
       'DistToRusherDefMean', 'DistToRusherDefMin', 'DistToRusherDefMeanNew',
       'DistToRusherDefMinNew', 'DistToRusherDefMeanHalfNew',
       'DistToRusherDefMinHalfNew', 'DefsInb1',
       'DefsInb2', 'DefsInb3', 'DefsInb4', 'DefsInb5', 'DefsInb6', 'DefsInb7',
       'DefsInb8', 'DefsInb9', 'DefsInb1HN', 'DefsInb2HN', 'DefsInb3HN',
       'DefsInb4HN', 'DefsInb5HN', 'DefsInb6HN', 'DefsInb7HN', 'DefsInb8HN',
       'DefsInb9HN', 'DefsInb1N', 'DefsInb2N', 'DefsInb3N', 'DefsInb4N',
       'DefsInb5N', 'DefsInb6N', 'DefsInb7N', 'DefsInb8N', 'DefsInb9N',
       'DefsIn1', 'DefsIn2', 'DefsIn3', 'DefsIn4', 'DefsIn5', 'DefsIn6',
       'DefsIn7', 'DefsIn8', 'DefsIn9', 'DefsIn10', 'DefsIn1HN', 'DefsIn2HN',
       'DefsIn3HN', 'DefsIn4HN', 'DefsIn5HN', 'DefsIn6HN', 'DefsIn7HN',
       'DefsIn8HN', 'DefsIn9HN', 'DefsIn10HN', 'DefsIn1N', 'DefsIn2N',
       'DefsIn3N', 'DefsIn4N', 'DefsIn5N', 'DefsIn6N', 'DefsIn7N', 'DefsIn8N',
       'DefsIn9N', 'DefsIn10N']

play_train_scaler = StandardScaler()

play_train_df[play_train_num_cols] = play_train_scaler.fit_transform(play_train_df[play_train_num_cols])

play_train_arr = play_train_df.values
play_train_arr.shape

full_train = play_train_arr



# CRPS for validation
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 

## Start building the model

#Class based on https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (output_len * X_train.shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (output_len * X_valid.shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


                      

def set_model():    

    model = Sequential()
    model.add(Dense(128, input_dim = full_train.shape[1], activation ='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(output_len, activation ='softmax'))
    
    return model


num_model_runs = 1
n_splits = 10
scores = []
models = []

full_model = True

if full_model:
    model = set_model()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
    model.fit(full_train, y, epochs = 100, batch_size=256)
    models.append(model)
else:
    for i in range(0,num_model_runs):

        print('Running KFold : ', i + 1)
        scores_single = []
        kf = KFold(n_splits = n_splits, shuffle = True, random_state = 54321) #random_state 54321

        for (tdx,vdx) in kf.split(full_train, y):   

            X_train, X_val, y_train, y_val = full_train[tdx], full_train[vdx], y[tdx], y[vdx] 

            model = set_model()
            model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])

            es = EarlyStopping(monitor='val_CRPS', 
                               mode='min',
                               restore_best_weights=True, 
                               verbose=0, 
                               patience=5)
            es.set_model(model)

            metric = Metric(model, [es], [(X_train,y_train), (X_val,y_val)])

            model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=256)

            predict_vals = model.predict(X_val)

            score = crps(y_val, predict_vals)

            scores_single.append(score)
            print('Running Mean CSRP : ', np.mean(scores_single))
            models.append(model)

        print('Mean CSRP for run:', i+1 , np.mean(scores_single))
        scores.append(np.mean(scores_single))      

    

iter_test = env.iter_test()
i=1
for (test_df, sample_prediction_df) in iter_test:
    print('Test case:',i)
    basetable_play = pre_process(test_df)
    
    basetable_play[play_train_num_cols] = play_train_scaler.transform(basetable_play[play_train_num_cols])
    
    test_plays =  basetable_play.values

    full_test = test_plays
    
    y_pred = np.mean([model.predict(full_test) for model in models], axis=0)
       
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    
    i += 1
    
    env.predict(preds_df)
    
env.write_submission_file()