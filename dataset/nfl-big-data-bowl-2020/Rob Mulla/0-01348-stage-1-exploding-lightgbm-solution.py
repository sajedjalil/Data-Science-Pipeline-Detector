"""
Created by Rob Mulla
Github: @RobMulla

- LightGBM Model
- Explode each possible yard into 199 rows
- Binary classification
- 0.01348 public LB
- 2200 iterations determined by CV offline
"""
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import os
from datetime import datetime
import time
import logging
import sys
import getpass
import gc
from tqdm import tqdm
#####################
# INPUT PARAMETERS
#####################
TRIAL_RUN = False
N_THREADS = 32
N_FOLDS = 5
RANDOM_STATE = 529
LEARNING_RATE = 0.01
KAGGLE_N_ESTIMATORS = 2500
EARLY_STOPPING_ROUNDS = 600
TRAIN_YARDS = [-99, -15, -14, -11, -10, -9, -8, -7, -6, -5,
               -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
               10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25,
               30, 35, 40, 50, 60, 80, 99]
ONLY_2018 = False
##########################################
# Determine if Running on Kaggle or Local
##########################################
RUN_ID = "{:%m%d_%H%M}".format(datetime.now())
user = getpass.getuser()
if user == 'robmulla':
    RUNNING_ON_KAGGLE = False
    MODEL_NUMBER = os.path.basename(__file__).split('.')[0]
    OOF_DIR = '../oof'
    SUB_DUR = '../sub'
    FI_DIR = '../fi'
    MODEL_DIR = '../models/'
    TQDM_MININTERVAL = 0.1
elif user == 'root':
    RUNNING_ON_KAGGLE = True
    MODEL_NUMBER = f'KERNEL{RUN_ID}'
    OOF_DIR = './'
    SUB_DIR = './'
    FI_DIR = './'
    MODEL_DIR = './'
    if TRIAL_RUN:
        KAGGLE_N_ESTIMATORS = 200
        TQDM_MININTERVAL = 0.1
    else:
        TQDM_MININTERVAL = 60
#####################
# TRACKING FUNCTION
#####################
def update_tracking(field,
                    value,
                    run_id=RUN_ID,
                    csv_file="../tracking/tracking.csv",
                    integer=False,
                    digits=None,
                    drop_incomplete_rows=False):
    """
    Function to update the tracking CSV with information about the model
    """
    if RUNNING_ON_KAGGLE:
        csv_file = 'tracking.csv'
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except FileNotFoundError:
        df = pd.DataFrame()
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    if drop_incomplete_rows and len(df) > 0:
        df = df.loc[~df['mean_crps'].isna()]
        df = df.loc[~df['trial_run']]
    df.loc[run_id, field] = value  # Model number is index
    df.to_csv(csv_file)
#####################
## SETUP LOGGER
#####################
def get_logger():
    """
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    os.environ["TZ"] = "US/Eastern"
    time.tzset()
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    if RUNNING_ON_KAGGLE:
        fhandler = logging.FileHandler(f'{MODEL_NUMBER}_{RUN_ID}.log')
    else:
        fhandler = logging.FileHandler(f'../logs/{MODEL_NUMBER}_{RUN_ID}.log')
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)
    fhandler.setFormatter(formatter)
    if RUNNING_ON_KAGGLE:
        logger.addHandler(handler)
    logger.addHandler(fhandler)
    return logger
logger = get_logger()
#####################
## Read input data
#####################
if RUNNING_ON_KAGGLE:
    from kaggle.competitions import nflrush
    env = nflrush.make_env()
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
else:
    train = pd.read_csv('../input/train.csv', low_memory=False)
if TRIAL_RUN:
    logger.warning('===== Trial Run. Only Running for 5 Games ======')
    train = train.loc[train['GameId'].isin(train['GameId'].unique()[:5])]
    logger.info(f'Train shape is {train.shape}')
if ONLY_2018:
    train = train.query('Season == 2018').copy()
##########################################
# Update Tracking with Basic Model Info
##########################################
update_tracking("model_number", MODEL_NUMBER, drop_incomplete_rows=True)
update_tracking("trial_run", TRIAL_RUN)
update_tracking("random_state", RANDOM_STATE)
update_tracking("n_threads", N_THREADS)
update_tracking("learning_rate", LEARNING_RATE)
update_tracking("n_fold", N_FOLDS)
update_tracking("n_train_yards", len(TRAIN_YARDS))
##########################################
# Feature Engineering Functions
##########################################
def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0

def get_angle(row, ro, rd):
    Xa = row[f'X_std_{rd}rank_defense']
    Ya = row[f'Y_std_{rd}rank_defense']
    Xb = row['X_std_rusher']
    Yb = row['Y_std_rusher']
    Xc = row[f'X_std_{ro}rank_offense']
    Yc = row[f'Y_std_{ro}rank_offense']
    a = np.array([Xa,Ya])
    b = np.array([Xb,Yb])
    c = np.array([Xc,Yc])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_vectorized_features(df, print_mem=False, test=False):
    """
    df : data in format of 22 lines per play (one per player)
    plays : data in format of 1 line per play
    """
    #################
    # FIX TEAM NAMES
    #################
    df.loc[df['PossessionTeam'] == 'ARZ', 'PossessionTeam'] = 'ARI'
    df.loc[df['PossessionTeam'] == 'BLT', 'PossessionTeam'] = 'BAL'
    df.loc[df['PossessionTeam'] == 'CLV', 'PossessionTeam'] = 'CLE'
    df.loc[df['PossessionTeam'] == 'HST', 'PossessionTeam'] = 'HOU'
    df.loc[df['FieldPosition'] == 'ARZ', 'FieldPosition'] = 'ARI'
    df.loc[df['FieldPosition'] == 'BLT', 'FieldPosition'] = 'BAL'
    df.loc[df['FieldPosition'] == 'CLV', 'FieldPosition'] = 'CLE'
    df.loc[df['FieldPosition'] == 'HST', 'FieldPosition'] = 'HOU'
    plays = df[['GameId','PlayId']].drop_duplicates()
    # Fix Speed Bug for 2017
    df.loc[df['Season'] == 2017, 'S'] = (df['S'][df['Season'] == 2017] - 2.4355) / 1.2930 * 1.4551 + 2.7570
    ###################################################
    # Standardized Player Level Features
    # IsBallCarrier
    # IsOnOffense
    # X_std, Y_std, Orientation_std, Dir_std
    ###################################################
    df['ToLeft'] = df.PlayDirection == "left"
    df['IsBallCarrier'] = df.NflId == df.NflIdRusher
    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df.loc[df['PossessionTeam'] == df['HomeTeamAbbr'], 'DefendingTeam'] = df['VisitorTeamAbbr']
    df.loc[df['PossessionTeam'] == df['VisitorTeamAbbr'], 'DefendingTeam'] = df['HomeTeamAbbr']
    df['IsOnOffense'] = df.Team == df.TeamOnOffense # Is player on offense?
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
          'YardLine_std'
         ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
          'YardLine']
    df['X_std'] = df['X']
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df['Y']
    df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y']
    df['Dir_rad_std'] = df['Dir_rad']
    df.loc[df.ToLeft, 'Dir_rad_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)
    df['Dir_std'] = df.Dir
    # df['Dir_std_180_shift'] = (df['Dir_std'] + 180).apply(lambda x: x if x <= 360 else x - 360)
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(180 + df.loc[df.ToLeft, 'Dir_std'], 360)
    ###################################################
    # YardLine, First Down, and Distance from Players X
    ####################################################
    df['FirstDownLine_std'] = df['YardLine_std'] + df['Distance']
    df['X_YardLine_std'] = df['YardLine_std'] + 10
    df['X_FirstDownLine_std'] = df['FirstDownLine_std'] + 10
    df['X_relative_YardLine'] = df['X_std'] - df['X_YardLine_std']
    df['X_relative_FirstDown'] = df['X_std'] - df['X_FirstDownLine_std']
    ####################################
    # Offensive and Defensive centroid
    ###################################
    temp = df.query('not IsOnOffense').groupby('PlayId').agg({'X_std': ['mean'],
                                                              'Y_std': ['mean']})
    temp.columns = ['_'.join(col).strip()+'_defense' for col in temp.columns.values]
    temp = temp.reset_index()
    df = df.merge(temp, on='PlayId', how='left')
    temp = df.query('IsOnOffense').groupby('PlayId').agg({'X_std': ['mean'],
                                                              'Y_std': ['mean']})
    temp.columns = ['_'.join(col).strip()+'_offense' for col in temp.columns.values]
    temp = temp.reset_index()
    df = df.merge(temp, on='PlayId', how='left')
    df['distance_to_offensive_centriod'] = np.sqrt( (df.X_std-df.X_std_mean_offense)**2 + \
                                                     (df.X_std-df.X_std_mean_offense)**2 )
    df['distance_to_defensive_centriod'] = np.sqrt( (df.X_std-df.X_std_mean_offense)**2 + \
                                                     (df.X_std-df.X_std_mean_offense)**2 )
    ##################################################
    # Using Velocity to predict positions later frames
    ##################################################
    df['v_horizontal'] = df['S'] * np.cos(df['Dir_rad_std'])
    df['v_vertical'] = df['S'] * np.sin(df['Dir_rad_std'])
    df['X_std_1frame'] = df['X_std'] + df['v_vertical']
    df['Y_std_1frame'] = df['Y_std'] + df['v_horizontal']
    df['X_std_2frame'] = df['X_std'] + (2 * df['v_vertical'])
    df['Y_std_2frame'] = df['Y_std'] + (2 * df['v_horizontal'])
    df['X_std_3frame'] = df['X_std'] + (3 * df['v_vertical'])
    df['Y_std_3frame'] = df['Y_std'] + (3 * df['v_horizontal'])
    df['X_std_4frame'] = df['X_std'] + (4 * df['v_vertical'])
    df['Y_std_4frame'] = df['Y_std'] + (4 * df['v_horizontal'])
    df['X_std_5frame'] = df['X_std'] + (5 * df['v_vertical'])
    df['Y_std_5frame'] = df['Y_std'] + (5 * df['v_horizontal'])
    #################
    # Add Rusher Info
    #################
    # plays = df[df['NflId'] == df['NflIdRusher']]
    df = pd.merge(df, df.query('NflId == NflIdRusher')[['PlayId',
                                                        'PlayerWeight','A','S',
                                                        'Dir_std',
                                                        'X_std', 'Y_std',
                                                        'X_relative_YardLine',
                                                        'X_relative_FirstDown',
                                                        'distance_to_offensive_centriod',
                                                        'distance_to_defensive_centriod',
                                                        'X_std_1frame','Y_std_1frame',
                                                        'X_std_2frame','Y_std_2frame',
                                                        'X_std_3frame','Y_std_3frame',
                                                        'X_std_4frame','Y_std_4frame',
                                                        'X_std_5frame','Y_std_5frame']
                                                      ],
                    on='PlayId',
                    suffixes=('', '_rusher'))
    df['distance_to_runner'] = np.sqrt( (df.X_std-df.X_std_rusher)**2 + (df.Y_std-df.Y_std_rusher)**2 )
    df['frame1_distance_to_runner'] = np.sqrt( (df.X_std_1frame-df.X_std_1frame_rusher)**2 + (df.Y_std_1frame-df.Y_std_1frame_rusher)**2 )
    df['frame2_distance_to_runner'] = np.sqrt( (df.X_std_2frame-df.X_std_2frame_rusher)**2 + (df.Y_std_2frame-df.Y_std_2frame_rusher)**2 )
    df['frame3_distance_to_runner'] = np.sqrt( (df.X_std_3frame-df.X_std_3frame_rusher)**2 + (df.Y_std_3frame-df.Y_std_3frame_rusher)**2 )
    df['frame4_distance_to_runner'] = np.sqrt( (df.X_std_4frame-df.X_std_4frame_rusher)**2 + (df.Y_std_4frame-df.Y_std_4frame_rusher)**2 )
    df['frame5_distance_to_runner'] = np.sqrt( (df.X_std_5frame-df.X_std_5frame_rusher)**2 + (df.Y_std_5frame-df.Y_std_5frame_rusher)**2 )
    ########################
    # Time to Tackle Rusher
    ########################
    df['TimeToTackle'] = df['S'] / df['distance_to_runner']
    ######################
    # Reduce to Play Level
    ######################
    play_data = df.query('NflId == NflIdRusher').reset_index(drop=True).copy()
    plays = df[['GameId','PlayId']].drop_duplicates().reset_index(drop=True)
    plays['GameId'] = plays['GameId']
    plays['PlayId'] = plays['PlayId']
    plays['DefendersInTheBox'] = play_data['DefendersInTheBox']
    plays['OffenseFormation'] = play_data['OffenseFormation']
    plays['Distance'] = play_data['Distance']
    plays['Down'] = play_data['Down']
    # Yardline and Firstdownline
    # plays['YardLine'] = play_data['YardLine'] # Same thing as DistanceFromEndzone
    plays['FirstDownLine_std'] = play_data['FirstDownLine_std']
    plays['X_YardLine_std'] = play_data['X_YardLine_std']
    # plays['X_FirstDownLine_std'] = play_data['X_FirstDownLine_std']
    plays['FirstDownLine_std'] = play_data['FirstDownLine_std']
    if not test:
        plays['Yards'] = play_data['Yards']
        ### Change to Yards + X_distance_from
    ##############
    # Add Rusher Info
    #################
    df_rusher = df.query("NflId == NflIdRusher") \
        .groupby('PlayId').first().reset_index()[['PlayId',
                                                  # 'PlayerWeight',
                                                  'X_relative_YardLine',
                                                  # 'X_relative_FirstDown',
                                                  # 'distance_to_offensive_centriod','distance_to_defensive_centriod'
                                                  ]]
    df_rusher.columns = [x+'_rusher' for x in df_rusher.columns]
    plays = pd.merge(plays, df_rusher,
                    left_on='PlayId',
                     right_on='PlayId_rusher',
                    how='left',
                    suffixes=('', '_rusher'))
    plays = plays.drop('PlayId_rusher', axis=1)
    #################
    # Add QB Info
    #################
    df_qb = df.query("Position == 'QB'") \
        .groupby('PlayId').first().reset_index()[['PlayId',
                                                  #'PlayerWeight','A',
                                                  'S',
                                                  'Dir_std',
                                                  # 'X_std', 'Y_std',
                                                  # 'X_relative_YardLine', 'X_relative_FirstDown'
                                                  ]]
    df_qb.columns = [x+'_qb' for x in df_qb.columns]
    plays = pd.merge(plays, df_qb,
                    left_on='PlayId',
                     right_on='PlayId_qb',
                    how='left',
                    suffixes=('', '_qb'))
    plays = plays.drop('PlayId_qb', axis=1)
    #################
    # Add Runner Info
    #################
    plays['X_std_rusher'] = play_data['X_std_rusher']
    plays['Y_std_rusher'] = play_data['Y_std_rusher']
    plays['Dir_std_rusher'] = play_data['Dir_std_rusher']
    plays['A_rusher'] = play_data['A_rusher']
    plays['S_rusher'] = play_data['S_rusher']
    # ##############################
    # # Parse the Gameclock
    # ##############################
#     play_data['seconds_left_in_quarter'] = (pd.to_datetime(play_data['GameClock']).dt.minute * 60) + \
#                                         (pd.to_datetime(play_data['GameClock']).dt.second)
#     play_data['seconds_left_in_game'] = ((4 - play_data.Quarter) * (15 * 60)) + play_data['seconds_left_in_quarter']
#     play_data['seconds_left_in_half'] = play_data['seconds_left_in_game']
#     play_data.loc[play_data['Quarter'].isin([1, 2]), 'seconds_left_in_half'] = \
#         play_data.loc[play_data['Quarter'].isin([1, 2])]['seconds_left_in_game'] - (60 * 30)
#     play_data.loc[play_data['Quarter'].isin([5]), 'seconds_left_in_half'] = \
#         (60 * 10) + play_data.loc[play_data['Quarter'].isin([5])]['seconds_left_in_game']
#     plays['seconds_left_in_quarter'] = play_data['seconds_left_in_quarter']
#     plays['seconds_left_in_game'] = play_data['seconds_left_in_game']
#     plays['seconds_left_in_half'] = play_data['seconds_left_in_half']
    play_data['seconds_into_quarter'] = (15 * 60) - play_data['GameClock'].str.split(':', expand=True)[0].astype('int') * 60 + \
        play_data['GameClock'].str.split(':', expand=True)[1].astype('int') 
    play_data['seconds_into_game'] = play_data['seconds_into_quarter'] + (60 * 15 * play_data['Quarter'] - 1)
    plays['seconds_into_game'] = play_data['seconds_into_game']
    plays['seconds_into_quarter'] = play_data['seconds_into_quarter']
    # ##############################
    # # Score of Teams before play
    # ##############################
    # play_data.loc[play_data['PossessionTeam'] == play_data['HomeTeamAbbr'], 'Pos_Team_Score'] = \
    #     play_data.loc[play_data['PossessionTeam'] == play_data['HomeTeamAbbr']]['HomeScoreBeforePlay']
    # play_data.loc[play_data['PossessionTeam'] == play_data['VisitorTeamAbbr'], 'Pos_Team_Score'] = \
    #     play_data.loc[play_data['PossessionTeam'] == play_data['VisitorTeamAbbr']]['VisitorScoreBeforePlay']

    # play_data.loc[play_data['PossessionTeam'] == play_data['HomeTeamAbbr'], 'Def_Team_Score'] = \
    #     play_data.loc[play_data['PossessionTeam'] == play_data['HomeTeamAbbr']]['VisitorScoreBeforePlay']
    # play_data.loc[play_data['PossessionTeam'] == play_data['VisitorTeamAbbr'], 'Def_Team_Score'] = \
    #     play_data.loc[play_data['PossessionTeam'] == play_data['VisitorTeamAbbr']]['VisitorScoreBeforePlay']
    # play_data['Point_Diff_Poss_to_Def'] = play_data['Pos_Team_Score'] - play_data['Def_Team_Score']
    # plays['Point_Diff_Poss_to_Def'] = play_data['Point_Diff_Poss_to_Def']
    # plays['Pos_Team_Score'] = play_data['Pos_Team_Score']
    # plays['Def_Team_Score'] = play_data['Def_Team_Score']
    ###############################
    # Yards from Own Endzone
    ###############################
    play_data['YardsFromOwnEndzone'] = 0
    play_data.loc[play_data['PossessionTeam'] == play_data['FieldPosition'], 'YardsFromOwnEndzone'] = \
        play_data.loc[play_data['PossessionTeam'] == play_data['FieldPosition']]['YardLine']
    play_data.loc[play_data['PossessionTeam'] != play_data['FieldPosition'], 'YardsFromOwnEndzone'] = \
        (50 - (play_data.loc[play_data['PossessionTeam'] != play_data['FieldPosition']]['YardLine']) + 50)
    plays['YardsFromOwnEndzone'] = play_data['YardsFromOwnEndzone']
    plays['YardsFromOppEndzone'] = (100 - play_data['YardsFromOwnEndzone'])
    # ###############################
    # # Distance of Players to rusher
    # ###############################
    # Defense
    defense_gp = df.query('not IsOnOffense').groupby(['PlayId', 'Team'])
    temp = defense_gp.agg({'distance_to_runner': 'mean'})
    temp.index = temp.index.get_level_values(0)
    plays['AvgDistDefensePlayerFromRusher'] = temp['distance_to_runner'].values.astype('float32')
    temp = defense_gp.agg({'TimeToTackle': 'max'})
    temp.index = temp.index.get_level_values(0)
    plays['MaxTimeToTackle'] = temp['TimeToTackle'].values.astype('float32')
    temp = defense_gp.agg({'TimeToTackle': 'mean'})
    temp.index = temp.index.get_level_values(0)
    plays['AvgTimeToTackle'] = temp['TimeToTackle'].values.astype('float32')
    # #############################################################################
    # # Group by players within 5 yards Y distance from runner and outside 5 yards
    # #############################################################################
    df['X_dist_to_rusher'] = np.abs(df['X_std'] - df['X_std_rusher'])
    df['Y_dist_to_rusher'] = np.abs(df['Y_std'] - df['Y_std_rusher'])
    temp = df.query('not IsOnOffense and Y_dist_to_rusher <= 5').groupby(['PlayId']).agg({
                                                                'X_dist_to_rusher': ['max', 'min'], #['min','max','mean'],
                                                                'Y_dist_to_rusher': ['min', 'mean'], #['min','max','mean'],
                                                                # 'distance_to_runner': ['min','max','mean'],
                                                                'S':['max','mean'], #['min','max','mean'],
                                                                'A':['max','mean'], #'min',
                                                                'Dir_std' : ['mean'], #['min','max','mean'],
                                                                })

    temp.columns = ['_'.join(col).strip()+'_within_5y_rush_lateral' for col in temp.columns.values]
    temp = temp.reset_index()
    plays = plays.merge(temp, on='PlayId', how='left')
    temp = df.query('not IsOnOffense and Y_dist_to_rusher > 5').groupby(['PlayId']).agg({
                                                                'X_dist_to_rusher':['mean'], #['min','max','mean'],
                                                                # 'Y_dist_to_rusher':['min','max','mean'],
                                                                # 'distance_to_runner': ['min','max','mean'],
                                                                # 'S':['min','max','mean'],
                                                                'A':['mean'], #['min','max','mean'],
                                                                # 'Dir_std' : ['min','max','mean'],
                                                                })

    temp.columns = ['_'.join(col).strip()+'_outside_5y_rush_lateral' for col in temp.columns.values]
    temp = temp.reset_index()
    plays = plays.merge(temp, on='PlayId', how='left')
    # #################################
    # # Avg Position Group from Rusher
    # #################################
    pos_group_list = [
                ['SS','S','FS','CB','DB'],
               # ['DE', 'DT','NT'],
               ['ILB','LB','OLB','MLB'] ,
               ['WR'],
               ['TE'],
               ['T','G','C','OT','OG'],
               ['RB','FB'],
                     ]
    for pg in pos_group_list:
        temp = df.query('Position in @pg').groupby(['PlayId']).agg({
                                                                    'X_dist_to_rusher':['min','mean'], #['min','max','mean'],
                                                                    'Y_dist_to_rusher':['mean'], #['min','max','mean'],
                                                                    # 'distance_to_runner': ['min','max','mean'],
                                                                    'S':['mean'], #['min','max','mean'],
                                                                    'A':['mean'], #['min','max','mean'],
                                                                    'Dir_std' : ['mean'], #['min','max','mean'],
                                                                    })
        pg_names = '_'.join(pg)
        temp.columns = ['_'.join(col).strip()+'_'+pg_names for col in temp.columns.values]
        temp = temp.reset_index()
        plays = plays.merge(temp, on='PlayId', how='left', suffixes=('',pg_names))
    # #################################
    # # Rank Distance to Rusher Details
    # #################################
    df['RankDistRunnerTeam'] = df.groupby(['PlayId','IsOnOffense'])['distance_to_runner'] \
        .rank(method='first') \
        .astype('int')
    for rnkdist in range(1, 12):
        if rnkdist != 1:
            temp = df.query('IsOnOffense and RankDistRunnerTeam == @rnkdist and Position != "QB"')[['PlayId',
                                                                               'X_std', 'Y_std',
                                                                               'X_dist_to_rusher',
                                                                               'Y_dist_to_rusher',
                                                                               'Dir_std',
                                                                               # 'X_relative_YardLine',
                                                                               # # 'X_relative_FirstDown',
                                                                               # 'distance_to_runner',
                                                                               # 'frame1_distance_to_runner',
                                                                               # 'frame2_distance_to_runner',
                                                                               # 'frame3_distance_to_runner',
                                                                               # 'frame4_distance_to_runner',
                                                                               'frame5_distance_to_runner',
                                                                               'A',
                                                                               'S'
                                                                               ]]
            temp = temp.reset_index(drop=True)
            temp.columns = [c+'_'+str(rnkdist)+'rank_offense' if c not in 'PlayId' else 'PlayId' for c in temp.columns]
            plays = plays.merge(temp, on='PlayId', how='left')
        temp = df.query('not IsOnOffense and RankDistRunnerTeam == @rnkdist')[['PlayId',
                                                                               'X_std', 'Y_std',
                                                                               'X_dist_to_rusher',
                                                                               'Y_dist_to_rusher',
                                                                               'Dir_std',
                                                                               # 'X_relative_YardLine', #'X_relative_FirstDown',
                                                                               'distance_to_runner',
                                                                               # 'frame1_distance_to_runner',
                                                                               # 'frame2_distance_to_runner',
                                                                               # 'frame3_distance_to_runner',
                                                                               # 'frame4_distance_to_runner',
                                                                               'frame5_distance_to_runner',
                                                                               'A',
                                                                               'S'
                                                                               ]]
        temp = temp.reset_index(drop=True)
        temp.columns = [c+'_'+str(rnkdist)+'rank_defense' if c not in 'PlayId' else 'PlayId' for c in temp.columns]
        plays = plays.merge(temp, on='PlayId', how='left') #, suffixes=('','_'+str(rnkdist)+'rank_defense'))

    # # Time it takes to get to rusher
    for rnk in [1]: #range(1, 12):
        plays[f'def_rank{rnk}_time_to_runner'] = \
            plays[f'distance_to_runner_{rnk}rank_defense'] / plays[f'S_{rnk}rank_defense']
    ###############################
    # Angles by dist to rusher
    ###############################
    for ofnd in range(2, 6):
        for dfnd in range(1, 6):
            plays[f'Ang_{dfnd}Def_rush_{ofnd}Off'] = plays.apply(get_angle,
                                                             ro=ofnd,
                                                             rd=dfnd,
                                                             axis=1)
    #############################
    # X - Y positions
    #############################
    # Defense
    t = defense_gp.agg({'X_std': 'min'})
    t.index = t.index.get_level_values(0)
    plays['X_min_Defense'] = t['X_std'].values.astype('float32')
    t = defense_gp.agg({'X_std': 'max'})
    t.index = t.index.get_level_values(0)
    plays['X_max_Defense'] = t['X_std'].values.astype('float32')
    plays['X_max_min_min_Defense'] = plays['X_max_Defense'] - plays['X_min_Defense']
    # #############################
    # # Runner horizontal speed
    # #############################
    radian_angles = (90 - plays['Dir_std_rusher']) * np.pi / 180.0
    plays['v_horizontal_rusher'] = np.abs(plays['S_rusher'] * np.cos(radian_angles))
    # plays['v_vertical_rusher'] = np.abs(plays['S_rusher'] * np.sin(radian_angles))
    plays['a_horizontal_rusher'] = np.abs(plays['A_rusher'] * np.cos(radian_angles))
    plays['a_vertical_rusher'] = np.abs(plays['A_rusher'] * np.sin(radian_angles))
    radian_angles = (90 - plays['Dir_std_qb']) * np.pi / 180.0
    plays['v_horizontal_qb'] = np.abs(plays['S_qb'] * np.cos(radian_angles))
    # plays['v_vertical_qb'] = np.abs(plays['S_qb'] * np.sin(radian_angles))
    # plays['a_horizontal_qb'] = np.abs(plays['A_qb'] * np.cos(radian_angles))
    # plays['a_vertical_qb'] = np.abs(plays['A_qb'] * np.sin(radian_angles))
    #############################
    # X Spread and Y Spread
    #############################
    t = defense_gp.agg({'X_std': 'std'})
    t.index = t.index.get_level_values(0)
    plays['X_spread_defense'] = t['X_std'].values.astype('float32')
    ################################################
    # Frame Features (average, max, min distance)
    ################################################
    for f in [5]: #[1, 2, 3, 4, 5]:
        plays[f'agg_frame{f}_mean_dist_defense'] = plays[[f'frame{f}_distance_to_runner_10rank_defense',
                                                          f'frame{f}_distance_to_runner_11rank_defense',
                                                          f'frame{f}_distance_to_runner_1rank_defense',
                                                          f'frame{f}_distance_to_runner_2rank_defense',
                                                          f'frame{f}_distance_to_runner_3rank_defense',
                                                          f'frame{f}_distance_to_runner_4rank_defense',
                                                          f'frame{f}_distance_to_runner_5rank_defense',
                                                          f'frame{f}_distance_to_runner_6rank_defense',
                                                          f'frame{f}_distance_to_runner_7rank_defense',
                                                          f'frame{f}_distance_to_runner_8rank_defense',
                                                          f'frame{f}_distance_to_runner_9rank_defense']].mean(axis=1)
        # Offense
        plays[f'agg_frame{f}_mean_dist_offense'] = plays[[f'frame{f}_distance_to_runner_10rank_offense',
                                                          f'frame{f}_distance_to_runner_11rank_offense',
                                                          f'frame{f}_distance_to_runner_2rank_offense',
                                                          f'frame{f}_distance_to_runner_3rank_offense',
                                                          f'frame{f}_distance_to_runner_4rank_offense',
                                                          f'frame{f}_distance_to_runner_5rank_offense',
                                                          f'frame{f}_distance_to_runner_6rank_offense',
                                                          f'frame{f}_distance_to_runner_7rank_offense',
                                                          f'frame{f}_distance_to_runner_8rank_offense',
                                                          f'frame{f}_distance_to_runner_9rank_offense']].mean(axis=1)
    ################################################
    # Season
    ################################################
    plays['Season'] = play_data['Season']
    #### FINAL CHECK OF PLAYS LENGTH
    if len(plays) != len(df['PlayId'].unique()):
        raise Exception('Wrong play length, must have merged wrong')

    if print_mem:
        play_mem = plays.memory_usage().sum() / 1024**2
        logger.info('Play memory usage is: {:.2f} MB'.format(play_mem))
    if not test and TRIAL_RUN:
        logger.info(sorted([f for f in plays.columns]))

    KEEP_COLS = ['A_10rank_defense',
                 'A_10rank_offense',
                 'A_11rank_defense',
                 'A_11rank_offense',
                 'A_1rank_defense',
                 'A_2rank_defense',
                 'A_3rank_offense',
                 'A_4rank_offense',
                 'A_5rank_offense',
                 'A_6rank_defense',
                 'A_6rank_offense',
                 'A_7rank_defense',
                 'A_7rank_offense',
                 'A_8rank_offense',
                 'A_max_within_5y_rush_lateral',
                 'A_mean_ILB_LB_OLB_MLB',
                 'A_mean_RB_FB',
                 'A_mean_TE',
                 'A_mean_outside_5y_rush_lateral',
                 'A_mean_within_5y_rush_lateral',
                 'A_rusher',
                 'Ang_1Def_rush_3Off',
                 'Ang_1Def_rush_4Off',
                 'Ang_1Def_rush_5Off',
                 'Ang_2Def_rush_3Off',
                 'Ang_3Def_rush_4Off',
                 'Ang_4Def_rush_4Off',
                 'Ang_4Def_rush_5Off',
                 'AvgDistDefensePlayerFromRusher',
                 'AvgTimeToTackle',
                 'Dir_std_10rank_offense',
                 'Dir_std_11rank_offense',
                 'Dir_std_1rank_defense',
                 'Dir_std_2rank_defense',
                 'Dir_std_5rank_offense',
                 'Dir_std_6rank_defense',
                 'Dir_std_6rank_offense',
                 'Dir_std_7rank_defense',
                 'Dir_std_8rank_offense',
                 'Dir_std_mean_RB_FB',
                 'Dir_std_mean_SS_S_FS_CB_DB',
                 'Dir_std_mean_TE',
                 'Dir_std_mean_within_5y_rush_lateral',
                 'Dir_std_rusher',
                 'MaxTimeToTackle',
                 'S_10rank_offense',
                 'S_11rank_offense',
                 'S_3rank_offense',
                 'S_4rank_offense',
                 'S_6rank_offense',
                 'S_9rank_offense',
                 'S_max_within_5y_rush_lateral',
                 'S_mean_ILB_LB_OLB_MLB',
                 'S_mean_RB_FB',
                 'S_mean_TE',
                 'S_mean_T_G_C_OT_OG',
                 'S_mean_within_5y_rush_lateral',
                 'S_qb',
                 'S_rusher',
                 'Season',
                 'X_dist_to_rusher_10rank_defense',
                 'X_dist_to_rusher_5rank_defense',
                 'X_dist_to_rusher_8rank_offense',
                 'X_dist_to_rusher_9rank_defense',
                 'X_dist_to_rusher_max_within_5y_rush_lateral',
                 'X_dist_to_rusher_mean_SS_S_FS_CB_DB',
                 'X_dist_to_rusher_mean_outside_5y_rush_lateral',
                 'X_dist_to_rusher_min_SS_S_FS_CB_DB',
                 'X_dist_to_rusher_min_TE',
                 'X_dist_to_rusher_min_T_G_C_OT_OG',
                 'X_dist_to_rusher_min_within_5y_rush_lateral',
                 'X_max_min_min_Defense',
                 'X_relative_YardLine_rusher',
                 'X_spread_defense',
                 'Y_dist_to_rusher_1rank_defense',
                 'Y_dist_to_rusher_3rank_defense',
                 'Y_dist_to_rusher_3rank_offense',
                 'Y_dist_to_rusher_4rank_defense',
                 'Y_dist_to_rusher_5rank_offense',
                 'Y_dist_to_rusher_6rank_defense',
                 'Y_dist_to_rusher_7rank_defense',
                 'Y_dist_to_rusher_8rank_defense',
                 'Y_dist_to_rusher_8rank_offense',
                 'Y_dist_to_rusher_9rank_defense',
                 'Y_dist_to_rusher_mean_TE',
                 'Y_dist_to_rusher_mean_T_G_C_OT_OG',
                 'Y_dist_to_rusher_mean_WR',
                 'Y_dist_to_rusher_mean_within_5y_rush_lateral',
                 'Y_dist_to_rusher_min_within_5y_rush_lateral',
                 'Y_std_10rank_defense',
                 'Yards',
                 'YardsGreaterOppEZ',
                 'YardsVSFirstDown',
                 'YardsVSOppEz',
                 'a_horizontal_rusher',
                 'a_vertical_rusher',
                 'agg_frame5_mean_dist_defense',
                 'agg_frame5_mean_dist_offense',
                 'def_rank1_time_to_runner',
                 'v_horizontal_qb',
                 'v_horizontal_rusher',
                 'seconds_into_game',
                 'seconds_into_quarter'
                 ]
    BASE_COLS = ['GameId','PlayId','YardsFromOwnEndzone','Distance','YardsFromOppEndzone']
    NOT_BASE_FEAT_COLS = ['Yards','YardsGreaterOppEZ','YardsVSFirstDown','YardsVSOppEz']
    KEEP_COLS = KEEP_COLS + [f for f in BASE_COLS if f not in KEEP_COLS]
    KEEP_COLS = [f for f in KEEP_COLS if f not in NOT_BASE_FEAT_COLS]
    if test:
        return plays[KEEP_COLS]
    num_na_yards = sum(plays['Yards'].isna())
    logger.info(f'Numer of Yards values that are now NA {num_na_yards}')
    return plays[KEEP_COLS + ['Yards']]

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        try:
            col_type = df[col].dtype
        except:
            print(col)
        if col_type == bool:
            continue
        elif col_type != object:
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
                df[col] = df[col].astype(np.float16) # Make all floats float16
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def explode_to_yards(df, save_csv=False, train_set=True,
                     train_yards_min=-99, train_yards_max=99):
    """
    Expand from 1 row per play to 199 rows per play
    each row had a different 'Yards' value.
    """
    df = df.copy()
    if train_set:
        logger.info('Exploding training data to 199 rows each play')
    # join on skeleton for Yards
    if train_set:
        if TRAIN_YARDS is not None:
            skel = pd.DataFrame(index=TRAIN_YARDS)
        else:
            skel = pd.DataFrame(index=range(train_yards_min, train_yards_max + 1))
    else:
        skel = pd.DataFrame(index=range(-99, 100))
    skel = skel.reset_index()
    skel['key'] = 1
    df['key'] = 1
    if 'Yards' in df.columns:
        df = df.rename(columns={'Yards':'Yards_act'})
    skel = skel.rename(columns={'index' : 'Yards'})
    df = skel.merge(df, on='key', how='left')
#     logger.info([f for f in df.columns])
    if 'Yards_act' in df.columns:
        df['Gained'] = 0
        df.loc[df['Yards'] >= df['Yards_act'], 'Gained'] = 1
    #### Create special features here
    # df['YardsGreaterEqualToDistanceToGain'] = df['Yards'] >= df['Distance']
    df['YardsGreaterOppEZ'] = df['Yards'] >=  df['YardsFromOppEndzone']
    df['YardsVSOppEz'] = df['Yards'] - df['YardsFromOppEndzone']
    df['YardsVSFirstDown'] = df['Yards'] - df['Distance']
    df = df.drop('key', axis=1)
    if train_set:
        logger.info('Done exploding training data')
    df = df.sort_values(['PlayId','Yards'])
    NEW_FEATS = ['YardsGreaterOppEZ',
                 'YardsVSOppEz','YardsVSFirstDown'
                #'YardsGreaterEqualToDistanceToGain',
                ]
    REMOVE_FEATS = ['YardsFromOppEndzone','Distance']
    return df, NEW_FEATS, REMOVE_FEATS
######################################
# Eval Metric
######################################
def crps_eval(y_pred, dataset, is_higher_better=False):
    labels = dataset.get_label()
    y_true = np.zeros((len(labels),199))
    for i, v in enumerate(labels):
        y_true[i, v:] = 1
    y_pred = y_pred.reshape(-1, 199, order='F')
    y_pred = np.clip(y_pred.cumsum(axis=1), 0, 1)
    return 'crps', np.mean((y_pred - y_true)**2), False

def crps_eval_exploded(y_pred, dataset, is_higher_better=False):
    labels = dataset.get_label().values
    #n_plays = int(len(y_pred)/199)
    y_pred = y_pred.reshape(-1, 199, order='F')
    y_true = labels.reshape(-1, 199, order='F')
    return 'crps', np.mean((y_pred - y_true)**2), False

def crps_eval_exploded2(y_true, y_pred):
    #labels = dataset.get_label().values
    n_plays = int(len(y_pred)/199)
    y_true = y_true.reshape(n_plays, 199, order='F')
    y_pred = y_pred.reshape(n_plays, 199, order='F')
    #y_pred = np.maximum.accumulate(y_pred, axis=0)
    return 'crps', np.mean((y_pred - y_true)**2), False

def crps_from_df(df, col_true, col_pred):
    df = df.sort_values(['PlayId','Yards'])
    y_true = df[col_true].values
    y_pred = df[col_pred].values
    n_plays = int(len(y_pred)/199)
    y_true = y_true.reshape(n_plays, 199, order='F')
    y_pred = y_pred.reshape(n_plays, 199, order='F')
    #y_pred = np.maximum.accumulate(y_pred, axis=0)
    return np.mean((y_pred - y_true)**2)
##########################################
# Create input data and FEATURE list
##########################################
logger.info('Creating vectorized features')
plays = get_vectorized_features(train)
#########################
# Create est_yards stats
#########################
def clean_play(df):
    """
    Takes the new Play and Makes sure it has all the corrected features for
    creating the estimated yards features.
    """
    df.loc[df['PossessionTeam'] == 'ARZ', 'PossessionTeam'] = 'ARI'
    df.loc[df['PossessionTeam'] == 'BLT', 'PossessionTeam'] = 'BAL'
    df.loc[df['PossessionTeam'] == 'CLV', 'PossessionTeam'] = 'CLE'
    df.loc[df['PossessionTeam'] == 'HST', 'PossessionTeam'] = 'HOU'
    df.loc[df['FieldPosition'] == 'ARZ', 'FieldPosition'] = 'ARI'
    df.loc[df['FieldPosition'] == 'BLT', 'FieldPosition'] = 'BAL'
    df.loc[df['FieldPosition'] == 'CLV', 'FieldPosition'] = 'CLE'
    df.loc[df['FieldPosition'] == 'HST', 'FieldPosition'] = 'HOU'
    df['YardsFromOwnGoal'] = np.where(df.FieldPosition == df.PossessionTeam,
                                       df.YardLine, 50 + (50-df.YardLine))
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"    
    df.loc[df['PossessionTeam'] == df['HomeTeamAbbr'], 'DefendingTeam'] = \
        df.loc[df['PossessionTeam'] == df['HomeTeamAbbr']]['VisitorTeamAbbr']
    df.loc[df['PossessionTeam'] == df['VisitorTeamAbbr'], 'DefendingTeam'] = \
        df.loc[df['PossessionTeam'] == df['VisitorTeamAbbr']]['HomeTeamAbbr']
    return df
def create_estimated_yards(df):
    """
    Estimates the yards of previous plays based on relationship to
    eachother
    """
    if 'NflId' in df.columns:
        df = df.query('NflId == NflIdRusher').copy()
    df[['prev_game', 'prev_play', 'prev_team', 'prev_yfog']] = \
        df[['GameId', 'PlayId', 'Team', 'YardsFromOwnGoal']].shift(1)
    filt = (df.GameId==df.prev_game) & (df.Team==df.prev_team) & (df.PlayId-df.prev_play<30)
    df.loc[filt,'est_prev_yards'] = df[filt]['YardsFromOwnGoal'] - df[filt]['prev_yfog']
    df['est_Yards'] = df['est_prev_yards'].shift(-1)
    df.loc[df['est_Yards'] < -20, 'est_Yards'] = np.nan
    df = df.drop(['prev_game', 'prev_play', 'prev_team', 'prev_yfog','est_prev_yards'], axis=1)
    if 'Yards' in df.columns:
        return df[['GameId','PlayId','Team','NflIdRusher','est_Yards','Yards',
                   'PossessionTeam','DefendingTeam','Season']].reset_index(drop=True)
    return df[['GameId','PlayId','Team','NflIdRusher','est_Yards',
               'PossessionTeam','DefendingTeam','Season']].reset_index(drop=True)

def create_yards_est_features(playdata, playdata_all):
    if 'Yards' in playdata.columns:
        playdata_t = playdata.query('NflId == NflIdRusher')[['GameId','PlayId','Team','NflIdRusher','Yards',
                                                             'PossessionTeam','DefendingTeam','FieldPosition','YardLine',
                                                             'HomeTeamAbbr','VisitorTeamAbbr','Season']].copy()
    else:
        playdata_t = playdata.query('NflId == NflIdRusher')[['GameId','PlayId','Team','NflIdRusher',
                                                             'PossessionTeam','DefendingTeam','FieldPosition','YardLine',
                                                             'HomeTeamAbbr','VisitorTeamAbbr','Season']].copy()
    playdata_c = clean_play(playdata_t)
    # At the current play, get some statistics
    CurrentPlayId = playdata_c['PlayId'].values[0]
    CurrentGameId = playdata_c['GameId'].values[0]
    CurrentPosTeam = playdata_c['PossessionTeam'].values[0]
    CurrentDefTeam = playdata_c['DefendingTeam'].values[0]
    CurrentNflIdRusher = playdata_c['NflIdRusher'].values[0]
    CurrentSeason = playdata_c['Season'].values[0]
    # Copy current play
    CurrentPlay = playdata_c.copy()
    # Within Game Est_yards stats
    playdata_all = pd.concat([playdata_c, playdata_all], axis=0, sort=False)
    est_yards = create_estimated_yards(playdata_all.sort_values('PlayId'))
    CurrentPlay['EstYardsAvgPossesionTeamGame'] = \
        est_yards.query('GameId == @CurrentGameId and PossessionTeam == @CurrentPosTeam')['est_Yards'].mean()
    CurrentPlay['EstYardsAvgRusherGame'] = \
        est_yards.query('GameId == @CurrentGameId and NflIdRusher == @CurrentNflIdRusher')['est_Yards'].mean()
    # Season long stats
    CurrentPlay['EstYardsAvgPossesionTeamSeason'] = \
        est_yards.query('Season == @CurrentSeason and PossessionTeam == @CurrentPosTeam')['est_Yards'].mean()
    CurrentPlay['EstYardsAvgDefendingTeamTeamSeason'] = \
        est_yards.query('Season == @CurrentSeason and DefendingTeam == @CurrentDefTeam')['est_Yards'].mean()
    CurrentPlay['EstYardsAvgPossesionRusherGame'] = \
        est_yards.query('Season == @CurrentSeason and NflIdRusher == @CurrentNflIdRusher')['est_Yards'].mean()
    # Concat results
    return CurrentPlay, playdata_all, est_yards

# Setup Dataframes to hold results
playdata_all = pd.DataFrame()
PastEstYardsStats = pd.DataFrame()
# Loop through training plays
logger.info(f'Looping through training plays and adding features based on est yards')
for play, playdata in tqdm(train.groupby('PlayId'), mininterval=TQDM_MININTERVAL):
    CurrentPlay, playdata_all, _ = create_yards_est_features(playdata, playdata_all)
    PastEstYardsStats = pd.concat([PastEstYardsStats, CurrentPlay], sort=False)
logger.info(f'Done looping')

logger.info(f'Plays shape before adding stats {plays.shape}')
plays = plays.merge(PastEstYardsStats[['PlayId','EstYardsAvgPossesionTeamGame',
                                       'EstYardsAvgRusherGame',
                                       'EstYardsAvgPossesionTeamSeason',
                                       'EstYardsAvgDefendingTeamTeamSeason',
                                       'EstYardsAvgPossesionRusherGame']], on=['PlayId'],
                    how='left')
n_w_avg_rush = len(plays.loc[~plays['EstYardsAvgRusherGame'].isna()])
n_wo_avg_rush = len(plays.loc[plays['EstYardsAvgRusherGame'].isna()])
logger.info(f'Numer of plays with EstYardsAvgRusherGame NA: {n_w_avg_rush} not NA {n_wo_avg_rush}')
logger.info(f'Plays shape after adding stats {plays.shape}')
#########################
# Determine Features
#########################
NON_FEAT_COLS = ['Gained','PlayId','GameId','key']
CAT_FEATURES = ['OffenseFormation', 'OffensePersonnel', 'DefensePersonnel', 'GameWeather_process']
FEATURES = [f for f in plays.columns if f not in NON_FEAT_COLS]
FEATURES = [f for f in FEATURES if f not in CAT_FEATURES] # Drop Cat Features
FEATURES_TO_EXPLODE = [f for f in FEATURES if f in plays.columns]
COLS_NEEDED_TO_EXPLODE = ['PlayId','GameId','Distance','YardsFromOppEndzone']
COLS_NEEDED_TO_EXPLODE = [f for f in COLS_NEEDED_TO_EXPLODE if f not in FEATURES_TO_EXPLODE]
plays = reduce_mem_usage(plays)
if TRAIN_YARDS is None:
    plays, NEW_FEATS, REMOVE_FEATS = explode_to_yards(plays[FEATURES_TO_EXPLODE + COLS_NEEDED_TO_EXPLODE])
    FEATURES = FEATURES + NEW_FEATS
    FEATURES = [f for f in FEATURES if f not in REMOVE_FEATS]
    logger.info('Features being used:')
    logger.info([f for f in FEATURES])
    y_train = plays['Gained'].copy()
######################################
# Setup Tracking DataFrames and Lists
######################################
feature_importance = pd.DataFrame()
scores = []
models = []
######################################
# Model Parameters
######################################
params = {'num_leaves': 31,
          'max_depth' : -1,
          # 'feature_fraction': 0.88,
          # 'min_data_in_leaf': 66,
          #"bagging_seed": 11,
          'subsample_for_bin' : 200000,
          'objective': 'binary',
          'metric':'None',
          'learning_rate': LEARNING_RATE,
          "boosting_type": "gbdt",
          'num_threads': N_THREADS,
          'random_state': RANDOM_STATE,
          'min_split_gain': 0.0,
          'min_child_weight':0.001,
          'min_child_samples':20,
          'subsample' : 1.0,
          'subsample_freq' :0,
          'colsample_bytree':1.0,
          'reg_alpha':0.0,
          'reg_lambda':0.0,
          'two_round':True
          }
################################
# Training Helper Functions
################################
def post_process_by_yards(df, pred_col):
    """
    Force 0 and 1 for impossible yards gained.
    """
    df['pred'] = df[pred_col]
    df.loc[df['Yards'] < (-1 * df['YardsFromOwnEndzone']), 'pred'] = 0
    df.loc[df['Yards'] > (100 - df['YardsFromOwnEndzone']), 'pred'] = 1
    return df
def create_oof(oof, val_df, y_pred, n_fold, save=True):
    logger.info('Creating oofs')
    oof_fold = val_df[['GameId','PlayId','YardsFromOwnEndzone','Yards','Gained']].copy()
    oof_fold['fold'] = n_fold + 1
    oof_fold['pred_raw'] = y_pred
    oof_fold['pred_monotonic'] = oof_fold.groupby('PlayId')['pred_raw'].apply(np.maximum.accumulate)
    oof_fold = post_process_by_yards(oof_fold, pred_col='pred_monotonic')
    oof = pd.concat([oof_fold, oof], axis=0, sort=False)
    if save and not TRIAL_RUN:
        oof.to_csv(f'{OOF_DIR}/oof_{MODEL_NUMBER}_{RUN_ID}.csv', index=False)
    return oof
def create_fi(model, X_train, n_fold, feature_importance, save=True):
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = X_train.columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = n_fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0, sort=False)
    if save and not TRIAL_RUN:
        feature_importance.to_csv(f'{FI_DIR}/fi_{MODEL_NUMBER}_{RUN_ID}.csv', index=False)
    return feature_importance

def log_oof_scores(oof, n_fold):
    f = n_fold + 1
    oof_score_raw = crps_from_df(oof.query('fold == @f'), col_true='Gained', col_pred='pred_raw')
    oof_score_mono = crps_from_df(oof.query('fold == @f'), col_true='Gained', col_pred='pred_monotonic')
    oof_score_final = crps_from_df(oof.query('fold == @f'), col_true='Gained', col_pred='pred')
    logger.info(f'Fold {n_fold+1}: OOF - CRPS Raw: {oof_score_raw:0.8f} CRPS Monotonic: {oof_score_mono:0.8f} CRPS Post Processed: {oof_score_final:0.8f}')
    return oof_score_final

if not RUNNING_ON_KAGGLE:
    feature_importance = pd.DataFrame()
    scores = []
    best_iterations = []
    models = []
    oof = pd.DataFrame()
    folds = GroupShuffleSplit(N_FOLDS, random_state=RANDOM_STATE)
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(plays[FEATURES], groups=plays['GameId'])):
        logger.info(f'Running fold {n_fold}')
        logger.info('Creating datasets')
        tr_df, val_df = plays.iloc[train_idx], plays.iloc[val_idx]
        if TRAIN_YARDS is not None:
            tr_df, NEW_FEATS, REMOVE_FEATS = explode_to_yards(tr_df[FEATURES_TO_EXPLODE + COLS_NEEDED_TO_EXPLODE])
            val_df, _, _ = explode_to_yards(val_df[FEATURES_TO_EXPLODE + COLS_NEEDED_TO_EXPLODE], train_set=False)
            FEATURES = FEATURES + NEW_FEATS
            FEATURES = [f for f in FEATURES if f not in REMOVE_FEATS]
            logger.info('Features being used:')
            logger.info([f for f in FEATURES])
        # tr_df = tr_df.query('Yards in @FILTER_TRAIN_YARDS')
        X_train, y_train = tr_df[FEATURES], tr_df['Gained']
        X_valid, y_valid = val_df[FEATURES], val_df['Gained']
        if n_fold == 0:
            logger.info(f'Number of features: {len(FEATURES)}')
            update_tracking("n_features", len(FEATURES))
        logger.info('Fitting Model')
        model = lgb.LGBMClassifier(n_estimators=10000,
                                   n_jobs=N_THREADS,
                                   learning_rate=LEARNING_RATE,
                                   importance_type='gain')
        model.set_params(**{'metric': 'None'})
        if TRAIN_YARDS is None:
            eval_set = [(X_train, y_train), (X_valid, y_valid)]
        else:
            eval_set = [(X_valid, y_valid)]
        model.fit(X_train, y_train,
                  eval_set = eval_set,
                  eval_metric=crps_eval_exploded2,
                  verbose = 50,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        y_pred = model.predict_proba(X_valid)[:,1]
        oof = create_oof(oof, val_df, y_pred, n_fold, save=True)
        fold_score = log_oof_scores(oof, n_fold)
        feature_importance = create_fi(model, X_train, n_fold, feature_importance, save=True)
        best_iterations.append(model.best_iteration_)
        scores.append(fold_score)
        update_tracking(f"f{n_fold}_crps", fold_score)
        update_tracking(f"f{n_fold}_best_iteration", model.best_iteration_)
        update_tracking(f"mean_crps", np.mean(scores))
        logger.info('Mean so far is  : {:0.8f}'.format(np.mean(scores)))
        if not TRIAL_RUN:
            model_fn = f'{MODEL_DIR}/{MODEL_NUMBER}_fold{n_fold+1}_{RUN_ID}_{model.best_iteration_}_{fold_score}.txt'
            model.booster_.save_model(model_fn)
    logger.info('=====Done training======')
    logger.info('Mean so far is  : {:0.8f}'.format(np.mean(scores)))
    _, oof_score, _ = crps_eval_exploded2(oof['pred_raw'].values, oof['Gained'].values)
    oof_score_raw = crps_from_df(oof, col_true='Gained', col_pred='pred_raw')
    oof_score_mono = crps_from_df(oof, col_true='Gained', col_pred='pred_monotonic')
    oof_score_final = crps_from_df(oof, col_true='Gained', col_pred='pred')
    logger.info(f'OOF - CRPS Raw: {oof_score_raw:0.8f} CRPS Monotonic: {oof_score_mono:0.8f} CRPS Post Processed: {oof_score_final:0.8f}')

if RUNNING_ON_KAGGLE:
    logger.info(f'Training on kaggle. Training with {KAGGLE_N_ESTIMATORS}')
    if TRAIN_YARDS is not None:
        plays, NEW_FEATS, REMOVE_FEATS = explode_to_yards(plays[FEATURES_TO_EXPLODE + COLS_NEEDED_TO_EXPLODE])
        FEATURES = FEATURES + NEW_FEATS
        FEATURES = [f for f in FEATURES if f not in REMOVE_FEATS]
        logger.info('Features being used:')
        logger.info([f for f in FEATURES])
    model = lgb.LGBMClassifier(n_estimators=KAGGLE_N_ESTIMATORS,
                                   learning_rate=LEARNING_RATE)
    X_train, y_train = plays[FEATURES], plays['Gained']
    model.fit(X_train, y_train,
              eval_set = [(X_train, y_train)],
              verbose = 50)
    feature_importance = pd.DataFrame()
    feature_importance["feature"] = X_train.columns
    feature_importance["importance"] = model.feature_importances_
    logger.info('====Done training!=====')
###############################################
# MAKE PREDICTIONS ON TEST SET - on Kaggle
###############################################
X_test_all = pd.DataFrame()
if RUNNING_ON_KAGGLE:
    # oof = pd.DataFrame()
    logger.info('Running predictions on kaggle')
    count = 0
    for (test_df, sample_prediction_df) in tqdm(env.iter_test(), total=3438, mininterval=TQDM_MININTERVAL):
        plays = get_vectorized_features(test_df, test=True)
        # Add current estimage yard features
        CurrentPlay, playdata_all, est_yards = create_yards_est_features(test_df.copy(), playdata_all)
        plays = plays.merge(CurrentPlay[['PlayId','EstYardsAvgPossesionTeamGame',
                                               'EstYardsAvgRusherGame',
                                               'EstYardsAvgPossesionTeamSeason',
                                               'EstYardsAvgDefendingTeamTeamSeason',
                                               'EstYardsAvgPossesionRusherGame']], on=['PlayId'],
                            how='left')
        plays, NEW_FEATS, REMOVE_FEATS = explode_to_yards(plays, train_set=False)
        X_test = plays[FEATURES].copy()

        sp = sample_prediction_df.T
        sp_idx = sp.columns.values[0]
        sp[sp_idx] = sp[sp_idx].astype('float')
        X_test['pred'] = np.maximum.accumulate(model.predict_proba(X_test)[:,1])
        X_test = post_process_by_yards(X_test, pred_col='pred')
        if TRIAL_RUN:
            X_test['GameId'] = plays['GameId'].values[0]
            X_test['PlayId'] = plays['PlayId'].values[0]
            X_test_all = pd.concat([X_test_all, X_test], axis=0, sort=False)
        sp[sp_idx] = X_test['pred'].values
        predictions_df = sp.T
        env.predict(predictions_df)
        count += 1
        if TRIAL_RUN and count == 100:
            break
    if not TRIAL_RUN:
        env.write_submission_file()
    logger.info('Done Running predictions on kaggle')
    # Save helpful CSVs
    playdata_all.to_csv('playdata_all.csv', index=False)
    X_test.to_csv('X_test_last.csv', index=False)
    CurrentPlay.to_csv('CurrentPlay.csv', index=False)
    est_yards.to_csv('est_yards.csv', index=False)
    if TRIAL_RUN:
        X_test_all.to_csv('X_test_all.csv', index=False)
