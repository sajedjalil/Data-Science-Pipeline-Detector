# import pandas as pd
# pd.options.display.max_rows = 200
# import numpy as np

# import datetime

# from scipy.spatial import ConvexHull, Voronoi

# import time


def preprocess_tracking_data(tracking_df,verbose=0, suppress_warnings=True, for_test=True):
    """
    input: dataframe of tracking plays with 22 records per play
    return: modeling_df, Yard_bin_target_df with single record per play
    """
    if suppress_warnings:
        pd.set_option('mode.chained_assignment', None)
    else:
        pd.set_option('mode.chained_assignment', 'warn')
    
    if verbose:
        print("cleaned up")
    
        start = time.time()  
    def adjust_2017_motion_metrics(metric):
        metric_mean_2018=tracking_df.query('Season==2018')[metric].mean()
        tracking_df[metric].loc[tracking_df.Season==2017]=tracking_df.query('Season==2017')[metric].transform(lambda x: (x - x.mean()) / x.std())+metric_mean_2018
    if not for_test:    
        adjust_2017_motion_metrics('S')
        adjust_2017_motion_metrics('A')  
     
    #fill NAs
    tracking_df['DefendersInTheBox'].fillna(7, inplace=True)#CLEAN: let's do this better if time
    tracking_df['Temperature'].fillna(62, inplace=True)#CLEAN: let's do this better if time
    
    #tracking_df.drop(columns=['PlayerCollegeName','Location','WindSpeed', 'WindDirection','JerseyNumber','DisplayName'], inplace=True)

    tracking_df.loc[tracking_df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    tracking_df.loc[tracking_df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    tracking_df.loc[tracking_df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    tracking_df.loc[tracking_df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    tracking_df.loc[tracking_df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    tracking_df.loc[tracking_df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    tracking_df.loc[tracking_df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    tracking_df.loc[tracking_df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    def side_of_ball(x):
        if x['Team']=='away':
            if x['VisitorTeamAbbr']==x['PossessionTeam']:

                return 'OFF' 
            else:
                return 'DEF'
        elif x['Team']=='home':
            if x['VisitorTeamAbbr']==x['PossessionTeam']:
                return 'DEF'
            else:
                return 'OFF'
        else:
            return 'UNK'
    tracking_df['side_of_ball']=tracking_df.apply(side_of_ball, axis=1)   
    
    #tracking_df.loc[tracking_df['PlayDirection']=='left','X']=120.0 - tracking_df['X'] #simplified from 60.0-(tracking_df['X']-60.0)
    def flip_x_same_direction(x):
        if x['PlayDirection']=='left':
            return 120.0 - x['X']
        else:
            return x['X']
    tracking_df['X_same_way']=tracking_df.apply(flip_x_same_direction, axis=1)
    
    def flip_y_same_direction(x):
        if x['PlayDirection']=='left':
            return 53.3 - x['Y']
        else:
            return x['Y']
    tracking_df['Y_same_way']=tracking_df.apply(flip_y_same_direction, axis=1)
    tracking_df['Y_dist_from_center']=tracking_df['Y_same_way']-53.3/2
    tracking_df['Y_abs_dist_from_center']=np.abs(tracking_df['Y_dist_from_center'])

    #angles are measured clockwise from vertical
    def flip_Dir_same_direction(x):
        if x['PlayDirection']=='left':
            return 360.0 - x['Dir']
        else:
            return x['Dir']
    tracking_df['Dir_same_way']=tracking_df.apply(flip_Dir_same_direction, axis=1)
    def shift_Dir_to_endzone(x):
        if x['Dir_same_way']<=270:
            return x['Dir_same_way']-90
        else:
            return -1.0*x['Dir_same_way']+180
    def fill_missing_Dir(x):
        if np.isnan(x['Dir_same_way']):
            if x['side_of_ball']=='OFF':
                return 0
            else:
                return 180
        else:
            return x['Dir_0Deg_to_endzone']
    tracking_df['Dir_0Deg_to_endzone']=tracking_df.apply(shift_Dir_to_endzone, axis=1)
    tracking_df['Dir_0Deg_to_endzone']=tracking_df.apply(fill_missing_Dir,axis=1)      
    tracking_df['Dir_abs_0Deg_to_endzone']=np.abs(tracking_df['Dir_0Deg_to_endzone'])

    def corrected_2017_orientation(x):
        if x['Season']!=2017:
            return x['Orientation']
        else:
            return np.mod(90+x['Orientation'],360)
    tracking_df['Orientation_corrected']=tracking_df.apply(corrected_2017_orientation, axis=1)

    def flip_Orientation_same_direction(x):
        if x['PlayDirection']=='left':
            return 360.0 - x['Orientation_corrected']
        else:
            return x['Orientation_corrected']
    tracking_df['Orientation_same_way']=tracking_df.apply(flip_Orientation_same_direction, axis=1) 
    def shift_Orientation_to_endzone(x):
        if x['Orientation_same_way']<=270:
            return x['Orientation_same_way']-90
        else:
            return -1.0*x['Orientation_same_way']+180
    tracking_df['Orientation_0Deg_to_endzone']=tracking_df.apply(shift_Orientation_to_endzone, axis=1)
    tracking_df['Orientation_abs_0Deg_to_endzone']=np.abs(tracking_df['Orientation_0Deg_to_endzone'])
    
    def calculate_yards_to_end_zone(x):
        if x['PossessionTeam']==x['FieldPosition']:
            return 100-x['YardLine']
        else:
            return x['YardLine']
    tracking_df['Yards_to_end_zone']=tracking_df.apply(calculate_yards_to_end_zone, axis=1)
    
    tracking_df['X_to_YardLine']=tracking_df['X_same_way']-(110-tracking_df['Yards_to_end_zone'])
    tracking_df['X_to_1stDown']=tracking_df['X_to_YardLine']-tracking_df['Distance']
    
    #Ball carrier Speed in direction of endzone
    tracking_df['speed_in_direction_of_EZ']=tracking_df['S']*np.cos(tracking_df['Dir_0Deg_to_endzone']/180*np.pi)
    tracking_df['X_to_YardLine_sec_momentum']=tracking_df['X_to_YardLine']\
                                            +tracking_df['speed_in_direction_of_EZ']
    tracking_df['Y_dist_from_center_sec_momentum']=tracking_df['Y_dist_from_center']\
                                            +tracking_df['S']*np.sin(tracking_df['Dir_0Deg_to_endzone']/180*np.pi)    
    if verbose:    
        print("going the same way")
        end = time.time()
        print(end - start)

        start = time.time()


    
    tracking_df['ball_carrier']=(tracking_df['NflId']==tracking_df['NflIdRusher'])
    tracking_df['Position'].loc[tracking_df['Position']=='SAF']='S'
    tracking_df['Position'].loc[tracking_df['Position']=='HB']='RB'
    tracking_df['NflIdRusher']=tracking_df['NflIdRusher'].astype(str)
    tracking_df['DefendingTeam']=tracking_df[['HomeTeamAbbr','VisitorTeamAbbr','PossessionTeam']]\
             .apply(lambda x: x['HomeTeamAbbr'] if x['PossessionTeam']==x['VisitorTeamAbbr'] else x['VisitorTeamAbbr'] 
                    , axis=1)
    if verbose:    
        print("Minor context and features done")
        end = time.time()
        print(end - start)
        
        
    #Player and Team Yard Summary reference tables
    global team_off_yard_reference, team_def_yard_reference, rusher_yard_reference, rusher_low_vol_reference
    if not for_test:
        start = time.time()
        team_off_yard_reference=tracking_df.groupby('PlayId').first()\
            .groupby(['PossessionTeam'])['Yards']\
            .describe(include='all')\
            .query('count>=20')\
            .sort_values(by='75%',ascending=False)
        team_off_yard_reference.columns=['PossessionTeam_Yards_'+col for col in team_off_yard_reference.columns]
        
        team_def_yard_reference=tracking_df.groupby('PlayId').first()\
            .groupby('DefendingTeam')\
            ['Yards']\
            .describe(include='all')\
            .query('count>=20')\
            .sort_values(by='75%',ascending=False)
        team_def_yard_reference.columns=['DefendingTeam_Yards_'+col for col in team_def_yard_reference.columns]

        rusher_yard_reference=tracking_df.groupby('PlayId').first()\
            .groupby(['NflIdRusher'])['Yards']\
            .describe(include='all')\
            .query('count>=20')
        rusher_yard_reference.columns=['Rusher_Yards_'+col for col in rusher_yard_reference.columns]
        rusher_yard_reference.reset_index(inplace=True)
        rusher_yard_reference.NflIdRusher=rusher_yard_reference.NflIdRusher.astype(str)
        #reference table for combining rushers with less than 20 carries
        rusher_low_vol_reference=tracking_df.groupby('PlayId').first().merge(rusher_yard_reference
                                                  , on='NflIdRusher'
                                                 , how='left')\
                .query('Rusher_Yards_count!=Rusher_Yards_count')['Yards']\
                .describe(include='all')
        rusher_low_vol_reference.index=['Rusher_Yards_'+col for col in rusher_low_vol_reference.index]
        

        
        if verbose:    
            print("Player and Team Yard Summary reference tables")
            end = time.time()
            print(end - start)
        
    start = time.time()
    #Ball carrier only metrics
    ball_carrier_df=tracking_df.query('ball_carrier==True').copy()
    #https://www.kaggle.com/miklgr500/fork-of-neural-networks-radam-repeatkfold
    def strtoseconds(txt):
        txt = txt.split(':')
        ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
        return ans
    ball_carrier_df['GameClock'] = ball_carrier_df['GameClock'].apply(strtoseconds)
    
    
    #https://www.kaggle.com/miklgr500/fork-of-neural-networks-radam-repeatkfold
    ball_carrier_df['TimeHandoff'] = ball_carrier_df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    ball_carrier_df['TimeSnap'] = ball_carrier_df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    ball_carrier_df['TimefromSnap'] = ball_carrier_df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    #https://www.kaggle.com/bestpredict/location-eda-8eb410
    ball_carrier_df['PlayerBirthDate'] = ball_carrier_df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60*60*24*365.25
    ball_carrier_df['PlayerAge'] = ball_carrier_df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    #https://www.kaggle.com/miklgr500/fork-of-neural-networks-radam-repeatkfold
    ball_carrier_df['PlayerHeight'] = ball_carrier_df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    ball_carrier_df['NflIdRusher']=ball_carrier_df['NflIdRusher'].astype(str)
    #from https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087
    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 
        'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 
        'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 
        'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 
        'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

    ball_carrier_df['Turf_type'] = ball_carrier_df['Turf'].map(Turf)
    def map_weather(txt):
        txt=str(txt).lower()
        if pd.isna(txt):
            return 'other'
        if 'rain' in txt or 'shower' in txt:
            return 'rain'
        if 'snow' in txt:
            return 'snow'
        return 'other'
    ball_carrier_df['GameWeather_simple'] = ball_carrier_df['GameWeather'].apply(map_weather)
    ball_carrier_df['score_diff']=(ball_carrier_df["HomeScoreBeforePlay"] - ball_carrier_df["VisitorScoreBeforePlay"])*(ball_carrier_df['Team']=='away')
    
    ball_carrier_df['OffenseFormation'].loc[ball_carrier_df['OffenseFormation']=='ACE']='SINGLEBACK'
    ball_carrier_df=ball_carrier_df.merge(rusher_yard_reference,
                                          on='NflIdRusher',
                                          how='left')\
                                    .fillna(rusher_low_vol_reference)
                                  
                                    
    if verbose:    
        print("ball carrier measures done")
        end = time.time()
        print(end - start)

        start = time.time()
    
    #Position counts   
    position_cnts=tracking_df.groupby('PlayId')['Position'].value_counts().unstack(level=-1).fillna(0)
    position_cnts.columns=['Position_cnt_'+pos for pos in position_cnts.columns]
    
    tracking_df['Position_group']=tracking_df['Position']
    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'DEF']=tracking_df['Position_group'].replace(['SAF','HB','RB','QB','TE','WR','CB','SS','FS','S','DB'],'DB')
    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'DEF']=tracking_df['Position_group'].replace(['DL','OG','OT','NT','C','G','T','DT','DE'],'DL')
    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'DEF']=tracking_df['Position_group'].replace(['LB','MLB','FB','OLB','ILB'],'LB')

    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'OFF']=tracking_df['Position_group'].replace(['WR','SS','FS','CB','DB','S','SAF'],'WR')
    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'OFF']=tracking_df['Position_group'].replace(['OG','OT','T','G','C','DE','DT','NT','DL'],'OL')
    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'OFF']=tracking_df['Position_group'].replace(['RB','FB','MLB','HB'],'RB')
    tracking_df['Position_group'].loc[tracking_df.side_of_ball == 'OFF']=tracking_df['Position_group'].replace(['TE','ILB','OLB','LB'],'TE')
    
    position_grp_cnts=tracking_df.groupby('PlayId')['Position_group'].value_counts().unstack(level=-1).fillna(0)
    position_grp_cnts.columns=['Position_grp_cnt_'+pos for pos in position_grp_cnts.columns]
    if 'Position_grp_cnt_OL' not in position_grp_cnts.columns:
        position_grp_cnts['Position_grp_cnt_OL']=0
    if 'Position_grp_cnt_DL' not in position_grp_cnts.columns:
        position_grp_cnts['Position_grp_cnt_DL']=0
    if 'Position_grp_cnt_TE' not in position_grp_cnts.columns:
        position_grp_cnts['Position_grp_cnt_TE']=0
    if 'Position_grp_cnt_LB' not in position_grp_cnts.columns:
        position_grp_cnts['Position_grp_cnt_LB']=0
        
    # Let's create some features to specify if the OL is covered 
    position_grp_cnts['Position_grp_cnt_Trench_diff'] = position_grp_cnts['Position_grp_cnt_OL']\
                                                        - position_grp_cnts['Position_grp_cnt_DL'] 
    position_grp_cnts['Position_grp_cnt_Trench_w_TE_diff'] = (position_grp_cnts['Position_grp_cnt_OL']\
                                                           + position_grp_cnts['Position_grp_cnt_TE'])\
                                                            - position_grp_cnts['Position_grp_cnt_DL']
    position_grp_cnts['Position_grp_cnt_Block_vs_runstuff_diff'] = position_grp_cnts['Position_grp_cnt_OL']\
                                                                   + position_grp_cnts['Position_grp_cnt_TE']\
                                                                - position_grp_cnts['Position_grp_cnt_DL']\
                                                                - position_grp_cnts['Position_grp_cnt_LB']
    # Let's create a feature to specify if the defense is preventing the run 
    # Let's just assume 7 or more DL and LB is run prevention 
    position_grp_cnts['Position_grp_cnt_RunStuff_def'] = ((position_grp_cnts['Position_grp_cnt_DL'] + position_grp_cnts['Position_grp_cnt_LB']) > 6).astype(int)

    if verbose:    
        print("position counts done")
        end = time.time()
        print(end - start)

        start = time.time()

    def distance_based_metrics(X='X_to_YardLine',Y='Y_dist_from_center',momentum=False):
        """
        suffix: recommend either '_to_YardLine' or '_sec_momentum'
        """
        if momentum:
            momentum_suffix='_sec_momentum'
        else:
            momentum_suffix=''
        X_field=X+momentum_suffix
        Y_field=Y+momentum_suffix
        
        #Calculate Distances to ball carrier
        start = time.time()
        off_dist_to_def=tracking_df[['PlayId','NflId','JerseyNumber',X_field,Y_field,'Orientation_0Deg_to_endzone','side_of_ball','ball_carrier']].query("side_of_ball=='OFF'")\
                    .merge(tracking_df[['PlayId','NflId','JerseyNumber',X_field,Y_field,'side_of_ball']].query("side_of_ball=='DEF'"),
                           on='PlayId', how='inner', suffixes=("_off","_def"))

        def calc_distance(x1,y1,x2,y2):
            loc1 = np.asarray([x1,y1])
            loc2 = np.asarray([x2,y2])
            return np.linalg.norm(loc1-loc2)
        def calc_dist_off_to_def(x):
            return calc_distance(x[X_field+'_off'],x[Y_field+'_off'],x[X_field+'_def'],x[Y_field+'_def'])

        off_dist_to_def['off_distance_to_def']=off_dist_to_def.apply(calc_dist_off_to_def,axis=1)
        off_dist_to_def['engaged_dist']=(off_dist_to_def['off_distance_to_def']<=4)

        engaged=off_dist_to_def.query('engaged_dist==True')
        def player_to_player_direction(x1,y1,x2,y2):
            try:
                return np.arctan((y2-y1)/(x2-x1))*180/np.pi
            except:
                return 0
        def blocker_to_def_dir(x):
            return player_to_player_direction(x[X_field+'_off'],x[Y_field+'_off'],x[X_field+'_def'],x[Y_field+'_def'])                 
        engaged['blocker_to_def_dir']=engaged.apply(blocker_to_def_dir, axis=1)
        engaged['engaged_orientation']=(np.abs(engaged['blocker_to_def_dir'] - engaged['Orientation_0Deg_to_endzone'])<=45)
        engaged=engaged.merge(ball_carrier_df[['PlayId',X_field,Y_field]]
                   ,on='PlayId')\
            .rename(columns={X_field:X_field+'_bc',Y_field:Y_field+'_bc'})
        def calc_dist_off_to_bc(x):
            return calc_distance(x[X_field+'_off'],x[Y_field+'_off'],x[X_field+'_bc'],x[Y_field+'_bc'])
        def calc_dist_def_to_bc(x):
            return calc_distance(x[X_field+'_def'],x[Y_field+'_def'],x[X_field+'_bc'],x[Y_field+'_bc'])
        engaged['off_distance_to_bc']=engaged.apply(calc_dist_off_to_bc,axis=1)
        engaged['def_distance_to_bc']=engaged.apply(calc_dist_def_to_bc,axis=1)
        engaged['engaged_blocker_between_bc_and_def']=(engaged['off_distance_to_bc']<engaged['def_distance_to_bc'])
        engaged['engaged']=engaged[['engaged_dist','engaged_orientation','engaged_blocker_between_bc_and_def']].all(axis='columns')
        engaged=engaged.query('engaged==True')
        engaged['Off_leverage_left']=(engaged[Y_field+'_off']>=engaged[Y_field+'_def'])
        engaged['Off_leverage_right']=(engaged[Y_field+'_off']<=engaged[Y_field+'_def'])

        off_dist_to_def=off_dist_to_def.merge(engaged[['PlayId','NflId_off','NflId_def','engaged']],
                         on=['PlayId','NflId_off','NflId_def'], how='left')\
                    .fillna(False)

        bc_dist_to_def=off_dist_to_def[['PlayId','NflId_def','off_distance_to_def','ball_carrier',X_field+'_def',Y_field+'_def']].query('ball_carrier==True').copy()\
            .rename(columns={'off_distance_to_def':'bc_distance_to_def'})
        bc_dist_to_def['bc_distance'+momentum_suffix+'_to_def_rank']=bc_dist_to_def.groupby('PlayId')['bc_distance_to_def'].rank(method='first').astype(int)
        bc_dist_to_def['def_left_to_right_rank']=bc_dist_to_def.groupby('PlayId')[Y_field+'_def'].rank(method='first',ascending=False).astype(int)
        #join defender number of engaged blockers 
        bc_dist_to_def=bc_dist_to_def.merge(engaged.groupby(['PlayId','NflId_def'])['NflId_off'].count(),
                             left_on=['PlayId','NflId_def'],right_index=True,
                             how='left')\
                    .rename(columns={'NflId_off':'engaged_by'})\
                    .fillna(0)
        bc_dist_to_def['engaged']=(bc_dist_to_def['engaged_by']>0)


        
        play_def_left_to_right_rank_df=bc_dist_to_def.pivot(index='PlayId',columns='def_left_to_right_rank',
                             values=['engaged','bc_distance_to_def',X_field+'_def',Y_field+'_def']).astype('float64')
        play_def_left_to_right_rank_df.columns = ['def_left_to_right'+momentum_suffix+'_rank'+str(col[1])+'_'+col[0] for col in play_def_left_to_right_rank_df.columns.values]

        play_bc_dist_to_def_rank_df=bc_dist_to_def.query('bc_distance'+momentum_suffix+'_to_def_rank<=11')[['PlayId','bc_distance_to_def','engaged','engaged_by', 'bc_distance'+momentum_suffix+'_to_def_rank']]\
            .pivot(index='PlayId',columns='bc_distance'+momentum_suffix+'_to_def_rank',values=['bc_distance_to_def','engaged','engaged_by'])\
            .fillna(bc_dist_to_def.bc_distance_to_def.max())
        play_bc_dist_to_def_rank_df.columns = ['bc_distance'+momentum_suffix+'_to_def_rank'+str(col[1])+'_'+col[0] for col in play_bc_dist_to_def_rank_df.columns.values]

        bc_dist_to_def_summaries=bc_dist_to_def.groupby('PlayId')['bc_distance_to_def'].agg(['mean','median','min', 'max'])
        bc_dist_to_def_summaries['range']=bc_dist_to_def_summaries['max'].subtract(bc_dist_to_def_summaries['min'])
        bc_dist_to_def_summaries.columns=['bc_dist'+momentum_suffix+'_to_def_'+col for col in bc_dist_to_def_summaries.columns]

        bc_dist_to_unblocked_def=bc_dist_to_def.query('engaged==False')[['PlayId','NflId_def','bc_distance_to_def']]
        bc_dist_to_unblocked_def['bc_distance'+momentum_suffix+'_to_unblocked_def_rank']=bc_dist_to_unblocked_def.groupby('PlayId')['bc_distance_to_def'].rank(method='first').astype(int)

        keep_N_unblocked_def=5
        base_unblocked_def_table=pd.DataFrame()
        rank_N_table=ball_carrier_df[['PlayId','Yards_to_end_zone',X_field,'Y_abs_dist_from_center']]
        for rank in range(1,keep_N_unblocked_def+1):
            rank_N_table['bc_distance'+momentum_suffix+'_to_unblocked_def_rank']=rank
            base_unblocked_def_table=pd.concat([base_unblocked_def_table,rank_N_table])


        def calc_dist_bc_to_nearest_EZ_corner(x):
            return calc_distance(x[X_field+'_bc'],x['Y_abs_dist_from_center'],x[X_field+'_def'],26.65)

        bc_dist_to_unblocked_def=base_unblocked_def_table\
                                .merge(bc_dist_to_unblocked_def,
                                        on=['PlayId','bc_distance'+momentum_suffix+'_to_unblocked_def_rank'], 
                                            how='left'
                                           )\
                                .merge(tracking_df\
                                           [['PlayId','NflId',X_field,Y_field]],
                                          left_on=['PlayId','NflId_def'], right_on=['PlayId','NflId'],
                                          how='left', suffixes=('_bc','_def'))\
                                .rename(columns={'bc_distance_to_def':'distance', Y_field:Y_field+'_def'})\
                                .reset_index(drop=True)
        bc_dist_to_unblocked_def.fillna({'NflId_def':9999999,
                                         X_field+'_def':bc_dist_to_unblocked_def\
                                                             ['Yards_to_end_zone'].reset_index(drop=True)+10,
                                         Y_field+'_def':26.65
                                        },
                                       inplace=True)
        bc_dist_to_unblocked_def.fillna({'distance':bc_dist_to_unblocked_def\
                                                 .apply(calc_dist_bc_to_nearest_EZ_corner,axis=1)\
                                                 .reset_index(drop=True)},
                                       inplace=True)

        bc_dist_to_unblocked_def_summaries=bc_dist_to_unblocked_def.groupby('PlayId')['distance'].agg(['mean','median','min', 'max'])
        bc_dist_to_unblocked_def_summaries['range']=bc_dist_to_unblocked_def_summaries['max'].subtract(bc_dist_to_unblocked_def_summaries['min'])
        bc_dist_to_unblocked_def_summaries.columns=['bc_dist'+momentum_suffix+'_to_unblocked_def_'+col for col in bc_dist_to_unblocked_def_summaries.columns]

        play_bc_dist_to_unblocked_def_rank_df=bc_dist_to_unblocked_def[['PlayId','distance', 'bc_distance'+momentum_suffix+'_to_unblocked_def_rank',X_field+'_def',Y_field+'_def']]\
            .pivot(index='PlayId',columns='bc_distance'+momentum_suffix+'_to_unblocked_def_rank',values=['distance',X_field+'_def',Y_field+'_def'])
        play_bc_dist_to_unblocked_def_rank_df.columns = ['bc_distance'+momentum_suffix+'_to_unblocked_def_rank'+str(col[1])+'_'+col[0] for col in play_bc_dist_to_unblocked_def_rank_df.columns.values]
        if verbose:    
            print("Distances to ball carrier"+momentum_suffix+"done")
            end = time.time()
            print(end - start)

            start = time.time()    
        #Defensive Hull Measures
        def calc_def_hull_measures(runner_v_def_points):
            hull_values=[]
            for layer_of_def in range(0,4):
                try:
                    #define hull
                    layer_hull = ConvexHull(runner_v_def_points[[X_field,Y_field]])
                    #calculate hull measures
                    hull_area=layer_hull.area
                    hull_expected_gain=runner_v_def_points[X_field].max()
                    hull_width=runner_v_def_points[Y_field].max()-runner_v_def_points[Y_field].min()
                    hull_defenders=len(layer_hull.vertices)-1

                    #related voronoi max_X_to_yardline
                    runner_vertice=runner_v_def_points.query("side_of_ball=='OFF'")[[X_field,Y_field]]
                    runner_vertice_backstop=runner_vertice.copy()
                    runner_vertice_backstop2=runner_vertice_backstop.copy()
                    runner_vertice_backstop[X_field]=runner_vertice_backstop[X_field]-1.0
                    runner_vertice_backstop[Y_field]=runner_vertice_backstop[Y_field]-1.0
                    runner_vertice_backstop2[X_field]=runner_vertice_backstop2[X_field]-1.0
                    runner_vertice_backstop2[Y_field]=runner_vertice_backstop2[Y_field]+1.0

                    #1st Down Sticks     
                    FD_stick_left=runner_v_def_points.query("side_of_ball=='OFF'")[[X_field,Y_field]]
                    FD_stick_right=FD_stick_left.copy()
                    FD_stick_left[X_field]=runner_v_def_points.query("side_of_ball=='OFF'")['Distance']
                    FD_stick_left[Y_field]=26.65
                    FD_stick_right[X_field]=runner_v_def_points.query("side_of_ball=='OFF'")['Distance']
                    FD_stick_right[Y_field]=-26.65

                    #Goal Post
                    Goal_Post=runner_v_def_points.query("side_of_ball=='OFF'")[[X_field,Y_field]]
                    Goal_Post[X_field]=runner_v_def_points.query("side_of_ball=='OFF'")['Yards_to_end_zone']+10
                    Goal_Post[Y_field]=0
                    
                    voronoi_points=runner_v_def_points[[X_field,Y_field]].iloc[layer_hull.vertices]\
                        .append(runner_vertice_backstop).append(runner_vertice_backstop2)\
                        .append(Goal_Post)\
                        .append(FD_stick_left).append(FD_stick_right)
                    if ~(runner_v_def_points[['side_of_ball']].iloc[layer_hull.vertices]=='OFF').any()[0]: #check to see if ball carrier is part of hull edge; may not be if defender has passed BC
                        voronoi_points=voronoi_points.append(runner_vertice)
                    vor = Voronoi(voronoi_points)        
                    #Find index of RB point by matching RB vertice to vor points (RB_point_ix)
                    RB_point_ix = np.where((vor.points==np.array(runner_vertice)).all(axis=1))[0][0]
                    #Find index of RB point index (RB_point_region_ix=vor.point_region[RB_point_ix])
                    RB_point_region_ix=vor.point_region[RB_point_ix]
                    #Find indexes of RB point region (RB_point_region_vertice_ixs= vor.regions[RB_point_region_ix])
                    RB_point_region_vertice_ixs= vor.regions[RB_point_region_ix]
                    #find vertices of RB point voronoi region (RB_point_region_vertices=vor.vertices[RB_point_region_vertice_ixs])
                    RB_point_region_vertices=vor.vertices[RB_point_region_vertice_ixs]
                    layer_voronoi_hull=ConvexHull(RB_point_region_vertices)
                    voronoi_area=layer_voronoi_hull.area

                    voronoi_expected_gain=RB_point_region_vertices[:,0].max()
                    voronoi_width=RB_point_region_vertices[:,1].max()-RB_point_region_vertices[:,1].min()

                    hull_values.append(hull_area)#area
                    hull_values.append(hull_expected_gain)#depth (max_X-min_X)
                    hull_values.append(hull_width)#width (Y_max-Ymin)
                    hull_values.append(hull_defenders)
                    hull_values.append(voronoi_area)#area
                    hull_values.append(voronoi_expected_gain)#depth (max_X-min_X)
                    hull_values.append(voronoi_width)#width (Y_max-Ymin)

                    #define rusher defender points for next layer
                    runner_v_def_points = runner_v_def_points.drop(index=runner_v_def_points.iloc[layer_hull.vertices].query("side_of_ball=='DEF'").index) 
                except:
                    #print(runner_v_def_points['PlayId'])
                    #break
                    #calculate hull measures - fill with previous layer metric? Or would that 
                    hull_values.append(hull_area)#area
                    hull_values.append(hull_expected_gain)#depth (max_X-min_X)
                    hull_values.append(hull_width)#width (Y_max-Ymin)
                    hull_values.append(0)
                    #related voronoi max_X_to_yardline
                    hull_values.append(voronoi_area)#area
                    hull_values.append(voronoi_expected_gain)#depth (max_X-min_X)
                    hull_values.append(voronoi_width)#width (Y_max-Ymin)

            hull_columns=[
                'hull_secondary'+momentum_suffix+'_area', 'hull_secondary'+momentum_suffix+'_expected_gain',
                'hull_secondary'+momentum_suffix+'_width','hull_secondary'+momentum_suffix+'_defenders',
                'voronoi_secondary'+momentum_suffix+'_area', 
                'voronoi_secondary'+momentum_suffix+'_expected_gain','voronoi_secondary'+momentum_suffix+'_width',
                'hull_contain'+momentum_suffix+'_area', 'hull_contain'+momentum_suffix+'_expected_gain',
                'hull_contain'+momentum_suffix+'_width','hull_contain'+momentum_suffix+'_defenders',
                'voronoi_contain'+momentum_suffix+'_area', 
                'voronoi_contain'+momentum_suffix+'_expected_gain','voronoi_contain'+momentum_suffix+'_width',
                'hull_2nd_attack'+momentum_suffix+'_area', 'hull_2nd_attack'+momentum_suffix+'_expected_gain',
                'hull_2nd_attack'+momentum_suffix+'_width','hull_2nd_attack'+momentum_suffix+'_defenders',
                'voronoi_2nd_attack'+momentum_suffix+'_area', 
                'voronoi_2nd_attack'+momentum_suffix+'_expected_gain','voronoi_2nd_attack'+momentum_suffix+'_width',
                'hull_1st_attack'+momentum_suffix+'_area', 'hull_1st_attack'+momentum_suffix+'_expected_gain',
                'hull_1st_attack'+momentum_suffix+'_width','hull_1st_attack'+momentum_suffix+'_defenders',
                'voronoi_1st_attack'+momentum_suffix+'_area', 
                'voronoi_1st_attack'+momentum_suffix+'_expected_gain','voronoi_1st_attack'+momentum_suffix+'_width'
                ]
            return pd.DataFrame([hull_values],columns=hull_columns)
        defender_hulls_df=tracking_df.query("(side_of_ball=='DEF' or ball_carrier==True)")[['PlayId',X_field,Y_field,'side_of_ball','Distance','Yards_to_end_zone']].groupby('PlayId').apply(calc_def_hull_measures).droplevel(level=1)
        if verbose:
            print("defensive hulls"+momentum_suffix+" done")
            end = time.time()
            print(end - start)

            start = time.time()
        #Generate Gap fields
        blockers_df=tracking_df.query("side_of_ball=='OFF' and Position not in ('WR','RB','QB','FB')")[['PlayId','Position',Y_field,X_field,'NflId']].sort_values(by=['PlayId',Y_field], ascending=False)
        blockers_df['blocker_left_to_right_rank']=blockers_df.groupby(['PlayId'])[Y_field].rank(method='first',ascending=False).astype(int)
        blockers_df['pair_rank']=blockers_df['blocker_left_to_right_rank']+1

        #CLEAN: How to deal with assign?
        left_sideline=blockers_df.query("blocker_left_to_right_rank==1").assign(Position='LS',Y_field=53.3, X_field=0, NflId=0000000,
               blocker_left_to_right_rank=0, pair_rank=1)
        gap_left_edges_df=pd.concat([blockers_df,left_sideline], sort=True).sort_values(by=['PlayId','blocker_left_to_right_rank'])

        gaps_df=gap_left_edges_df.merge(gap_left_edges_df, how='left'
                          , left_on=['PlayId','pair_rank'],right_on=['PlayId','blocker_left_to_right_rank'], suffixes=['_left','_right'])\
            .fillna(value={'Position_right':'RS',Y_field+'_right':0,X_field+'_right':0,'NflId_right':1111111})
        gaps_df=gaps_df.merge(tracking_df.query('ball_carrier==True')[['PlayId',Y_field,X_field,'Orientation_0Deg_to_endzone','Dir_0Deg_to_endzone']], on='PlayId')\
            .rename(columns={Y_field:Y_field+'_ball_carrier',X_field:X_field+'_ball_carrier','Orientation_0Deg_to_endzone':'Orientation_ball_carrier','Dir_0Deg_to_endzone':'Dir_ball_carrier'})

        #gap width
        gaps_df['gap_width']=np.linalg.norm(gaps_df[[X_field+'_left', Y_field+'_left']].values - gaps_df[[X_field+'_right', Y_field+'_right']].values, axis=1)
        #gap center (adjust for blocker momentum?)
        gaps_df['gap_center_X']=(gaps_df[X_field+'_left']+gaps_df[X_field+'_right'])/2.0
        gaps_df['gap_center_Y']=(gaps_df[Y_field+'_left']+gaps_df[Y_field+'_right'])/2.0
        #distance of ball carrier to gap center
        gaps_df['gap_distance_from_ball_carrier']=np.linalg.norm(gaps_df[['gap_center_X', 'gap_center_Y']].values - gaps_df[[X_field+'_ball_carrier', Y_field+'_ball_carrier']].values, axis=1)

        #determine gap leverage
        gaps_df=gaps_df.merge(engaged[['PlayId','NflId_off','Off_leverage_right']].query('Off_leverage_right==False').drop_duplicates()
                             , left_on=['PlayId','NflId_left']
                             , right_on=['PlayId','NflId_off']
                             , how='left')\
                            .drop(columns='NflId_off')\
                            .fillna(True)\
                            .merge(engaged[['PlayId','NflId_off','Off_leverage_left']].query('Off_leverage_left==False').drop_duplicates()
                             , left_on=['PlayId','NflId_right']
                             , right_on=['PlayId','NflId_off']
                             , how='left')\
                            .drop(columns='NflId_off')\
                            .fillna(True)
        gaps_df['Off_leverage_right'].loc[gaps_df['Position_left']=='LS']=False
        gaps_df['Off_leverage_left'].loc[gaps_df['Position_right']=='RS']=False
        gaps_df['Off_full_leverage']=gaps_df[['Off_leverage_right','Off_leverage_left']].all(axis=1)
        gaps_df['Off_half_leverage']=(gaps_df[['Off_leverage_right','Off_leverage_left']].any(axis=1) & ~gaps_df['Off_full_leverage'])
        gaps_df['Off_no_leverage']=~gaps_df[['Off_leverage_right','Off_leverage_left']].any(axis=1)

        #ball carrier dir cosine/angular similarity to gap center
        #ball carrier orientation cosine/angular similarity to gap center
        def ball_carrier_to_gap_direction(x):
            if x['gap_center_Y']>x[Y_field+'_ball_carrier']:
                to_center_direction = np.arctan((x['gap_center_X']-x[X_field+'_ball_carrier'])/(x['gap_center_Y']-x[Y_field+'_ball_carrier']))/np.pi*180.0
            elif x['gap_center_Y']<x[Y_field+'_ball_carrier']:
                to_center_direction =180.0 + np.arctan((x['gap_center_X']-x[X_field+'_ball_carrier'])/(x['gap_center_Y']-x[Y_field+'_ball_carrier']))/np.pi*180.0
            else:
                to_center_direction = 90.0
            return to_center_direction#1 - np.abs(x[angle_field]-to_center_angle)/180.0
        gaps_df['ball_carrier_to_gap_direction']=gaps_df.apply(ball_carrier_to_gap_direction, axis=1)
        gaps_df['ball_carrier_Dir_to_gap_ang_sim']=gaps_df.apply(lambda x: 1 - np.abs(x['Dir_ball_carrier']-x['ball_carrier_to_gap_direction'])/180.0, axis=1)
        gaps_df['ball_carrier_Orientation_to_gap_ang_sim']=gaps_df.apply(lambda x: 1 - np.abs(x['Orientation_ball_carrier']-x['ball_carrier_to_gap_direction'])/180.0, axis=1)
        ###gaps_df['ball_carrier_Dir_to_gap_cos_sim']=gaps_df.apply(lambda x: (np.cos((x['Dir_ball_carrier']-x['ball_carrier_to_gap_direction'])/180.0*np.pi)+1)/2, axis=1)
        ###gaps_df['ball_carrier_Orientation_to_gap_cos_sim']=gaps_df.apply(lambda x: (np.cos((x['Orientation_ball_carrier']-x['ball_carrier_to_gap_direction'])/180.0*np.pi)+1)/2, axis=1)

        gaps_df['gap_distance'+momentum_suffix+'_from_ball_carrier_rank']=gaps_df.groupby(['PlayId'])['gap_distance_from_ball_carrier'].rank(method='first',ascending=True).astype(int)
        gaps_df['ball_carrier_Dir'+momentum_suffix+'_to_gap_ang_sim_rank']=gaps_df.groupby(['PlayId'])['ball_carrier_Dir_to_gap_ang_sim'].rank(method='first',ascending=True).astype(int)
        gaps_df['ball_carrier_Orientation'+momentum_suffix+'_to_gap_ang_sim_rank']=gaps_df.groupby(['PlayId'])['ball_carrier_Orientation_to_gap_ang_sim'].rank(method='first',ascending=True).astype(int)
        gaps_df['gap_width'+momentum_suffix+'_rank']=gaps_df.groupby(['PlayId'])['gap_width'].rank(method='first',ascending=False).astype(int)
        ###gaps_df['ball_carrier_Dir_to_gap_cos_sim_rank']=gaps_df.groupby(['PlayId'])['ball_carrier_Dir_to_gap_cos_sim'].rank(method='first',ascending=True).astype(int)
        ###gaps_df['ball_carrier_Orientation_to_gap_cos_sim_rank']=gaps_df.groupby(['PlayId'])['ball_carrier_Orientation_to_gap_cos_sim'].rank(method='first',ascending=True).astype(int)

        #Min gap distance to unblocked def    
        gap_dist_to_unblocked_def=gaps_df[['PlayId','NflId_left','gap_center_X','gap_center_Y']]\
            .merge(bc_dist_to_unblocked_def[['PlayId',X_field+'_def',Y_field+'_def']],
                      on='PlayId',
                      how='inner')
        def calc_dist_gap_to_def(x):
            return calc_distance(x['gap_center_X'],x['gap_center_Y'],x[X_field+'_def'],x[Y_field+'_def'])    
        gap_dist_to_unblocked_def['gap_distance_from_unblocked_def']=gap_dist_to_unblocked_def.apply(calc_dist_gap_to_def,axis=1)
        gap_min_dist_to_unblocked_def=gap_dist_to_unblocked_def.groupby(['PlayId','NflId_left'])[['gap_distance_from_unblocked_def']].min()

        gaps_df=gaps_df.merge(gap_min_dist_to_unblocked_def,
                             left_on=['PlayId','NflId_left'],
                             right_index=True,
                             how='left')\
                        .fillna(55)#CLEAN: distance to closest endzone corner

        gap_float_metrics=['gap_distance_from_ball_carrier', 
                 'gap_width',
                 'ball_carrier_Dir_to_gap_ang_sim','ball_carrier_Orientation_to_gap_ang_sim',
                 'gap_distance_from_unblocked_def']
        gap_bool_metrics=['Off_full_leverage','Off_half_leverage','Off_no_leverage'
                    ]

        def pivot_ranked_gaps(gap_rank_col,gap_values):
            play_gaps_rank_df= gaps_df.query('{gap_rank_col}<=6'.format(gap_rank_col=gap_rank_col))\
                    [['PlayId',gap_rank_col]+gap_values]\
                    .pivot(index='PlayId',columns=gap_rank_col
                           ,values=gap_values)
            play_gaps_rank_df.columns = [gap_rank_col+str(col[1])+'_'+col[0] for col in play_gaps_rank_df.columns.values]
            return play_gaps_rank_df
        play_gaps_dist_rank_df=pivot_ranked_gaps('gap_distance'+momentum_suffix+'_from_ball_carrier_rank',gap_float_metrics)
        play_gaps_dist_rank_df_bools=pivot_ranked_gaps('gap_distance'+momentum_suffix+'_from_ball_carrier_rank',gap_bool_metrics) 

        play_gaps_Dir_ang_rank_df=pivot_ranked_gaps('ball_carrier_Dir'+momentum_suffix+'_to_gap_ang_sim_rank',gap_float_metrics)
        play_gaps_Dir_ang_rank_df_bools=pivot_ranked_gaps('ball_carrier_Dir'+momentum_suffix+'_to_gap_ang_sim_rank',gap_bool_metrics) 

        play_gaps_Orientation_ang_rank_df=pivot_ranked_gaps('ball_carrier_Orientation'+momentum_suffix+'_to_gap_ang_sim_rank',gap_float_metrics)
        play_gaps_Orientation_ang_rank_df_bools=pivot_ranked_gaps('ball_carrier_Orientation'+momentum_suffix+'_to_gap_ang_sim_rank',gap_bool_metrics) 

        play_gaps_width_rank_df=pivot_ranked_gaps('gap_width'+momentum_suffix+'_rank',gap_float_metrics)
        play_gaps_width_rank_df_bools=pivot_ranked_gaps('gap_width'+momentum_suffix+'_rank',gap_bool_metrics) 

        ###play_gaps_Dir_cos_rank_df=gaps_df.query('ball_carrier_Dir_to_gap_cos_sim_rank<=6')[['PlayId','gap_distance_from_ball_carrier', 'gap_width','ball_carrier_Dir_to_gap_cos_sim','ball_carrier_Dir_to_gap_cos_sim_rank']]\
        ###    .pivot(index='PlayId',columns='ball_carrier_Dir_to_gap_cos_sim_rank',values=['gap_distance_from_ball_carrier', 'gap_width','ball_carrier_Dir_to_gap_cos_sim'])
        ###play_gaps_Dir_cos_rank_df.columns = ['ball_carrier_Dir_to_gap_cos_sim_rank'+str(col[1])+'_'+col[0] for col in play_gaps_Dir_cos_rank_df.columns.values]

        ###play_gaps_Orientation_cos_rank_df=gaps_df.query('ball_carrier_Orientation_to_gap_cos_sim_rank<=6')[['PlayId','gap_distance_from_ball_carrier', 'gap_width','ball_carrier_Orientation_to_gap_cos_sim','ball_carrier_Orientation_to_gap_cos_sim_rank']]\
        ###    .pivot(index='PlayId',columns='ball_carrier_Orientation_to_gap_cos_sim_rank',values=['gap_distance_from_ball_carrier', 'gap_width','ball_carrier_Orientation_to_gap_cos_sim'])
        ###play_gaps_Orientation_cos_rank_df.columns = ['ball_carrier_Orientation_to_gap_cos_sim_rank'+str(col[1])+'_'+col[0] for col in play_gaps_Orientation_cos_rank_df.columns.values]

        play_gaps_df=pd.concat([play_gaps_dist_rank_df,
                                play_gaps_dist_rank_df_bools,
                                play_gaps_Dir_ang_rank_df,
                                play_gaps_Dir_ang_rank_df_bools,
                                play_gaps_Orientation_ang_rank_df,
                                play_gaps_Orientation_ang_rank_df_bools,
                                play_gaps_width_rank_df,
                                play_gaps_width_rank_df_bools,
                               ### ,play_gaps_Dir_cos_rank_df,play_gaps_Orientation_cos_rank_df
                               ], axis=1)

        if verbose:    
            print("gaps"+momentum_suffix+"done")
            end = time.time()
            print(end - start)

            start = time.time()
        return play_gaps_df,\
                defender_hulls_df,\
                bc_dist_to_def_summaries,\
                play_def_left_to_right_rank_df,\
                play_bc_dist_to_def_rank_df,\
                bc_dist_to_unblocked_def_summaries,\
                play_bc_dist_to_unblocked_def_rank_df

    play_gaps_df,\
    defender_hulls_df,\
    bc_dist_to_def_summaries,\
    play_def_left_to_right_rank_df,\
    play_bc_dist_to_def_rank_df,\
    bc_dist_to_unblocked_def_summaries,\
    play_bc_dist_to_unblocked_def_rank_df=distance_based_metrics(X='X_to_YardLine',Y='Y_dist_from_center',momentum=False)

    play_gaps_sec_momentum_df,\
    defender_hulls_sec_momentum_df,\
    bc_dist_to_def_sec_momentum_summaries,\
    play_def_left_to_right_sec_momentum_rank_df,\
    play_bc_dist_to_def_sec_momentum_rank_df,\
    bc_dist_to_unblocked_def_sec_momentum_summaries,\
    play_bc_dist_to_unblocked_def_sec_momentum_rank_df=distance_based_metrics(X='X_to_YardLine',Y='Y_dist_from_center',momentum=True)
    
    #Combine features
    modeling_df=pd.get_dummies(
                    ball_carrier_df.merge(team_off_yard_reference,left_on='PossessionTeam',right_index=True, how='inner')\
                                    .merge(team_def_yard_reference,left_on='DefendingTeam',right_index=True, how='inner')\
                                    .merge(position_cnts,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(position_grp_cnts,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_gaps_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(defender_hulls_df, left_on='PlayId',right_index=True, how='inner')\
                                    .merge(bc_dist_to_def_summaries,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_def_left_to_right_rank_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_bc_dist_to_def_rank_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(bc_dist_to_unblocked_def_summaries,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_bc_dist_to_unblocked_def_rank_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_gaps_sec_momentum_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(defender_hulls_sec_momentum_df, left_on='PlayId',right_index=True, how='inner')\
                                    .merge(bc_dist_to_def_sec_momentum_summaries,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_def_left_to_right_sec_momentum_rank_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_bc_dist_to_def_sec_momentum_rank_df,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(bc_dist_to_unblocked_def_sec_momentum_summaries,left_on='PlayId',right_index=True, how='inner')\
                                    .merge(play_bc_dist_to_unblocked_def_sec_momentum_rank_df,left_on='PlayId',right_index=True, how='inner')\
                                    .drop(columns=['GameId','PlayId','X','Y','Orientation','Dir','NflId','YardLine','Yards','Orientation_corrected','ball_carrier','side_of_ball',
                                                   'TimeHandoff','TimeSnap',
                                                   'StadiumType','Stadium','Turf','GameWeather','FieldPosition','PlayerBirthDate','HomeTeamAbbr','VisitorTeamAbbr','Humidity',
                                                  "HomeScoreBeforePlay", "VisitorScoreBeforePlay",'OffensePersonnel','DefensePersonnel',
                                                  'NflIdRusher','PossessionTeam', 'DefendingTeam',
                                                   'bc_dist_to_def_min','bc_dist_to_unblocked_def_min',
                                                  'PlayerCollegeName','Location','WindSpeed', 'WindDirection','JerseyNumber','DisplayName',
                                                  'X_same_way','Y_same_way','Dir_same_way','Orientation_same_way'])
                )\
                .drop(columns=['Position_cnt_QB','Position_cnt_WR','Position_cnt_TE',
                               'Position_CB', 'Position_DE', 'Position_DT',
                               'Team_away','PlayDirection_right','Turf_type_Artificial'], errors='ignore')
    
    Yard_bins=['Yards'+str(i) for i in range(-99,100)]
    Yard_bin_target_df=pd.cut(ball_carrier_df['Yards'],bins=range(-100,100),labels=Yard_bins)
    if verbose:
        print("combine features done")
        end = time.time()
        print(end - start)
        print("preprocessing done")
    
    return modeling_df, Yard_bin_target_df, team_off_yard_reference, team_def_yard_reference, rusher_yard_reference, rusher_low_vol_reference