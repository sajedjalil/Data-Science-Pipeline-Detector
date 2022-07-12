# Gather play level data.
# Go through NGS data and get info like events, etc.
import pandas as pd
import numpy as np
import tracemalloc
import gc

tracemalloc.start()

# Read in non-NGS data sources
ppd = pd.read_csv('../input/player_punt_data.csv')
gd = pd.read_csv('../input/game_data.csv')
pprd = pd.read_csv('../input/play_player_role_data.csv')
vr = pd.read_csv('../input/video_review.csv')
vfi = pd.read_csv('../input/video_footage-injury.csv')
pi = pd.read_csv('../input/play_information.csv')

all_dfs = [ppd, gd, pprd, vr, vfi, pi]
for mydf in all_dfs:
    mydf.columns = [col.lower() for col in mydf.columns]

NGS_csv_files = [
    'NGS-2016-pre.csv',
    'NGS-2016-reg-wk1-6.csv',
    'NGS-2016-reg-wk13-17.csv',
    'NGS-2016-reg-wk7-12.csv',
    'NGS-2017-post.csv',
    'NGS-2016-post.csv',
    'NGS-2017-pre.csv',
    'NGS-2017-reg-wk1-6.csv',
    'NGS-2017-reg-wk13-17.csv',
    'NGS-2017-reg-wk7-12.csv',
]

ppd_unique = ppd.groupby('gsisid').agg(lambda x: ', '.join(x)).reset_index()

# Detailed role info
# I made this myself and may include errors require me to rerun later
role_info_dict = {'GL': ['Gunner', 'Punting_Team'],
                  'GLi': ['Gunner', 'Punting_Team'],
                  'GLo': ['Gunner', 'Punting_Team'],
                  'GR': ['Gunner', 'Punting_Team'],
                  'GRi': ['Gunner', 'Punting_Team'],
                  'GRo': ['Gunner', 'Punting_Team'],
                  'P': ['Punter', 'Punting_Team'],
                  'PC': ['Punter_Protector', 'Punting_Team'],
                  'PPR': ['Punter_Protector', 'Punting_Team'],
                  'PPRi': ['Punter_Protector', 'Punting_Team'],
                  'PPRo': ['Punter_Protector', 'Punting_Team'],
                  'PDL1': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL2': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL3': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR1': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR2': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR3': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL5': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL6': ['Defensive_Lineman', 'Returning_Team'],
                  'PFB': ['PuntFullBack', 'Punting_Team'],
                  'PLG': ['Punting_Lineman', 'Punting_Team'],
                  'PLL': ['Defensive_Backer', 'Returning_Team'],
                  'PLL1': ['Defensive_Backer', 'Returning_Team'],
                  'PLL3': ['Defensive_Backer', 'Returning_Team'],
                  'PLS': ['Punting_Longsnapper', 'Punting_Team'],
                  'PLT': ['Punting_Lineman', 'Punting_Team'],
                  'PLW': ['Punting_Wing', 'Punting_Team'],
                  'PRW': ['Punting_Wing', 'Punting_Team'],
                  'PR': ['Punt_Returner', 'Returning_Team'],
                  'PRG': ['Punting_Lineman', 'Punting_Team'],
                  'PRT': ['Punting_Lineman', 'Punting_Team'],
                  'VLo': ['Jammer', 'Returning_Team'],
                  'VR': ['Jammer', 'Returning_Team'],
                  'VL': ['Jammer', 'Returning_Team'],
                  'VRo': ['Jammer', 'Returning_Team'],
                  'VRi': ['Jammer', 'Returning_Team'],
                  'VLi': ['Jammer', 'Returning_Team'],
                  'PPL': ['Punter_Protector', 'Punting_Team'],
                  'PPLo': ['Punter_Protector', 'Punting_Team'],
                  'PPLi': ['Punter_Protector', 'Punting_Team'],
                  'PLR': ['Defensive_Backer', 'Returning_Team'],
                  'PRRo': ['Defensive_Backer', 'Returning_Team'],
                  'PDL4': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR4': ['Defensive_Lineman', 'Returning_Team'],
                  'PLM': ['Defensive_Backer', 'Returning_Team'],
                  'PLM1': ['Defensive_Backer', 'Returning_Team'],
                  'PLR1': ['Defensive_Backer', 'Returning_Team'],
                  'PLR2': ['Defensive_Backer', 'Returning_Team'],
                  'PLR3': ['Defensive_Backer', 'Returning_Team'],
                  'PLL2': ['Defensive_Backer', 'Returning_Team'],
                  'PDM': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR5': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR6': ['Defensive_Lineman', 'Returning_Team'],
                  }

role_info = pd.DataFrame.from_dict(role_info_dict,
                                   orient='index',
                                   columns=['generalized_role', 'punting_returning_team']) \
    .reset_index() \
    .rename(columns={'index': 'role'})

pprd_detailed = pd.merge(pprd, role_info, how='left', on='role')

play_count = 0
plays_all_data = pd.DataFrame()
plays_action = pd.DataFrame()
for ngs_file in NGS_csv_files:
    # Loop through each file so that we save space
    ngs = pd.read_csv('../input/{}'.format(ngs_file))
    ngs.columns = [col.lower() for col in ngs.columns]

    # groupby and loop through play
    grouped = ngs.groupby(['season_year', 'gamekey', 'playid'])

    for s_gk_pid, df in grouped:
        play_count += 1
        try:
            print('========RUNNING FOR PLAY NUMBER {} =============='.format(play_count))
            print('Running for season year gamekey playid: {}'.format(s_gk_pid))
            try:
                print(pd.merge(df, pi)['playdescription'].values[0])
            except Exception as e:
                print('Brole when trying to merge NGS with play info')
                print('exception {}'.format(e))
                with open("broke_plays.txt", "a") as myfile:
                    myfile.write("NO PLAY INFO {} \n".format(s_gk_pid))
                continue
            rows_before = len(df)

            # Merge possible player jersey number and position
            df = pd.merge(df, ppd_unique, how='left', on='gsisid')
            if len(df) != rows_before:
                raise 'Shape has changed! This is not right'

            # Merge player punt role. Drop any player that does not have a role in the play
            # This includes players on sideline who are captured on the field during the play
            df = pd.merge(df, pprd_detailed,
                          on=['season_year', 'gamekey', 'playid', 'gsisid'], how='inner')

            df = pd.merge(df, vr, on=['season_year', 'gamekey', 'playid'],
                          how='left', suffixes=('', '_injured'))

            # Get all events and the event times within the play
            events = df.groupby(['event', 'time'])
            for event, d in events:
                df[event[0]] = event[1]  # Save event as column with time of event

            df['mph'] = df['dis'] * 20.4545455  # miles per hour
            df['injured_player'] = df.apply(
                lambda row: True if row['gsisid'] == row['gsisid_injured'] else False, axis=1)
            df['primary_partner_player'] = df.apply(
                lambda row: True if row['gsisid'] == row['primary_partner_gsisid'] else False, axis=1)

            # Find out if play is left to right - or right to left
            try:
                punt_returner_x_at_snap = df.loc[(df['role'] == 'PR') & (df['event'] == 'ball_snap')]['x'].values[0]
                long_snapper_x_at_snap = df.loc[(df['role'] == 'PLS') & (df['event'] == 'ball_snap')]['x'].values[0]
                if punt_returner_x_at_snap < long_snapper_x_at_snap:
                    df['left_to_right'] = False
                else:
                    df['left_to_right'] = True
            except Exception as e:
                print('Broke when trying to determine if play is going to left or right')
                df['left_to_right'] = np.nan
                with open("broke_plays.txt", "a") as myfile:
                    myfile.write("COULDNT DETERMINE LEFT TO RIGHT {} \n".format(s_gk_pid))
            # Join play information
            # Comment out because uncessary
            # df = pd.merge(df, pi, on=['season_year',
            #               'gamekey', 'playid'], how='left')
            
            # join with master df
            df.to_parquet('playlevel-{}-{}-{}-all_data.parquet'.format(s_gk_pid[0], s_gk_pid[1], s_gk_pid[2]))
            plays_all_data = pd.concat([df, plays_all_data])

            ##############################################
            # Cut off from start of play to end of play
            ##############################################

            # Only keep time within the play that matters
            print(df['event'].unique())

            if len(df.loc[df['event'] == 'ball_snap']['time'].values) == 0:
                print('........No Snap for this play')
                ball_snap_time = df['time'].min()
            else:
                ball_snap_time = df.loc[df['event'] == 'ball_snap']['time'].values.min()

            try:
                end_time = df.loc[(df['event'] == 'out_of_bounds') |
                                  (df['event'] == 'downed') |
                                  (df['event'] == 'tackle') |
                                  (df['event'] == 'punt_downed') |
                                  (df['event'] == 'fair_catch') |
                                  (df['event'] == 'touchback') |
                                  (df['event'] == 'touchdown')]['time'].values.max()
            except ValueError as e:
                print('Broke when trying to find the end of the play')
                print('.......No end to play')
                end_time = df['time'].values.max()
                with open("broke_plays.txt", "a") as myfile:
                    myfile.write("NO END TO THE PLAY {} \n".format(s_gk_pid))

            df = df.loc[(df['time'] >= ball_snap_time) & (df['time'] <= end_time)]

            if len(df) == 0:
                print('BROKE FOR {} - No NGS Data Available'.format(s_gk_pid))
            else:
                df.to_parquet('playlevel-{}-{}-{}-during_play.parquet'.format(
                    s_gk_pid[0], s_gk_pid[1], s_gk_pid[2]))
                plays_action = pd.concat([df, plays_action])
        except Exception as e:
            print('Broke somewhere else in the process')
            print('Exception {}'.format(e))
            with open("broke_plays.txt", "a") as myfile:
                myfile.write("BROKE SOMEHWERE ELSE {} \n".format(s_gk_pid))
        if play_count % 500 == 0:
            # Save to file every 500 plays
            print('Play number a multiple of 500, saving parquet files')
            plays_all_data.to_parquet('plays_all_data{}.parquet'.format(play_count))
            plays_action.to_parquet('plays_action{}.parquet'.format(play_count))
            del plays_all_data
            del plays_action
            gc.collect()
            plays_all_data = pd.DataFrame()
            plays_action = pd.DataFrame()
    # Remove data from memory
    del ngs
    gc.collect()