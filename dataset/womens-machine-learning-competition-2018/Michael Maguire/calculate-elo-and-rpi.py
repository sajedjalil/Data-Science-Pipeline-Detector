# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools

data_dir = '../input/'
start_year = 2003
end_year   = 2017

#===================================================================================================
def double_df(datafile, stats):
    df = datafile.copy()
    add_df = datafile.copy()
    df['target'] = np.ones(df.shape[0])
    rename_dict = {'team_id':'opp_id','opp_id':'team_id'}
    for s in stats:
        rename_dict['team_'+s] = 'opp_'+s
        rename_dict['opp_'+s] = 'team_'+s
    loc_dict = {'H':'A','N':'N','A':'H'}
    add_df['target'] = np.zeros(add_df.shape[0])
    add_df.rename(columns=rename_dict, inplace=True)
    add_df['team_loc'] = add_df.team_loc.apply(lambda x: loc_dict[x])
    df = df.append(add_df)
    df.sort_values(['Season','team_id','game_day'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_RPI(df, tm_list, strt, end, rank_dts):

    def wp_weight(site, targ):
        if (site=='A') & (targ==1):
            return 1.4
        elif (site=='A') & (targ==0):
            return 0.6
        elif (site=='H') & (targ==1):
            return 0.6
        elif (site=='H') & (targ==0):
            return 1.4
        else:
            return 1.0   
    
    def get_OWP(games, team_list):
        oppWP_df = pd.DataFrame()
        for t in tm_list:
            tm  = games[['team_id','opp_id']][games.team_id==t].copy()
            opp = games[['team_id','opp_id','target']][games.opp_id!=t].copy()
            temp = opp[['team_id','target']].groupby('team_id', as_index=False).mean()
            temp.rename(columns={'team_id':'opp_id','target':'OWP'}, inplace=True)
            tm = tm.merge(temp, on='opp_id', how='left')
            oppWP_df = oppWP_df.append(tm[['team_id','OWP']])
        out = oppWP_df.groupby('team_id', as_index=False).mean()
        out['opp_id'] = out.team_id
        out['OOWP'] = out.OWP
        return out
    
    #=========================================
    df['wp_wt'] = list(map(lambda x,y: wp_weight(x,y), df.team_loc, df.target))
    df['win_wt'] = df.target * df.wp_wt
    
    rpi_reg = pd.DataFrame()
    for s in range(strt, end+1):
        print('   ...'+str(s)+' season...')
        season = df[df.Season==s].copy()
        for d in rank_dts:
            WP = season[['team_id','wp_wt','win_wt']][season.game_day<d].groupby('team_id', as_index=False).sum()
            WP['WP'] = WP.win_wt / WP.wp_wt
            OWP = get_OWP(season[season.game_day<d], tm_list)
            
            temp = season[['team_id','opp_id']][season.game_day<d]
            temp = temp.merge(OWP[['opp_id','OOWP']], on='opp_id', how='left')
            OOWP = temp[['team_id','OOWP']].groupby('team_id', as_index=False).mean()
            
            stats = WP[['team_id','WP']].merge(OWP[['team_id','OWP']], on='team_id', how='left')
            stats = stats.merge(OOWP, on='team_id', how='left')
            stats.OWP.fillna(0, inplace=True)
            stats.OOWP.fillna(0, inplace=True)
            stats['team_RPI_score'] = (0.25*stats.WP) + (0.5*stats.OWP) + (0.25*stats.OOWP)
            stats['team_RPI_rank'] = stats.team_RPI_score.rank(method='min', ascending=False)
            stats['Season'] = [s]*stats.shape[0]
            stats['rank_day'] = [d]*stats.shape[0]
            stats['opp_id'] = stats.team_id
            stats['opp_RPI_score'] = stats.team_RPI_score
            stats['opp_RPI_rank'] = stats.team_RPI_rank
            rpi_reg = rpi_reg.append(stats)
    
    rpi_reg.sort_values(['Season','team_id','rank_day'], inplace=True)
    rpi_trny = rpi_reg[rpi_reg.rank_day==133]
    return rpi_reg, rpi_trny

#===================================================
def get_season_elo(datafile, tm_list):
    mean_elo = 1500
    elo_width = 400
    k_factor = 32
    
    def expected_result(elo_a, elo_b):
        expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
        return expect_a
    
    def update_elo(winner_elo, loser_elo):
        expected_win = expected_result(winner_elo, loser_elo)
        change_in_elo = k_factor * (1-expected_win)
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
        return winner_elo, loser_elo
    
    def update_end_of_season(elos):
        diff_from_mean = elos - mean_elo
        elos -= diff_from_mean/3
        return elos
    
    current_elos = pd.Series(np.ones(len(tm_list)) * mean_elo, index=tm_list)
    regular = datafile[['Season','team_id','opp_id','game_day']].copy()
    reg_tm = []
    reg_op = []
    for i in range(datafile.shape[0]):
        reg_tm += [current_elos[datafile.team_id[i]]]
        reg_op += [current_elos[datafile.opp_id[i]]]
        if datafile.target[i] == 1:
            wteam = datafile.team_id[i]
            lteam = datafile.opp_id[i]
        else:
            wteam = datafile.opp_id[i]
            lteam = datafile.team_id[i]
        
        w_elo_before = current_elos[wteam]
        l_elo_before = current_elos[lteam]
        w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before)
        current_elos[wteam] = w_elo_after
        current_elos[lteam] = l_elo_after
        
    #current_elos = update_end_of_season(current_elos)
    regular['team_ELO_score'] = reg_tm
    regular['opp_ELO_score'] = reg_op
    tourney = pd.DataFrame({'team_id':tm_list, 'team_elo_score':current_elos.values})
    return regular, tourney

def get_elo_scores(datafile, team_id, strt, end):
    reg_out  = pd.DataFrame()
    trny_out = pd.DataFrame()
    for y in range(strt, end+1):
        print('   ...'+str(y)+' season...')
        elo_games = datafile[(datafile.team_id < datafile.opp_id) & (datafile.Season==y)].copy()
        elo_games.sort_values(['Season','game_day'], inplace=True)
        elo_games.reset_index(drop=True, inplace=True)
        elo_reg, elo_trny = get_season_elo(elo_games, team_id)
        elo_trny['Season'] = [y]*elo_trny.shape[0]
        reg_out = reg_out.append(elo_reg)
        trny_out = trny_out.append(elo_trny)
    return reg_out, trny_out

def get_elo_ranks(datafile, rk_dates, strt, end):
    reg_out = pd.DataFrame()
    for item in itertools.product(list(range(strt, end+1)), rk_dates):
        games = datafile[(datafile.Season==item[0]) & (datafile.game_day<item[1])].drop_duplicates(['Season','team_id'], keep='last')
        games['team_ELO_rank'] = games.team_ELO_score.rank(method='min', ascending=False)
        rks = games[['Season','team_id','team_ELO_rank']].copy()
        rks['rank_day'] = [item[1]]*rks.shape[0]
        reg_out = reg_out.append(rks)
    reg_out['opp_id'] = reg_out.team_id
    reg_out['opp_ELO_rank'] = reg_out.team_ELO_rank
    trny_out = reg_out[['Season','team_id','team_ELO_rank']][reg_out.rank_day==133]
    return reg_out, trny_out

#===================================================================================================
#===================================================================================================
teams = pd.read_csv(data_dir+'WTeams.csv', usecols=['TeamID','TeamName'])
teams.rename(columns={'TeamID':'team_id','TeamName':'team_name'}, inplace=True)
team_list = teams.team_id.tolist()

games_regular = pd.read_csv(data_dir+'WRegularSeasonCompactResults.csv')
games_regular.rename(columns={'DayNum':'game_day','WTeamID':'team_id','WScore':'team_pts','LTeamID':'opp_id','LScore':'opp_pts','WLoc':'team_loc'}, inplace=True)
games_regular = games_regular[games_regular.Season>=start_year]
games_regular = double_df(games_regular, ['pts'])
games_regular['rank_day']  = games_regular.game_day.apply(lambda x: x if (x%7==0) | (x==133) else np.min([133,(int(x/7)+1)*7]))
games_regular['rank_day']  = games_regular.groupby(['Season','team_id'])['rank_day'].ffill()
games_regular['rank_day']  = games_regular.groupby(['Season','team_id'])['rank_day'].bfill()
rank_dates = games_regular.rank_day.unique().tolist()
rank_dates.sort()

print('-------------------------')
print('Get RPI scores & ranks')
rpi_regular, rpi_tourney = get_RPI(games_regular, team_list, start_year, end_year, rank_dates)

print('-------------------------')
print('Get ELO scores & ranks')
elo_regular, elo_tourney = get_elo_scores(games_regular, team_list, start_year, end_year)
elo_rank_reg, elo_rank_trny = get_elo_ranks(elo_regular, rank_dates, start_year, end_year)

regular = games_regular[['Season','team_id','opp_id','game_day','rank_day']].merge(rpi_regular[['Season','team_id','rank_day','team_RPI_score','team_RPI_rank']], on=['Season','team_id','rank_day'], how='left')
regular = regular.merge(rpi_regular[['Season','opp_id','rank_day','opp_RPI_score','opp_RPI_rank']], on=['Season','opp_id','rank_day'], how='left')
regular = regular.merge(elo_regular[['Season','team_id','game_day','team_ELO_score']], on=['Season','team_id','game_day'], how='left')
regular = regular.merge(elo_regular[['Season','opp_id','game_day','opp_ELO_score']], on=['Season','opp_id','game_day'], how='left')
regular = regular.merge(elo_rank_reg[['Season','team_id','rank_day','team_ELO_rank']], on=['Season','team_id','rank_day'], how='left')
regular = regular.merge(elo_rank_reg[['Season','opp_id','rank_day','opp_ELO_rank']], on=['Season','opp_id','rank_day'], how='left')

tourney = rpi_tourney.merge(elo_tourney, on=['Season','team_id'], how='left')
tourney = tourney.merge(elo_rank_trny, on=['Season','team_id'], how='left')

regular.to_csv('regular_rankings.csv', index=False)
tourney.to_csv('pretrny_rankings.csv', index=False)

# Any results you write to the current directory are saved as output.