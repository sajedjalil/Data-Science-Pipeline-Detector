# Hello all, this is my approach for this competition.
# I think this approach is not used much by other people.
# On disadvantage is that the code is slow, so if you have any suggestions for speed improvements, please let me know.
# also the final score in this script is not 0.49, but with more data and feature engineering/selection I came to 0.49 on local CV


# Load in packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import progressbar
from sklearn.metrics.classification import log_loss
from sklearn.utils import shuffle 
import os



# Load in data
df_seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
df_season_detailed = pd.read_csv("../input/RegularSeasonDetailedResults.csv")
df_season_detailed.drop(labels=['WLoc','NumOT'],inplace=True,axis=1)

# Make sure Every game is reversed to we have wining and lossing games 
general_columns = df_season_detailed[['Season',"DayNum"]]
TeamA = df_season_detailed[[col for col in df_season_detailed if col.startswith('W')]]
TeamB = df_season_detailed[[col for col in df_season_detailed if col.startswith('L')]]

TeamA.columns = [word.replace('W','TeamA_') for word in TeamA.columns]
TeamB.columns = [word.replace('L','TeamB_') for word in TeamB.columns]

general = list(general_columns.columns)
TeamA_columns = list(TeamA.columns) 
TeamB_columns = list(TeamB.columns)

df_season_switched = pd.concat([general_columns, TeamB,TeamA],axis=1)
df_season = pd.concat([general_columns,TeamA,TeamB],axis=1)
general.extend(TeamA_columns)
general.extend(TeamB_columns)
df_season_switched.columns = general
df_season.columns = general

df_season_extended = pd.concat([df_season,df_season_switched],axis=0)
df_season_extended = df_season_extended.sort_values(by="DayNum")
df_season_extended = df_season_extended.reset_index()
del df_season_extended['index']
df_season_extended["TeamA_score_against"] = df_season_extended['TeamB_Score'] 
df_season_extended['TeamB_score_against'] = df_season_extended['TeamA_Score']
df_season_extended = df_season_extended.rename(columns={"TeamA_TeamID":"TeamA","TeamB_TeamID":"TeamB"})

# Delete variables that are not needed any more
del  df_season, df_season_detailed,general_columns,df_season_switched,TeamA,TeamA_columns,TeamB,TeamB_columns


#%% Seed to in converter
def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int

# Transform Seed and change name
df_seeds['Seed'] = df_seeds.Seed.apply(seed_to_int) 
df_seeds_TeamA = df_seeds.rename(columns={"TeamID":"TeamA","Seed":"TeamA_Seed"})
df_seeds_TeamB = df_seeds.rename(columns={"TeamID":"TeamB","Seed":"TeamB_Seed"})    

# Make dataframe in which only games where at least on of the teams that made if to the NCAA final tournament played.
df_season_extended = pd.merge(df_season_extended,df_seeds_TeamA, how = 'left', on = ['Season','TeamA'])
df_season_extended = pd.merge(df_season_extended, df_seeds_TeamB, how = 'left', on = ['Season','TeamB'])
df_season_extended = df_season_extended.fillna(17) # Fillna with 17
df_season_extended = df_season_extended[~((df_season_extended['TeamA_Seed'] == 17) & (df_season_extended['TeamB_Seed'] == 17))] # Remove games with both 17 seed (these did not make it to the tournament at the end of the season)


# Transform training set
# Last_x_games is used to choose how many games back the statistics should be picked.
# For example: If this is 7, then the last 7 Games for both team A and team B are picked and the mean is calculated.
def Transform_train(df_season2017, Last_x_games=7):
    count = 0
    totalbar = progressbar.ProgressBar(maxval=len(df_season2017)).start()
    
    All = pd.DataFrame()
    U_seasons = sorted(list(df_season2017['Season'].unique()))
    for j in range(0,len(U_seasons)):
        Seasons = df_season2017[df_season2017['Season'] == U_seasons[j]]
    
        for i in range(0,len(Seasons)):
            count += 1
            totalbar.update(count)
            if i < 1:
                Games = Seasons.reset_index() 
                Games = Games.sort_values(by="DayNum")
            else:
                Games = Seasons[:-i].reset_index() # Continuesly delete last columns
            
            del Games['index']
            Last_game = Games.iloc[-1:].reset_index() # Pick last game in dataframe so both team A and teamB can be picked
            
            TeamA = Last_game.loc[0,"TeamA"] # Pick team A 
            TeamB = Last_game.loc[0,'TeamB'] # Pick team B
            DayNum = Last_game.loc[0,"DayNum"]
            Season = Last_game.loc[0,"Season"]
            TeamA_score = Last_game.loc[0,"TeamA_Score"] # Remember Actual score of TeamA
            TeamB_score = Last_game.loc[0,'TeamB_Score'] # Remember Actual score of TeamB
            
            Games2 = Games[:-1] # Delete last column so statistics of current game are not picked
            Games2 = Games2[~((Games2['TeamA'] == TeamB) & (Games2['TeamB'] == TeamA) & (Games2['DayNum'] == DayNum) & (Games2['Season'] == Season))] # delete reversed game so this is not picked
        
            TeamA_data = Games2[Games2['TeamA'] == TeamA].tail(Last_x_games)
            TeamB_data = Games2[Games2['TeamB'] == TeamB].tail(Last_x_games)
            
            General_columns = ['Season']
            TeamA_columns = [col for col in TeamA_data if col.startswith("TeamA")] 
            TeamB_columns = [col for col in TeamB_data if col.startswith("TeamB")] 
            General_columns.extend(TeamA_columns)
            TeamA_data = TeamA_data[General_columns].reset_index()
            TeamB_data = TeamB_data[TeamB_columns].reset_index()
            del TeamA_data['index'], TeamB_data['index']

            
            if (len(TeamA_data) < Last_x_games) or (len(TeamB_data) < Last_x_games): # If nr of Games is smaller than nr of games we want, then continue and pick next game.
                continue
            else:
                if len(All) == 0:
                    All = pd.concat([TeamA_data, TeamB_data],axis=1)
                    All.loc['mean'] = All.mean()
                    All = All.tail(1)
                    All['TeamA_match_score'] = TeamA_score # Return actual score of game
                    All['TeamB_match_score'] = TeamB_score # Return actual score of game
                    
                else:
                    Home_away = pd.concat([TeamA_data, TeamB_data], axis=1)
                    Home_away.loc['mean'] = Home_away.mean()
                    Home_away = Home_away.tail(1) 
                    Home_away['TeamA_match_score'] = TeamA_score #Return actual score of game
                    Home_away['TeamB_match_score'] = TeamB_score #Return actual score of game
                    All = All.append(Home_away)
                    All = All.reset_index()  
                    del All['index']
    return All
    
    
#%% Check how wel the model would perform with crossvalidation

def Predict_proba_crossvalidation(model,X_train,Y_train,folds=10):
    len_folds = int(len(X_train) / folds)    
    begin = 0
    end = len_folds
    scores = [] 
    X_train, Y_train = shuffle(X_train,Y_train,random_state = 10)
    X_train, Y_train = pd.DataFrame(X_train).reset_index(), pd.DataFrame(Y_train).reset_index()
    del X_train['index'], Y_train['index']
    for i in range(0,folds):
        rows = list(range(begin, end))
        X_train1, Y_train1 = X_train[~X_train.index.isin(rows)], Y_train[~Y_train.index.isin(rows)]
        X_val, Y_val = X_train[X_train.index.isin(rows)], Y_train[Y_train.index.isin(rows)]
        Y_train1 = np.array(Y_train1).reshape(len(Y_train1),)
        Y_val = np.array(Y_val).reshape(len(Y_val,))
        model = model
        model.fit(X_train1, Y_train1)
        Y_pred = pd.DataFrame(model.predict_proba(X_val))
        Y_pred = list(Y_pred.iloc[:,1])
        
        score = log_loss(Y_val,Y_pred)
        scores.append(score)
        begin += len_folds
        end += len_folds 
        
        if i == (folds-2):
            end = len(X_train)
    return scores

#%% Make target
def target(c):
    if c['TeamA_match_score'] > c["TeamB_match_score"]:
        return int('1') 
    else:
        return int('0')  

        
#%% Transform train data and make X_train and y_train
df_season_extended2 = df_season_extended[df_season_extended["Season"].isin([2018])] # Pick only 2018 games
X_train = Transform_train(df_season_extended2, Last_x_games=10) # transform and use last 10 games
print("X_train consists of",len(X_train),"Games")
y_train = X_train.apply(target,axis=1)
X_train.drop(labels = ['TeamA_match_score',"TeamB_match_score","TeamB","TeamA","Season"],inplace=True,axis=1)

#%% Test scores on own predict_proba_crossvalidation function
model = LogisticRegression(C=0.001)
scores = Predict_proba_crossvalidation(model,X_train,y_train,folds=10)
print("based on",len(X_train),"games, mean score =",np.mean(scores))
    
