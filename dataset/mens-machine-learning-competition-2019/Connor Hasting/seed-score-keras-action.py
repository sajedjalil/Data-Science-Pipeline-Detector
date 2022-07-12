# This is my first Kaggle competition. I also don't know much about basketball so I am going to keep this pretty simple to start with.
# I will be using the season, winning score, losing score, seed, seed region
# I would like to feed my model the seed difference, score difference and the winning region

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from statistics import mean 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from subprocess import check_output
print(check_output(["ls", "../"]).decode("utf8"))
# We want to load files in from here
data_directory = '../input/datafiles/'
final_submission_data_directory = '../input/stage2datafiles'

base_directory = '../input/'
raw_seeds = pd.read_csv(data_directory + 'NCAATourneySeeds.csv')
compact_results = pd.read_csv(data_directory + 'NCAATourneyCompactResults.csv')
compact_results.append(pd.read_csv(data_directory + 'RegularSeasonCompactResults.csv'), ignore_index=True) # We want as much data as possible so let's add all regular season data too
massey_ordinals = pd.read_csv( base_directory + '/masseyordinals/MasseyOrdinals.csv')

raw_seeds_stage_2 = pd.read_csv(final_submission_data_directory + '/NCAATourneySeeds.csv')
compact_results_stage_2 = pd.read_csv(final_submission_data_directory + '/NCAATourneyCompactResults.csv')
compact_results_stage_2.append(pd.read_csv(final_submission_data_directory + '/RegularSeasonCompactResults.csv'), ignore_index=True) # We want as much data as possible so let's add all regular season data too
#print(massey_ordinals.head())
massey_ordinals.drop(labels=['RankingDayNum'], inplace=True, axis=1) # This is the region and placement combined, we no longer need this

#print(compact_results.size)

#print(raw_seeds.head())
#print(compact_results.head())

#Get just the digits from the seeding. Return as int
def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int
    
def extract_seed_region(seed):
    # Get the region from the seeding
    s_reg = seed[0:1]
    return s_reg
    
print("Seeds")
raw_seeds['seed_int'] = raw_seeds.Seed.apply(seed_to_int)
raw_seeds['seed_region'] = raw_seeds['Seed'].apply(extract_seed_region)
raw_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the region and placement combined, we no longer need this

raw_seeds_stage_2['seed_int'] = raw_seeds_stage_2.Seed.apply(seed_to_int)
raw_seeds_stage_2['seed_region'] = raw_seeds_stage_2['Seed'].apply(extract_seed_region)
raw_seeds_stage_2.drop(labels=['Seed'], inplace=True, axis=1) # This is the region and placement combined, we no longer need this
#print(raw_seeds.head())
#print(compact_results.head())

# get rid of the compact results info we don't need
compact_results.drop(labels=['DayNum', 'WLoc', 'NumOT'], inplace=True, axis=1)
compact_results_stage_2.drop(labels=['DayNum', 'WLoc', 'NumOT'], inplace=True, axis=1)

#print(compact_results.head())

# Now we will create a table of all of the seeds with WTeamID and then one for LTeamID, we will be using these to merge into the compact results
winning_seeds = raw_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed', 'seed_region':"WRegion"})
losing_seeds = raw_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed', 'seed_region':"LRegion"})
winning_seeds_stage_2 = raw_seeds_stage_2.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed', 'seed_region':"WRegion"})
losing_seeds_stage_2 = raw_seeds_stage_2.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed', 'seed_region':"LRegion"})

temp = pd.merge(left=compact_results, right=winning_seeds, how='left', on=['Season', 'WTeamID'])
temp_stage_2 = pd.merge(left=compact_results_stage_2, right=winning_seeds_stage_2, how='left', on=['Season', 'WTeamID'])

#print(temp.head())
temp2 = pd.merge(left=temp, right=losing_seeds, on=['Season', 'LTeamID'])
temp2_stage_2 = pd.merge(left=temp_stage_2, right=losing_seeds_stage_2, on=['Season', 'LTeamID'])

#print(temp2.head())

#winning_ordinals = massey_ordinals.rename(columns={'TeamID':'WTeamID', 'OrdinalRank':'WOrdinalRank'})
#print("Ordinals")
# Now we will do the same thing to losing_ordinals
#winning_ordinals['SystemName'] = pd.Categorical(winning_ordinals.SystemName)
#winning_ordinals['SystemName'] = pd.factorize(winning_ordinals['SystemName'])[0]
# Now we convert this to the lowest possible ints
#winning_ordinals.apply(pd.to_numeric,downcast='unsigned')
#print(winning_ordinals.columns)

#team_season_sum_massey_win = winning_ordinals.groupby(['WTeamID', 'Season'], as_index=False).WOrdinalRank.mean()
#print(team_season_sum_massey_win)

#losing_ordinals = massey_ordinals.rename(columns={'TeamID':'LTeamID', 'OrdinalRank':'LOrdinalRank'})

#temp3 = pd.merge(left=temp2, right=team_season_sum_massey_win, on=['Season','WTeamID'])

# We have to make temp3 and losing_ordinals use less space to remove the memory constraints

#So first we convert all strings to categories, then change the categories to numbers
temp2['WRegion'] = pd.Categorical(temp2.WRegion)
temp2['WRegion'] = pd.factorize(temp2['WRegion'])[0]
temp2['LRegion'] = pd.Categorical(temp2.LRegion)
temp2['LRegion'] = pd.factorize(temp2['LRegion'])[0]

temp2_stage_2['WRegion'] = pd.Categorical(temp2_stage_2.WRegion)
temp2_stage_2['WRegion'] = pd.factorize(temp2_stage_2['WRegion'])[0]
temp2_stage_2['LRegion'] = pd.Categorical(temp2_stage_2.LRegion)
temp2_stage_2['LRegion'] = pd.factorize(temp2_stage_2['LRegion'])[0]

# We will then downcast them to the lowest possible byte size possible
#temp3.apply(pd.to_numeric,downcast='unsigned')

#temp3down = temp3.select_dtypes(include=['int'])
#print(temp3.dtypes)
#print(temp3.head())


# Now we will do the same thing to losing_ordinals
#losing_ordinals['SystemName'] = pd.Categorical(losing_ordinals.SystemName)
#losing_ordinals['SystemName'] = pd.factorize(losing_ordinals['SystemName'])[0]
# Now we convert this to the lowest possible ints
#losing_ordinals.apply(pd.to_numeric,downcast='unsigned')
#team_season_sum_massey_lose = losing_ordinals.groupby(['LTeamID', 'Season'], as_index=False).LOrdinalRank.mean()
#print(team_season_sum_massey_lose)

full_data = temp2.copy() #pd.merge(left=temp3, right=team_season_sum_massey_lose, on=['Season', 'LTeamID'])
full_data_stage_2 = temp2_stage_2.copy()

print(full_data_stage_2.dtypes)
# Now we can remove any duplicates because I'm sure I duplicated things somehwere (actually looks like I did, went from a size of 86115156 to 27407232)
#full_data.drop_duplicates(keep=False,inplace=True) 

#print(full_data.head())

#print(full_data.shape)
#print(full_data.size)
#print(full_data.dtypes)

#print(full_data.head())
#print(full_data.size)
#print(full_data)

### TODO ###
# We will want to grab the winning team's average score for the season and the losing team's average total score for the season
# We will have the score be the labels
# This will be for the final one
#full_data['WAvgScore'] = full_data['Season', 'WTeamID', 'WScore'].apply(average_score)

# Result of 1 means that team 0 won
full_data['Result'] = 1

# I will essentially be doubling the data points to train the model on both the winning and losing scenarios since I am not using the differences in data
reverse_data = full_data.copy()

# We can now state that team 1 will win since we'll be swapping those values around
reverse_data.Result = 0
#print(reverse_data.head())
#print(full_data.head())
# I will be standardizing the names. For the full data it will be 0 for the winning team and for the reverse it will be 1 for the winning team
full_data.rename(columns={"WTeamID": "TeamID0", "LTeamID": "TeamID1", "WScore": "Score0", "LScore": "Score1", "WSeed": "Seed0", "WRegion": "Region0", "LSeed": "Seed1", "LRegion": "Region1"}, inplace=True)
reverse_data.rename(columns={"LTeamID": "TeamID0", "WTeamID": "TeamID1", "LScore": "Score0", "WScore": "Score1", "LSeed": "Seed0", "LRegion": "Region0", "WSeed": "Seed1", "WRegion": "Region1"}, inplace=True)
full_data_stage_2.rename(columns={"WTeamID": "TeamID0", "LTeamID": "TeamID1", "WScore": "Score0", "LScore": "Score1", "WSeed": "Seed0", "WRegion": "Region0", "LSeed": "Seed1", "LRegion": "Region1"}, inplace=True)


#print(full_data.head())

#print(reverse_data.head())
#print(full_data.head())

# Gather the dataframes we want to merge
frames = [full_data, reverse_data]
# Concatenate them together
features_label = pd.concat(frames, sort=False)

# We have to shuffle the data otherwise it'll overfit to a certain thing, double shuffle just for the heck of it
features_label = features_label.sample(frac=1).reset_index(drop=True)
features_label = features_label.sample(frac=1).reset_index(drop=True)


features_label['SeedDiff'] = features_label.Seed0 - features_label.Seed1
features_label['ScoreDiff'] = features_label.Score0 - features_label.Score1

full_data_stage_2['SeedDiff'] = full_data_stage_2.Seed0 - full_data_stage_2.Seed1
full_data_stage_2['ScoreDiff'] = full_data_stage_2.Score0 - full_data_stage_2.Score1

X_stage_2 = full_data_stage_2.copy()
X_stage_2.drop(labels=['Season', 'TeamID0', 'TeamID1', 'Score1', 'Seed1'], inplace=True, axis=1)
#print(features_label.head())
X = features_label.copy()
#print(X.dtypes)
y = X['Result']
#print(y)
X.drop(labels=['Season', 'TeamID0', 'TeamID1', 'Result', 'Score1', 'Seed1'], inplace=True, axis=1)
features_label.drop(labels=['Result'], inplace=True, axis=1)

y = np.array(y).reshape(-1, 1)
#print(y.shape)

y = np.asanyarray(y)
#print(y.shape)

#print(y_train.shape)
#print('Create data matrix')
#data_dmatrix = xgb.DMatrix(data=X,label=y)

model = Sequential()
model.add(Dense(256, input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.35))

model.add(Dense(1)) # Output layer
model.add(Activation("sigmoid"))

print('Create the classifier')
model.compile(loss="binary_crossentropy",
                            optimizer="adam",
                            metrics=['accuracy'])

print('Fit model')

model.fit(X, y, batch_size = 32, epochs=15, validation_split=0.15, shuffle=True)

#xg_reg = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 38, alpha = 10, n_estimators = 100)


### TODO ###
#xg_reg.fit(X, y.ravel())

print('Model has been fit')

# We're going to have to make the data processable for predictions. Score is going to be average score of the season, seed will be seed of season, 
# ordinal ranking will be ranking & system of season, Etc.

sample_submission = pd.read_csv(base_directory + '/SampleSubmissionStage1.csv')
final_submission = pd.read_csv(base_directory + '/SampleSubmissionStage2.csv')
#print(sample_submission.head())
#print(len(sample_submission))

def get_season_t0_t1(ID):
    """Return a tuple with ints `season`, `team0` and `team1`."""
    return (int(x) for x in ID.split('_'))

#print(features_label.shape)
#print(features_label.dtypes)
print(X.shape)
stage_1_submission = np.zeros(shape=(len(sample_submission), 6))
#print(stage_1_submission)

# We're going to have to make the data processable for predictions. Score is going to be average score of the season, seed will be seed of season, 
# oridnal ranking will be ranking & system of season, Etc.

# First let's grab ALL of the team's information for the season and whatnot
#print(features_label.dtypes)
features_label = pd.DataFrame(features_label)
team_average_season_score = features_label.groupby(['TeamID0', 'Season'], as_index=False).Score0.mean()
#print(team_average_season_score.dtypes)
#team_season_sum_massey = team_season_sum_massey_lose.copy()
#team_season_sum_massey = team_season_sum_massey.rename(columns={'LTeamID': 'TeamID0'})
team_average_season_seed = features_label.groupby(['TeamID0', 'Season', 'Region0'], as_index=False).Seed0.mean()
#print(team_average_season_seed.head())

#print(X.dtypes)
print('Create the final dataframe')
for ii, row in sample_submission.iterrows():
 #   print(row.ID)
    season, t0, t1 = get_season_t0_t1(row.ID)
    average_score_row_0 = team_average_season_score[(team_average_season_score.TeamID0 == t0) & (team_average_season_score.Season == season)]
   # print(average_score_row_0)
    average_seed_row_0 = team_average_season_seed[(team_average_season_seed.TeamID0 == t0) & (team_average_season_seed.Season == season)]
  #  print(average_seed_row_0)    
  #  sum_massey_row_0 = team_season_sum_massey[(team_season_sum_massey.TeamID0 == t0) & (team_season_sum_massey.Season == season)]
    #print(sum_massey_row_0)
    
    average_score_row_1 = team_average_season_score[(team_average_season_score.TeamID0 == t1) & (team_average_season_score.Season == season)]
    #print(average_score_row_1)
    average_seed_row_1 = team_average_season_seed[(team_average_season_seed.TeamID0 == t1) & (team_average_season_seed.Season == season)]
    #print(average_seed_row_1)    
  #  sum_massey_row_1 = team_season_sum_massey[(team_season_sum_massey.TeamID0 == t1) & (team_season_sum_massey.Season == season)]
   # print(sum_massey_row_1)
  #  print(average_seed_row_0.iloc[0]['Seed0'])
  #  print(X.dtypes)
    stage_1_submission[ii, 0] = average_score_row_0['Score0']
  #  stage_1_submission[ii, 1] = average_score_row_1['Score0']
    stage_1_submission[ii, 1] = average_seed_row_0['Seed0']
    stage_1_submission[ii, 2] = average_seed_row_0['Region0']
  #  stage_1_submission[ii, 4] = average_seed_row_1['Seed0']
    stage_1_submission[ii, 3] = average_seed_row_1['Region0']
    stage_1_submission[ii, 4] = average_seed_row_0.iloc[0]['Seed0'] - average_seed_row_1.iloc[0]['Seed0']
    stage_1_submission[ii, 5] = average_score_row_0.iloc[0]['Score0'] - average_score_row_1.iloc[0]['Score0']


print("Created submission dataframe")
stage_1_submission = pd.DataFrame(stage_1_submission)
stage_1_submission = stage_1_submission.apply(pd.to_numeric,downcast='integer')
stage_1_submission = stage_1_submission.apply(pd.to_numeric,downcast='unsigned')
stage_1_submission.columns = X.columns
stage_1_submission['Score0'].replace(to_replace=0, method='ffill')
stage_1_submission['Seed0'].replace(to_replace=0, method='ffill')
stage_1_submission['Region0'].replace(to_replace=0, method='ffill')
stage_1_submission['Region1'].replace(to_replace=0, method='ffill')
stage_1_submission['SeedDiff'].replace(to_replace=0, method='ffill')
stage_1_submission['ScoreDiff'].replace(to_replace=0, method='ffill')
#stage_1_submission.drop(labels=['Score1', 'Seed1'], inplace=True, axis=1)
print(stage_1_submission.columns)
#print(X.dtypes)


print("Create stage 2 info")
stage_2_submission = np.zeros(shape=(len(final_submission), 6))
#print(stage_1_submission)

# We're going to have to make the data processable for predictions. Score is going to be average score of the season, seed will be seed of season, 
# oridnal ranking will be ranking & system of season, Etc.

# First let's grab ALL of the team's information for the this season
#print(features_label.dtypes)
full_data_stage_2 = pd.DataFrame(full_data_stage_2)
print(full_data_stage_2.head())
team_average_season_score_0 = full_data_stage_2.groupby(['TeamID0', 'Season'], as_index=False).Score0.mean()
team_average_season_score_1 = full_data_stage_2.groupby(['TeamID1', 'Season'], as_index=False).Score1.mean()

#print(team_average_season_score.dtypes)
#team_season_sum_massey = team_season_sum_massey_lose.copy()
#team_season_sum_massey = team_season_sum_massey.rename(columns={'LTeamID': 'TeamID0'})
team_average_season_seed_0 = full_data_stage_2.groupby(['TeamID0', 'Season', 'Region0'], as_index=False).Seed0.mean()
team_average_season_seed_1 = full_data_stage_2.groupby(['TeamID1', 'Season', 'Region1'], as_index=False).Seed1.mean()
print(team_average_season_seed_1.head())
#print(team_average_season_seed.head())
print(stage_2_submission.shape)
#print(X.dtypes)
print('Create the final dataframe')
for ii, row in final_submission.iterrows():

    season, t0, t1 = get_season_t0_t1(row.ID)
    average_score_row_0 = team_average_season_score_0[(team_average_season_score_0.TeamID0 == t0)].tail(1)
    average_seed_row_0 = team_average_season_seed[(team_average_season_seed.TeamID0 == t0)].tail(1)

    
    average_score_row_1 = team_average_season_score_1[(team_average_season_score_1.TeamID1 == t1)].tail(1)
    average_seed_row_1 = team_average_season_seed_1[(team_average_season_seed_1.TeamID1 == t1)].tail(1)
    if(len(average_score_row_0.index) == 0):
        average_score_row_0 = pd.DataFrame([65.0], columns=['Score0',])
    if(len(average_seed_row_0.index) == 0):
        average_seed_row_0 = pd.DataFrame([[8, 1]], columns=['Seed0', 'Region0'])
    if(len(average_score_row_1.index) == 0):
        average_score_row_1 = pd.DataFrame([65.0], columns=['Score1'])
    if(len(average_seed_row_1.index) == 0):
        average_seed_row_1 = pd.DataFrame([[8, 1]], columns=['Seed1', 'Region1'])
    stage_2_submission[ii, 0] = average_score_row_0['Score0']
    stage_2_submission[ii, 1] = average_seed_row_0['Seed0']
    stage_2_submission[ii, 2] = average_seed_row_0['Region0']
    stage_2_submission[ii, 3] = average_seed_row_1['Region1']
    stage_2_submission[ii, 4] = average_seed_row_0.iloc[0]['Seed0'] - average_seed_row_1.iloc[0]['Seed1']
    stage_2_submission[ii, 5] = average_score_row_0.iloc[0]['Score0'] - average_score_row_1.iloc[0]['Score1']


print("Created submission dataframe")
stage_2_submission = pd.DataFrame(stage_2_submission)
stage_2_submission = stage_2_submission.apply(pd.to_numeric,downcast='integer')
stage_2_submission = stage_2_submission.apply(pd.to_numeric,downcast='unsigned')
stage_2_submission.columns = X.columns
stage_2_submission['Score0'].replace(to_replace=0, method='ffill')
stage_2_submission['Seed0'].replace(to_replace=0, method='ffill')
stage_2_submission['Region0'].replace(to_replace=0, method='ffill')
stage_2_submission['Region1'].replace(to_replace=0, method='ffill')
stage_2_submission['SeedDiff'].replace(to_replace=0, method='ffill')
stage_2_submission['ScoreDiff'].replace(to_replace=0, method='ffill')
#stage_1_submission.drop(labels=['Score1', 'Seed1'], inplace=True, axis=1)
print(stage_2_submission.columns)

print("Save model")

model.save("NCAA2019-keras.model")

print("Create predictions")

model =  tf.keras.models.load_model("NCAA2019-keras.model")

predictions = model.predict(stage_2_submission)
#print(y_test)
predictions = pd.DataFrame(predictions)
predictions.columns = ['Pred']
#print(predictions.columns)

#print(predictions.shape)
sample_submission = pd.DataFrame(sample_submission)
#print(sample_submission.columns)
#predictions = pd.Series(predictions,name="Pred")
print(predictions.head())
#predictions['Pred'] = predictions['Pred']
predictions['Pred'] = predictions['Pred'].where(predictions['Pred'] <= .75, .75)
print(predictions.head())

predictions = pd.DataFrame(predictions['Pred'].where(predictions['Pred'] >= .35, .35))
print(predictions.head())
print(predictions.head())
print(predictions.Pred)
#print(sample_submission)
print("We have made the output correct")
print(predictions.head())
final_submission['Pred'] = predictions['Pred']
print(sample_submission)
print(final_submission)
final_submission.to_csv("stage_2_submission.csv",index=False)