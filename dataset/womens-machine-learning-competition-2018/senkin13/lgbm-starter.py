import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split 
import lightgbm as lgb

tourneycompactresults = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
tourneyseeds = pd.read_csv('../input/WNCAATourneySeeds.csv')
sub = pd.read_csv('../input/WSampleSubmissionStage1.csv')

def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
    
tourneyseeds['seed_int'] = tourneyseeds.Seed.apply(seed_to_int)
tourneyseeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
tourneycompactresults.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
winseeds = tourneyseeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
lossseeds = tourneyseeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
dummy = pd.merge(left=tourneycompactresults, right=winseeds, how='left', on=['Season', 'WTeamID'])
concat = pd.merge(left=dummy, right=lossseeds, on=['Season', 'LTeamID'])
concat['SeedDiff'] = concat.WSeed - concat.LSeed
wins = pd.DataFrame()
wins['SeedDiff'] = concat['SeedDiff']
wins['Result'] = 1
losses = pd.DataFrame()
losses['SeedDiff'] = -concat['SeedDiff']
losses['Result'] = 0
df_predictions = pd.concat((wins, losses))

X = df_predictions.SeedDiff.values.reshape(-1,1)
y = df_predictions.Result.values

n_test_games = len(sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))
    
X_test = np.zeros(shape=(n_test_games, 1))

for ii, row in sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = tourneyseeds[(tourneyseeds.TeamID == t1) & (tourneyseeds.Season == year)].seed_int.values[0]
    t2_seed = tourneyseeds[(tourneyseeds.TeamID == t2) & (tourneyseeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)

params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": 0.1,
          "num_leaves": 31,
          "max_bin": 256,
          "feature_fraction": 0.8,
          "verbosity": 0,
          "min_data_in_leaf": 10,
          "min_child_samples": 10,
          "subsample": 0.8
          }
          
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
bst = lgb.train(params, dtrain, 1000, valid_sets=dvalid, verbose_eval=50,
    early_stopping_rounds=50)

test_pred = bst.predict(
    X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)
    
sub.Pred = test_pred   
sub.to_csv('sub.csv', index=False) 