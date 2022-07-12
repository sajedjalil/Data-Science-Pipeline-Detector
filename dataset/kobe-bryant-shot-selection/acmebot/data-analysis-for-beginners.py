# # Let's perform analysis!
# 
# Hello, I'm Dixhom. Here I talk about how to preform feature engineering, delete unwanted variables, build a model and make submission data! So this is a tutorial for data science beginners. So let's get the ball rolling.
# 
# (This is for a kaggle competition 'Kobe Bryant Shot Selection' (https://www.kaggle.com/c/kobe-bryant-shot-selection))

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
# import data
filename= "../input/data.csv"
raw = pd.read_csv(filename)
# # Feature engineering
# Now let's start feature engineering. There are many features which should be modified or deleted for brevity. Let's take a look into variables.
# 
# First, let's take a look at all the variables.

raw.head()
# ## Dropping nans
# We are gonna make a variable without `nan` for our exploratory analysis. 


nona =  raw[pd.notnull(raw['shot_made_flag'])]
# ## loc_x, loc_y, lat and lon
# What do these mean? From their names, these sound like **location_x, location_y, latitude and longitude**. Let's confirm this assumption. 

alpha = 0.02
plt.figure(figsize=(10,10))

# loc_x and loc_y
plt.subplot(121)
plt.scatter(nona.loc_x, nona.loc_y, color='blue', alpha=alpha)
plt.title('loc_x and loc_y')

# lat and lon
plt.subplot(122)
plt.scatter(nona.lon, nona.lat, color='green', alpha=alpha)
plt.title('lat and lon')
# These plot are shaped like basket ball courts. So loc_x, loc_y, lat and lon seem to mean the position from which the ball was tossed. However, since the region under the net is half-circle-shaped, it would be more suitable to transform the variable into **polar coodinate**.
raw['dist'] = np.sqrt(raw['loc_x']**2 + raw['loc_y']**2)

loc_x_zero = raw['loc_x'] == 0
raw['angle'] = np.array([0]*len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi / 2 
# Since some of loc_x values cause an error by zero-division, we set just `np.pi / 2` to the corresponding rows.
# 
# ## minutes_remaining and seconds_remaining
# `minutes_remaining` and `seconds_remaining` seem to be a pair, so let's combine them together.
raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']
# ## action_type, combined_shot_type, shot_type
# These represents how the player shot a ball.
print(nona.action_type.unique())
print(nona.combined_shot_type.unique())
print(nona.shot_type.unique())
# ## Season
# `Season` looks like consisting of two parts.
nona['season'].unique()
# `Season` seems to be composed of two parts: season year and season ID. Here we only need season ID. Let's modify the data.
raw['season'] = raw['season'].apply(lambda x: int(x.split('-')[1]) )
raw['season'].unique()
# ## team_id and team_name
# These contain the same one value for each. Seem useless. 
print(nona['team_id'].unique())
print(nona['team_name'].unique())
# ## opponent , matchup
# These are basically the same information. 
pd.DataFrame({'matchup':nona.matchup, 'opponent':nona.opponent})
# Only opponent is needed.
# ## Shot distance
# We already defined this.
plt.figure(figsize=(5,5))

plt.scatter(raw.dist, raw.shot_distance, color='blue')
plt.title('dist and shot_distance')
# `shot_distance` is proportional to `dist` and this won't be necessary.
# 
# ## shot_zone_area, shot_zone_basic, shot_zone_range
# These sound like some regions on the court, so let's visualize it.
import matplotlib.cm as cm
plt.figure(figsize=(20,10))

def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = nona.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)

# shot_zone_area
plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')

# shot_zone_basic
plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')

# shot_zone_range
plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')
# As we thought, these represent regions on the court. However, these regions can be separated by `dist` and `angle`. So we don't need these.
# ## dropping unneeded variables
# Let's drop unnecessary variables.
drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw = raw.drop(drop, 1)
# ## make dummy variables
# We are going to use randomForest classifier for building our models but this doesn't accept string variables like 'action_type'. So we are going to make dummy variables for those.
# turn categorical variables into dummy variables
categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
    raw = raw.drop(var, 1)
# ## separating data for training and submission
# Now let's separate data.
df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag', 1)
# We are separating `df` further into explanatory and response variables.
# separate df into explanatory and response variables
train = df.drop('shot_made_flag', 1)
train_y = df['shot_made_flag']
# ## logloss
# Submissions are evaluated on the log loss. We are going to use it for evaluating our model.
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
# # Building a model
# Now it's time to build a model. We use randomForest classifier and k-fold cross validation for testing our model.
# We are going to...
# 
# 1. pick a `n` from `n_range` for the number of estimators in randomForestClassifier.
# 1. divide the training data into 10 pieces
# 2. pick 9 of them for building a model and use the remaining 1 for testing a model
# 3. repeat the same process for the other 9 pieces.
# 4. calculate score for each and take an average of them
# 5. pick the next `n` and do the process again
# 6. find the `n` which gave the best score among `n_range`
# 7. repeat the same process with the tree depth parameter.
# 
# You can change the value of `np.logspace` for searching optimum value in broader area.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import time


# find the best n_estimators for RandomForestClassifier
print('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(0,2,num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()
    
    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n
        
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2-t1))
print(best_n, min_score)


# find best max_depth for RandomForestClassifier
print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []
range_m = np.logspace(0,2,num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()
    
    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        #rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m
    
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2-t1))
print(best_m, min_score)

# # Visualizing parameters for randomForest
# By visualizing the parameters, we can check if the chosen parameter is really the best.
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(range_n, scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')

plt.subplot(122)
plt.plot(range_m, scores_m)
plt.ylabel('score')
plt.xlabel('max depth')
# # Building a final model
# Let's use the parameters we just got for the final model and prediction.
model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train, train_y)
pred = model.predict_proba(submission)
# # Making submission data
# Predicted shot_made_flag is written to a csv file.
sub = pd.read_csv("../input/sample_submission.csv")
sub['shot_made_flag'] = pred
sub.to_csv("real_submission.csv", index=False)