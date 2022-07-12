# First of all I have to thank ArjoonnSharma
# I have learned a lot from the script Preliminary Exploration
# this script is largely based on his/her work, and I hope it's OK

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../input/data.csv")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
def test_it(data_test):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=5)  # A simple classifier
    return cross_val_score(clf, data_test.drop('shot_made_flag', 1), data_test.shot_made_flag,
                           scoring='roc_auc', cv=10
                          )
# define the accuracy plotting function
def get_acc(df, against):
    ct = pd.crosstab(df.shot_made_flag, df[against]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(against)
    plt.ylabel('% shots made')
# define the sort & enumeration function
def sort_encode(df, field):
    ct = pd.crosstab(df.shot_made_flag, df[field]).apply(lambda x:x/x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    df[new_field] = df[field].map(new_map)
    return new_field
auc_list = {}
# action_type
action_map = {action: i for i, action in enumerate(data.action_type.unique())}
data['action_type_enumerated'] = data.action_type.map(action_map)
new_field = sort_encode(data, 'action_type_enumerated')

data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# combined_shot_type
action_map = {action: i for i, action in enumerate(data.combined_shot_type.unique())}
data['combined_shot_type_enumerated'] = data.combined_shot_type.map(action_map) 
new_field = sort_encode(data, 'combined_shot_type_enumerated')

data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# minutes_remaining
new_field = 'minutes_remaining'
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# seconds_remaining
new_field = 'seconds_remaining'
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# combine minutes and seconds remaining
data['time_remaining'] = data.minutes_remaining*60 + data.seconds_remaining
data['time_remaining_enumerated'] = 99
data.loc[data.time_remaining<10, 'time_remaining_enumerated'] = 0
data.loc[data.time_remaining>=10, 'time_remaining_enumerated'] = 1

new_field = 'time_remaining_enumerated'
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# period
new_field = 'period'
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# playoffs
new_field = 'playoffs'
data_test = data[['playoffs', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# season
data['season_start_year'] = data.season.str.split('-').str[0]
data['season_start_year'] = data['season_start_year'].astype(int)

new_field = 'season_start_year'
data_test = data[['season_start_year', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_distance
new_field = 'shot_distance'
data_test = data[['shot_distance', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_type (2 or 3 points)
action_map = {action: i for i, action in enumerate(data.shot_type.unique())}
data['shot_type_enumerated'] = data.shot_type.map(action_map) 

new_field = 'shot_type_enumerated'
data_test = data[['shot_type_enumerated', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_zone_area
action_map = {action: i for i, action in enumerate(data.shot_zone_area.unique())}
data['shot_zone_area_enumerated'] = data.shot_zone_area.map(action_map) 

new_field = sort_encode(data, 'shot_zone_area_enumerated')

data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_zone_basic
# there is a lot of variance for this feature
action_map = {action: i for i, action in enumerate(data.shot_zone_basic.unique())}
data['shot_zone_basic_enumerated'] = data.shot_zone_basic.map(action_map) 

new_field = sort_encode(data, 'shot_zone_basic_enumerated')

data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_zone_range
# there is a lot of variance for this feature
action_map = {action: i for i, action in enumerate(data.shot_zone_range.unique())}
data['shot_zone_range_enumerated'] = data.shot_zone_range.map(action_map) 

new_field = sort_encode(data, 'shot_zone_range_enumerated')

data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# opponent
# there is not much variance here
action_map = {action: i for i, action in enumerate(data.opponent.unique())}
data['opponent_enumerated'] = data.opponent.map(action_map) 

new_field = sort_encode(data, 'opponent_enumerated')

data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# matchup
# Kobe's home game performance is slightly better than away game, but not much
# create a new feature consisting of 0 (away game) and 1 (home game)
data['home_or_away'] = 99
data.loc[data.matchup.str.find('@')==-1, 'home_or_away'] = 1
data.loc[data.matchup.str.find('vs')==-1, 'home_or_away'] = 0

new_field = 'home_or_away'
data_test = data[['home_or_away', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
auc_list
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_distance'
data_test = data[[f1, f2, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_area_enumerated_sort_enumerated'
data_test = data[[f1, f2, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_basic_enumerated_sort_enumerated'
data_test = data[[f1, f2, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_range_enumerated_sort_enumerated'
data_test = data[[f1, f2, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_range_enumerated_sort_enumerated'
f3 = 'opponent_enumerated_sort_enumerated'
data_test = data[[f1, f2, f3, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_range_enumerated_sort_enumerated'
f3 = 'home_or_away'
data_test = data[[f1, f2, f3, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_range_enumerated_sort_enumerated'
f3, f4= 'opponent_enumerated_sort_enumerated', 'home_or_away'
data_test = data[[f1, f2, f3, f4, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=7, random_state=2016) # a more powerful classifier

f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_range_enumerated_sort_enumerated'
f3 = 'home_or_away'
train = data.loc[~data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_made_flag']]
test = data.loc[data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_id']]

# Impute
mode = test.action_type_enumerated_sort_enumerated.mode()[0]
test.action_type_enumerated_sort_enumerated.fillna(mode, inplace=True)

# Train and predict
clf.fit(train.drop('shot_made_flag', 1), train.shot_made_flag)
predictions = clf.predict_proba(test.drop('shot_id', 1))

# convert to CSV
submission = pd.DataFrame({'shot_id': test.shot_id,
                           'shot_made_flag': predictions[:, 1]})
submission[['shot_id', 'shot_made_flag']].to_csv('submission.csv', index=False)
clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=7, random_state=2016) # a more powerful classifier

f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_distance'
f3 = 'home_or_away'
train = data.loc[~data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_made_flag']]
test = data.loc[data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_id']]

# Impute
mode = test.action_type_enumerated_sort_enumerated.mode()[0]
test.action_type_enumerated_sort_enumerated.fillna(mode, inplace=True)

# Train and predict
clf.fit(train.drop('shot_made_flag', 1), train.shot_made_flag)
predictions = clf.predict_proba(test.drop('shot_id', 1))

# convert to CSV
submission = pd.DataFrame({'shot_id': test.shot_id,
                           'shot_made_flag': predictions[:, 1]})
submission[['shot_id', 'shot_made_flag']].to_csv('submission.csv', index=False)
