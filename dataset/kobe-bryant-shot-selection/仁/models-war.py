# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Reading data
###
data = pd.read_csv('../input/data.csv')
data.set_index('shot_id', inplace=True)
# The following should be explicitly made categorical as they are encoded using integers
data["action_type"] = data["action_type"].astype('object')
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["game_event_id"] = data["game_event_id"].astype('category')
data["game_id"] = data["game_id"].astype('category')
data["period"] = data["period"].astype('object')
data["playoffs"] = data["playoffs"].astype('category')
data["season"] = data["season"].astype('category')
data["shot_made_flag"] = data["shot_made_flag"].astype('category')
data["shot_type"] = data["shot_type"].astype('category')
data["team_id"] = data["team_id"].astype('category')
###
###

## data cleaning
##
unknown_mask = data['shot_made_flag'].isnull()
data_cl = data.copy()
target = data_cl['shot_made_flag'].copy()
data_cl.drop('team_id', inplace=True, axis=1) #only 1 category
data_cl.drop('lat', inplace=True, axis=1) # correlated with loc_x
data_cl.drop('lon', inplace=True, axis=1) # correlated with loc_y
data_cl.drop('game_id', inplace=True, axis=1) # should not be dependent on game id, furthermore it's contained in opponent/match
data_cl.drop('game_event_id', inplace=True, axis=1) # independent, unique for every shots in a game
data_cl.drop('team_name', inplace=True, axis=1) # always LA Lakers
data_cl.drop('shot_made_flag', inplace=True, axis=1) # target variables
##
##

##### Feature Engineering
#####
##### time remaining
# new features
data_cl['seconds_from_period_end'] = 60 * data_cl['minutes_remaining'] + data_cl['seconds_remaining']
data_cl['last_5_sec_in_period'] = data_cl['seconds_from_period_end'] < 5
# drop redundant features
data_cl.drop('minutes_remaining', axis=1, inplace=True)
data_cl.drop('seconds_remaining', axis=1, inplace=True)
data_cl.drop('seconds_from_period_end', axis=1, inplace=True)
#####
## Matchup -- (away/home)
data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')
data_cl.drop('matchup', axis=1, inplace=True)
# Game date -- extract year and month
data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])
data_cl['game_year'] = data_cl['game_date'].dt.year
data_cl['game_month'] = data_cl['game_date'].dt.month
data_cl.drop('game_date', axis=1, inplace=True)
# Loc_x, and loc_y binning
data_cl['loc_x'] = pd.cut(data_cl['loc_x'], 25)
data_cl['loc_y'] = pd.cut(data_cl['loc_y'], 25)
# Replace 20 least common action types with value 'Other'
rare_action_types = data_cl['action_type'].value_counts().sort_values().index.values[:20]
data_cl.loc[data_cl['action_type'].isin(rare_action_types), 'action_type'] = 'Other'
# One-hot encoding of categorical variables
categorial_cols = [
    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
    'game_month', 'opponent', 'loc_x', 'loc_y']
for cc in categorial_cols:
    dummies = pd.get_dummies(data_cl[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cl.drop(cc, axis=1, inplace=True)
    data_cl = data_cl.join(dummies)
######
######




# Train/validation split
# Separate dataset for validation
data_submit = data_cl[unknown_mask]
# Separate dataset for training
X = data_cl[~unknown_mask]
Y = target[~unknown_mask]





##### Feature selection
#####
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
# features with high variance
threshold = 0.90
vt = VarianceThreshold().fit(X)
feat_var_threshold = data_cl.columns[vt.variances_ > threshold * (1-threshold)]
# Top 20 features according to Random Forest
model = RandomForestClassifier()
model.fit(X, Y)
feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
feat_imp_20
# Top 20 features using chi^2 test
X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, Y)
feature_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': X_scored.scores_
    })
feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
feat_scored_20
# Recursive feature elimination using Logistic Regression
rfe = RFE(LogisticRegression(), 20)
rfe.fit(X, Y)
feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })
feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe_20
# Our final features
features = np.hstack([
        feat_var_threshold, 
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])
features = np.unique(features)
print('Final features set:\n')
for f in features:
    print("\t-{}".format(f))
data_cl = data_cl.ix[:, features]
data_submit = data_submit.ix[:, features]
X = X.ix[:, features]
print('Clean dataset shape: {}'.format(data_cl.shape))
print('Subbmitable dataset shape: {}'.format(data_submit.shape))
print('Train features shape: {}'.format(X.shape))
print('Target label shape: {}'. format(Y.shape))
######


############################# Algorithms ###################################
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier

# Set CV
seed = 7
processors=1
num_folds=3
num_instances=len(X)
scoring='neg_log_loss'
kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)

# BASIC MODELS
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))

# Evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
#     results.append(cv_results)
#     names.append(name)
#     print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))

# from sklearn.grid_search import GridSearchCV
# lr_grid = GridSearchCV(
#     estimator = LogisticRegression(random_state=seed),
#     param_grid = {
#         'penalty': ['l1', 'l2'],
#         'C': [0.001, 0.01, 1, 10, 100, 1000]
#     }, 
#     cv = kfold, 
#     scoring = scoring, 
#     n_jobs = processors)
model = LogisticRegression()
model.fit(X,Y)
preds = model.predict_proba(data_submit)
submission = pd.DataFrame()
submission["shot_id"] = data_submit.index
submission["shot_made_flag"]= preds[:,0]

submission.to_csv("sub.csv",index=False)


# # BAGGING
# cart = DecisionTreeClassifier()
# num_trees = 100
# model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
# print("Bagging: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

# # RANDOM FOREST
# num_trees = 100
# num_features = 10
# model = RandomForestClassifier(n_estimators=num_trees, max_features=num_features)
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
# print("RF: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

# # EXTRA TREE CLASSIFIER
# num_trees = 100
# num_features = 10
# model = ExtraTreesClassifier(n_estimators=num_trees, max_features=num_features)
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
# print("ExtraTree: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

# # ADABOOST CLASSIFIER
# model = AdaBoostClassifier(n_estimators=100, random_state=seed)
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
# print("Adaboost: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

# # GRADIENT BOOSTING CLASSIFIER
# model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
# print("GradientBoosting: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

# Parameter tuning


# lda_grid = GridSearchCV(
#     estimator = LinearDiscriminantAnalysis(),
#     param_grid = {
#         'solver': ['lsqr'],
#         'shrinkage': [0, 0.25, 0.5, 0.75, 1],
#         'n_components': [None, 2, 5, 10]
#     }, 
#     cv = kfold, 
#     scoring = scoring, 
#     n_jobs = processors)
# lda_grid.fit(X, Y)
# print("LDA gridsearch: {0:.3f}".format(lda_grid.best_score_))
# print(lda_grid.best_params_)

# rf_grid = GridSearchCV(
#     estimator = RandomForestClassifier(warm_start=True, random_state=seed),
#     param_grid = {
#         'n_estimators': [100, 200],
#         'criterion': ['gini', 'entropy'],
#         'max_features': [18, 20],
#         'max_depth': [8, 10],
#         'bootstrap': [True]
#     }, 
#     cv = kfold, 
#     scoring = scoring, 
#     n_jobs = processors)
# rf_grid.fit(X, Y)
# print("RF gridsearch: {0:.3f}".format(rf_grid.best_score_))
# print(rf_grid.best_params_)


# ada_grid = GridSearchCV(
#     estimator = AdaBoostClassifier(random_state=seed),
#     param_grid = {
#         'algorithm': ['SAMME', 'SAMME.R'],
#         'n_estimators': [10, 25, 50],
#         'learning_rate': [1e-3, 1e-2, 1e-1]
#     }, 
#     cv = kfold, 
#     scoring = scoring, 
#     n_jobs = processors)
# ada_grid.fit(X, Y)
# print("ADA gridsearch: {0:.3f}".format(ada_grid.best_score_))
# print(ada_grid.best_params_)


# gbm_grid = GridSearchCV(
#     estimator = GradientBoostingClassifier(warm_start=True, random_state=seed),
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [2, 3, 4],
#         'max_features': [10, 15, 20],
#         'learning_rate': [1e-1, 1]
#     }, 
#     cv = kfold, 
#     scoring = scoring, 
#     n_jobs = processors)
# gbm_grid.fit(X, Y)
# print("GBM gridsearch: {0:.3f}".format(gbm_grid.best_score_))
# print(gbm_grid.best_params_)

# # Create sub models
# estimators = []

# estimators.append(('lr', LogisticRegression(penalty='l2', C=1)))
# estimators.append(('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, max_features=15, warm_start=True, random_state=seed)))
# estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, n_estimators=200, max_features=20, criterion='entropy', random_state=seed)))
# estimators.append(('ada', AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1e-2, n_estimators=10, random_state=seed)))

# # create the ensemble model
# ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1])

# results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring,n_jobs=processors)
# print("Ensemble: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))


# model = ensemble

# model.fit(X, Y)
# preds = model.predict_proba(data_submit)

