# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as m
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/data.csv')

df.head()
shot_id_rows = pd.DataFrame({'shot_id' : df[df['shot_made_flag'].isnull()].reset_index()['shot_id']})
'''
df.groupby('action_type')['shot_made_flag'].agg({"returns": [np.sum, np.size, np.mean]})
de = df[df['shot_made_flag'].isnull()]
de.groupby('action_type').count()
'''
cols_to_preserve = [ 'period', 'season', 'shot_distance', 'alleyoop', 'bank', 'cutting',
       'driving', 'dunk', 'fadeaway', 'fingerroll', 'floating',
       'followup', 'hook', 'jump', 'layup', 'pullup', 'putback',
       'reverse', 'running', 'slam', 'stepback', 'tip', 'turnaround',
       'seconds', 'hurry_regular_shot', 'home',
       'action_type#AlleyOop Dunk Shot',
       'action_type#AlleyOop Layup shot', 'action_type#Driving Dunk Shot',
       'action_type#Driving FingerRoll Layup Shot',
       'action_type#Driving FingerRoll Shot',
       'action_type#Driving Floating Bank Jump Shot',
       'action_type#Driving Floating Jump Shot',
       'action_type#Driving Hook Shot', 'action_type#Driving Jump shot',
       'action_type#Driving Layup Shot',
       'action_type#Driving Reverse Layup Shot',
       'action_type#Driving Slam Dunk Shot', 'action_type#Dunk Shot',
       'action_type#Fadeaway Bank shot', 'action_type#Fadeaway Jump Shot',
       'action_type#FingerRoll Layup Shot', 'action_type#FingerRoll Shot',
       'action_type#Floating Jump shot', 'action_type#FollowUp Dunk Shot',
       'action_type#Hook Bank Shot', 'action_type#Hook Shot',
       'action_type#Jump Bank Shot', 'action_type#Jump Hook Shot',
       'action_type#Jump Shot', 'action_type#Layup Shot',
       'action_type#Pullup Bank shot', 'action_type#Pullup Jump shot',
       'action_type#Putback Layup Shot', 'action_type#Reverse Dunk Shot',
       'action_type#Reverse Layup Shot', 'action_type#Running Bank shot',
       'action_type#Running Dunk Shot',
       'action_type#Running FingerRoll Layup Shot',
       'action_type#Running Hook Shot', 'action_type#Running Jump Shot',
       'action_type#Running Layup Shot',
       'action_type#Running Reverse Layup Shot',
       'action_type#Running Tip Shot', 'action_type#Slam Dunk Shot',
       'action_type#StepBack Jump shot', 'action_type#Tip Layup Shot',
       'action_type#Tip Shot', 'action_type#Turnaround Bank shot',
       'action_type#Turnaround Fadeaway shot',
       'action_type#Turnaround FingerRoll Shot',
       'action_type#Turnaround Hook Shot',
       'action_type#Turnaround Jump Shot',
       'shot_zone_basic#Above the Break 3', 'shot_zone_basic#Backcourt',
       'shot_zone_basic#In The Paint (Non-RA)',
       'shot_zone_basic#Left Corner 3', 'shot_zone_basic#Mid-Range',
       'shot_zone_basic#Restricted Area',
       'shot_zone_basic#Right Corner 3', 'angle_bin#0', 'angle_bin#1',
       'angle_bin#2', 'angle_bin#3', 'angle_bin#4', 'angle_bin#5',
       'angle_bin#6', 'pt_class#0', 'pt_class#1', 'pt_class#2', 'shot_made_flag']
       
columns_to_drop = ['combined_shot_type', 'game_id', 'game_event_id', 'lat', 'lon', 'loc_x', 'loc_y',
                    'minutes_remaining', 'seconds_remaining','shot_zone_range', 'shot_zone_area',  
                    'angle', 'playoffs', 'team_id', 'team_name', 'game_date', 'matchup',  'shot_type', 'shot_id']
                    
categorial_cols = ['action_type', 'shot_zone_basic', 'angle_bin', 'pt_class', 'opponent']

#nisu bitna velika i mala slova

df['action_type'] = df.action_type.apply(lambda x: x.replace('-', ''))
df['action_type'] = df.action_type.apply(lambda x: x.replace('Follow Up', 'FollowUp'))
df['action_type'] = df.action_type.apply(lambda x: x.replace('Finger Roll','FingerRoll'))
df['action_type'] = df.action_type.apply(lambda x: x.replace('Alley Oop','AlleyOop'))
df['action_type'] = df.action_type.apply(lambda x: x.replace('Step Back','StepBack'))

df.loc[df['action_type']== 'Running Slam Dunk Shot', 'action_type'] = 'Slam Dunk Shot'
df.loc[df['action_type']== 'Driving Floating Bank Jump Shot', 'action_type'] = 'Jump Bank Shot'

#za svaki red iz action_type napravi se niz ponavljanja svakog moguceg tokena

cv = CountVectorizer(max_features=50, stop_words=['shot'])
shot_features = cv.fit_transform(df['action_type']).toarray()
shot_features = pd.DataFrame(shot_features, columns=cv.get_feature_names())

df = pd.concat([df,shot_features], axis="columns")

df['seconds'] = df['minutes_remaining']*60 + df['seconds_remaining']

df['angle'] = df.apply(lambda row: 90 if row['loc_y']==0 else m.degrees(m.atan(row['loc_x']/abs(row['loc_y']))),axis=1)


df['angle_bin'] = pd.cut(df.angle, 5, labels=range(5))
df['angle_bin'] = df.angle_bin.astype(int)
df.loc[df['shot_zone_basic'] == 'Restricted Area','angle_bin'] = 5
df.loc[df['shot_distance'] > 30, 'angle_bin'] = 6


df['hurry_regular_shot'] = False
df.loc[(df['seconds'] <=2) & (df['shot_distance'] < 30), 'hurry_regular_shot'] = True

df['season'] = df.season.str.split('-').str[0]
df['season'] = df.season.astype(int)

df['home'] = ~df.matchup.str.contains('@')

# 3PT above 30ft
df['pt_class'] = 2
# 3PT under equal 30ft
df.loc[(df.shot_type.str.contains('3')) & (df['shot_distance'] < 30) , 'pt_class'] = 1
# 2PT
df.loc[ df.shot_type.str.contains('2'), 'pt_class' ] = 0


df[df['action_type'] == 'Dunk Shot']
(df.loc[:, 'action_type'].value_counts() - df.loc[df['shot_made_flag'].isnull(), 'action_type'].value_counts()).sort_values()
#ukloni poslednja cetiri suta
#df['combined_shot_type'].apply(lambda x: x if len(x.split(' ')) == 2 else x + " Shot" )


actiontypes = dict(df.action_type.value_counts())
#df['action_type'] = df.apply(lambda row: row['action_type'] if actiontypes[row['action_type']] > 1\
                         # else row['combined_shot_type'], axis=1)

#df.groupby(['season', 'opponent'])['game_id'].unique()
#le = LabelEncoder()
#df['action_type'] = le.fit_transform(df['action_type'])


for cc in categorial_cols:
    dummies = pd.get_dummies(df[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    df.drop(cc, axis=1, inplace=True)
    df = df.join(dummies)
    
df.drop(columns_to_drop, axis=1, inplace=True)
pd.set_option('display.max_columns', 500)
df.columns.values

#df = df[cols_to_preserve]

  
nanRows = df.loc[df['shot_made_flag'].isnull()]
nanRows.index = range(len(nanRows)) 



fullRows = df.loc[~df['shot_made_flag'].isnull()]

X_test = nanRows.drop('shot_made_flag', axis=1)

Y = fullRows['shot_made_flag'].copy()
X = fullRows.drop('shot_made_flag', axis=1)


#X = fullRows[cols_to_preserve]

seed =50
processors=1
num_folds=5
num_instances=len(X)
scoring='neg_log_loss'


num_trees = 100
num_features = 20
# Prepare some basic models

seed = 50
num_folds = 5
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle = True)

models = []
models.append(('Logistic regression', LogisticRegression()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('Ada Boost', AdaBoostClassifier()))
models.append(('Gradient Boosting', GradientBoostingClassifier()))
models.append(('XGBoost', xgb.XGBClassifier()))

#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('K-NN', KNeighborsClassifier(n_neighbors=5)))
#models.append(('Decision Tree', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('BAGGED',BaggingClassifier()))
#models.append(('SVC', SVC(probability=True)))

# Evaluate each model in turn
results = []
names = []
stds = []
means =[]
for name, model in models:
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='neg_log_loss', n_jobs=1)
    print(cv_results)
    print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
    results.append(cv_results)
    names.append(name)
    stds.append(cv_results.std())
    means.append(abs(cv_results.mean()))

    
x_pos = np.arange(len(models))
# Build the plot
fig, ax = plt.subplots()
rects1 = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Absolute value of Logarithmic loss')
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_title('Machine learning models')
ax.yaxis.grid(True)
plt.subplots_adjust(top=1.1)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.04*height,
                '%.3f' % height,
                ha='center', va='bottom')

autolabel(rects1)
# Save the figure and show
#plt.tight_layout()
plt.xticks(rotation='vertical')
plt.show()
    
#model = xgb.XGBClassifier(seed=1, learning_rate=0.01, n_estimators=500, max_depth=7, subsample=0.8, colsample_bytree=0.6)

model = xgb.XGBClassifier(seed=1, learning_rate=0.01, n_estimators=500, max_depth=7, subsample=0.6, colsample_bytree=0.6)
model.fit(X, Y)

nanRows['shot_made_flag'] = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame()
submission['shot_id'] = shot_id_rows['shot_id']
submission['shot_made_flag'] = nanRows['shot_made_flag']
submission
submission.to_csv('submission.csv', index=False)



''' 

rf_grid = GridSearchCV(
    estimator = xgb.XGBClassifier(seed = 1),
    param_grid = {
        'n_estimators': [70, 80, 90, 150, 500, 600],
        'learning_rate' : [0.01, 0.05, 0.1],
        'n_estimators': [70, 100, 250, 500],
        'max_depth': [4, 7, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [100, 150, 500],
        'learning_rate' : [0.01, 0.05],
        'max_depth': [7, 10],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)

rf_grid.fit(X, Y)

print(rf_grid.best_score_)
print(rf_grid.best_params_)

rfe = RFECV(estimator=model, step=5, cv=kfold,
              scoring=scoring)
rfe.fit(X, Y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe_20

 
rfe = RFECV(estimator=GradientBoostingClassifier(), step=5, cv=kfold,
              scoring=scoring)
rfe.fit(X, Y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe_20

model = RandomForestClassifier()
model.fit(X, Y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(30).index
feat_imp_20

rf_grid = GridSearchCV(
    estimator = xgb.XGBClassifier( seed = seed,col_sample_bytree=1.0, gamma=0.5, max_depth=5, min_child_weight=5, subsample=1.0),
    param_grid = {
        'n_estimators': [70, 80, 90, 150, 500, 600],
        'learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.1]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)

rf_grid.fit(X, Y)

print(rf_grid.best_score_)
print(rf_grid.best_params_)
gbm_grid = GridSearchCV(
    estimator = GradientBoostingClassifier(),
    param_grid = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)

gbm_grid.fit(X, Y)

print(gbm_grid.best_score_)
print(gbm_grid.best_params_)
    
rf_grid = GridSearchCV(
    estimator = xgb.XGBClassifier(),
    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }, 
    cv = kfold, 
    scoring = scoring, 
    n_jobs = processors)

rf_grid.fit(X, Y)

print(rf_grid.best_score_)
print(rf_grid.best_params_)

model = RandomForestClassifier()
model.fit(X, Y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
feat_imp_20
rfe = RFECV(estimator=xgb.XGBClassifier(), step=1, cv=kfold,
              scoring=scoring)
rfe.fit(X, Y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe_20

df = df[['period', 'season', 'shot_distance', 'alleyoop', 'bank', 'driving',
       'dunk', 'fadeaway', 'fingerroll', 'floating', 'hook', 'jump',
       'pullup', 'reverse', 'running', 'slam', 'stepback', 'turnaround',
       'seconds', 'home', 'shot_zone_basic#Mid-Range', 'angle_bin#0',
       'angle_bin#2', 'angle_bin#3', 'angle_bin#4', 'pt_class#0']]

'''
#NO PSYCHOLOGY EFFECT
'''

SECOND_THRESHOLD_HOT_HAND = 240

df.loc[0, 'last_made'] = 0.5
df.loc[0, 'game_acc'] = 0.44
df.loc[0, 'asc'] = 0.5
df['streak'] = np.nan
df.loc[0, 'streak'] = 0
game_made = 0 
game_total = 0
streak = 0
for shot in range(1,df.shape[0]):

    # make sure the current shot and last shot were all in the same period of the same game
    sameGame   = df.loc[shot,'game_id'] == df.loc[shot-1,'game_id']
    samePeriod = df.loc[shot,'period']    == df.loc[shot-1,'period']

    if sameGame:
        nanLastShot = pd.isna(df.loc[shot - 1, 'shot_made_flag'])
        
        if(nanLastShot):
            df.loc[shot, 'last_made'] = 0.5
            game_made = game_made + 0.44
            game_total = game_total + 1
            df.loc[shot,'game_acc'] = game_made / game_total
            df.loc[shot,'asc'] = 0.5
            if df.loc[shot - 1, 'shot_distance'] >= 16:
                df.loc[shot, 'streak'] = streak / 2;
                streak = streak/2
            continue
        
        madeLastShot = df.loc[shot-1,'shot_made_flag'] == 1
        
        if(madeLastShot):
            game_made = game_made + 1
            if df.loc[shot - 1, 'shot_distance'] >= 16:
                if df.loc[shot - 1, 'streak'] > 0:
                     df.loc[shot, 'streak'] = streak + 1
                     streak = streak + 1
                else:
                     streak = 1
                     df.loc[shot, 'streak'] = 1    
        else:
            if df.loc[shot - 1, 'shot_distance'] >= 16:
                if df.loc[shot - 1, 'streak'] < 0:
                     df.loc[shot, 'streak'] = streak - 1
                     streak = streak - 1
                else:
                    streak = -1
                    df.loc[shot, 'streak'] = -1    
        
        game_total = game_total + 1
        
        df.loc[shot,'game_acc'] = game_made / game_total
        
        if(df.loc[shot,'game_acc'] > df.loc[shot - 1, 'game_acc'] > 0):
            df.loc[shot,'asc'] = 1
        else:
            df.loc[shot,'asc'] = 0
            
        timeDifferenceFromLastShot = df.loc[shot, 'seconds'] - df.loc[shot-1, 'seconds']

        if( (timeDifferenceFromLastShot > SECOND_THRESHOLD_HOT_HAND) or (not samePeriod) ):
            if madeLastShot:
                df.loc[shot, 'last_made'] = 1
            else:
                df.loc[shot, 'last_made'] = 0
        elif madeLastShot:
            df.loc[shot, 'last_made'] = 1
        else:
            df.loc[shot, 'last_made'] = 0
    else:
        df.loc[shot, 'last_made'] = 0.5
        game_made = 0 
        game_total = 0
        df.loc[shot, 'game_acc'] = 0.44
        df.loc[shot,'asc'] = 0.5
        streak = 0

df.loc[ df['game_acc'] >0.99]['shot_made_flag'].mean()
df.loc[df['last_made'] == 0.5].count()
df['streak']

df.loc[(df['streak']<= -3) & (df['shot_distance'] >=16 )]['shot_made_flag'].count()
df.loc[df['shot_distance'] >=16 ]['shot_made_flag'].mean()
'''