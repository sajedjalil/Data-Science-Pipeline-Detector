import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import xgboost as xgb

df=pd.read_csv('../input/data.csv')

for i in df.columns:
    if df[i].dtypes=='object':
        print(df[i].head())


#do some nice eda to inspect data

sns.countplot(df['shot_made_flag'])

#inspect numerical features

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.scatter(df.loc[df['shot_made_flag']==1, 'loc_x'], df.loc[df['shot_made_flag']==1, 'loc_y'], alpha=0.1, color='green')
plt.title('Shots made')
plt.ylim(-100, 900)
plt.subplot(1,2,2)
plt.scatter(df.loc[df['shot_made_flag']==0, 'loc_x'], df.loc[df['shot_made_flag']==0, 'loc_y'], alpha=0.1, color='red')
plt.show()

#Encode all object types and change dates to date time, action_type, combined_shot_type, !season, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range, team_name, game_date, matchup, opponent. Seems like we can do some tricks with opponent, matchup, and team_name columns to clean data more effectively.

#lat and lon are crap data, going to drop along with team_id and team_name


df['season']=df['season'].apply(lambda x: x[0:4])
df['time_remains']=df['minutes_remaining']*60 +df['seconds_remaining']
df['game_date']=pd.to_datetime(df['game_date'])
df['game_date_month']=df['game_date'].dt.month


df.drop(['lat', 'lon','team_id', 'team_name', 'game_id', 'game_event_id','seconds_remaining', 'minutes_remaining', 'combined_shot_type', 'game_date'], axis=1, inplace=True)


df['matchup']=df['matchup'].apply(lambda x: 1 if '@' in x else 0)
df['shot_type']=df['shot_type'].apply(lambda x: 1 if '3' in x else 0)
#cleaned matchup to away and home, 1 for away, 0 for home, shot type to 1 for 3 pters 0 for 2pters.

df=pd.get_dummies(df)


#split data into train and test

holdout=df[df['shot_made_flag'].isnull()]
data=df[df['shot_made_flag'].notnull()]

y=data['shot_made_flag']
X=data.drop(['shot_made_flag', 'shot_id'], axis=1)
#model doesn't predict on shot_id

X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=42, test_size=0.3)

#create baseline using Log Regression

steps=[('clf',LogisticRegression())]

pl=Pipeline(steps)

pl.fit(X_train, y_train)

preds=pl.predict(X_test)

preds=pd.Series(preds)

#preds=preds.apply(lambda x: 1 if x>0.5 else 0)

print(accuracy_score(y_test, preds))

#67.6% accuracy with basic model. apply to holdout and submit as baseline.

holdout_predict=pl.predict_proba(holdout.drop(['shot_made_flag', 'shot_id'], axis=1))
h_pred=pd.Series(holdout_predict[:,1])
#h_pred=h_pred.apply(lambda x: 1 if x>0.5 else 0)
h_pred.name='shot_made_flag'

submission=pd.DataFrame(holdout['shot_id'])
submission.index=range(5000)

submission=pd.concat([submission, h_pred], axis=1)

submission.to_csv('submission_baseline.csv',index=0)

#now we do with xgboost and gridsearchCV
#cv_params1={'max_depth':[3,5,7], 'min_child_weight':[1,3,5]}

#classifier_params1={'learning_rate': 0.1 , 'n_estimators':1000, 'seed':42, 'subsample':0.8, 'colsample_bytree':0.8, 'objective':'multi:softprob', 'num_class':2}

#cv1=GridSearchCV(xgb.XGBClassifier(**classifier_params1), cv_params1, scoring='r2', cv=5, n_jobs=3)

#cv1.fit(X_train, y_train)

#print(cv1.best_params_, cv1.best_score_)
#output was {'max_depth': 3, 'min_child_weight': 3} -0.3218649722369132
#use best params from above and below for final xgboost params

#cv_params2={'learning_rate':[0.1,0.01], 'subsample':[0.7,0.8,0.9]}

#classifier_params2={'n_estimators':1000, 'seed':42, 'colsample_by_tree':0.8, 'objective':'multi:softprob', 'num_class':2, 'max_depth': 3, 'min_child_weight':3}

#cv2=GridSearchCV(xgb.XGBClassifier(**classifier_params2), cv_params2, scoring='r2', cv=5, n_jobs=2)

#cv2.fit(X_train, y_train)

#print(cv2.best_params, cv2.best_score_)

final_params={'n_estimators':1000, 'seed':42, 'colsample_by_tree':0.8, 'objective':'multi:softprob', 'num_class':2, 'max_depth': 3, 'min_child_weight':3, 'learning_rate':0.01, 'subsample':0.7}
final_xgb=xgb.XGBClassifier(**final_params)

final_xgb.fit(X_train, y_train)
testpred=final_xgb.predict_proba(X_test)[:,1]
testpred=pd.Series(testpred)
testpred=testpred.apply(lambda x: 1 if x>0.5 else 0)
print(accuracy_score(y_test, testpred))
#we get accuracy score of 67.7% with xgboost tuned hyperparameters.

xgb_pred=final_xgb.predict_proba(holdout.drop(['shot_made_flag', 'shot_id'], axis=1))
xgb_pred=pd.Series(xgb_pred[:,1])
xgb_pred.name='shot_made_flag'

xgb_sub=pd.DataFrame(holdout['shot_id'])
xgb_sub.index=range(5000)

xgb_sub=pd.concat([xgb_sub, xgb_pred], axis=1)

xgb_sub.to_csv('xgb submission.csv', index=0)


xgb.plot_importance(final_xgb, max_num_features=10)


#looks like time remaining is the most important feature with shot distance and then location coming next.



















