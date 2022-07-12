# import library
import numpy as np
import pandas as pd
import datetime as dt

import sklearn
import xgboost

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import TweedieRegressor, GammaRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor

train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')
submission = pd.read_csv('../input/bike-sharing-demand/sampleSubmission.csv')

X_train = train.iloc[:,1:9]
y_train = train['count']
X_test = test.iloc[:,1:].copy()

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model_list = [AdaBoostRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(), RandomForestRegressor(),
    TweedieRegressor(), GammaRegressor(), PoissonRegressor(), KNeighborsRegressor(), NuSVR(), 
              DecisionTreeRegressor(), XGBRFRegressor()]

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

for i in range(len(model_list)):
    clf = model_list[i]
    clf.fit(X_tra, y_tra)
    pred = pd.DataFrame(clf.predict(X_val))
    print(model_list[i],':',np.sqrt(mean_squared_log_error(y_val, round(pred))))
    
    
#ExtraTreesRegressor() : 1.26722893374922
#RandomForestRegressor() : 1.2424650121337644
#KNeighborsRegressor() : 1.2756966796287237

parameters = {'n_estimators':[100, 300, 500], 'max_depth':[5, 10, 15]}
clf = GridSearchCV(ExtraTreesRegressor(random_state=0) , parameters, scoring = 'neg_mean_squared_log_error')
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)

#-1.9259232151567716
#{'max_depth': 10, 'n_estimators': 300}

parameters = {'n_estimators':[100, 300, 500], 'max_depth':[5, 10, 15]}
clf = GridSearchCV(RandomForestRegressor(random_state=0) , parameters, scoring = 'neg_mean_squared_log_error')
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)

#-1.9411437665765867
#{'max_depth': 10, 'n_estimators': 100}

parameters = {'n_neighbors':[3,5,10,15,20], 'weights':['uniform', 'distance']}
clf = GridSearchCV(KNeighborsRegressor() , parameters, scoring = 'neg_mean_squared_log_error')
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)

# -1.9072958256578905
# {'n_neighbors': 20, 'weights': 'uniform'}


clf = KNeighborsRegressor(n_neighbors=20, weights='uniform')
clf.fit(X_train, y_train)
submission['count'] = clf.predict(X_test)
submission['count'] = round(submission['count'])
submission.to_csv('./bike_submission.csv', index=False)
# score : 1.31106
