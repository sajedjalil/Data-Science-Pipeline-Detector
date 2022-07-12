import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

train = pd.read_csv('../input/train_v2.csv')
sample_submission = pd.read_csv('../input/sample_submission_v2.csv')
transactions = pd.read_csv('../input/transactions_v2.csv')
user_logs = pd.read_csv('../input/user_logs_v2.csv')
members = pd.read_csv('../input/members_v3.csv')

# set the options so the output format can be displayed correctly
pd.set_option('expand_frame_repr', True)
pd.set_option('display.max_rows', 30000000)
pd.set_option('display.max_columns', 100)

# check the number of duplicate accounts in each table
train.duplicated('msno').sum()
sample_submission.duplicated('msno').sum()
transactions.duplicated('msno').sum()
user_logs.duplicated('msno').sum()
members.duplicated('msno').sum()

# returns the max value of numerical variables and membership_expire_date
# returns the min value of transaction date
# returns the mode of ordinal variable and dummy variables, if multiple values share the same frequency, keep the first one
transactions_v2 = transactions.groupby('msno', as_index = False).agg({'payment_method_id': lambda x:x.value_counts().index[0], 'payment_plan_days': 'max', 'plan_list_price': 'max',
                                       'actual_amount_paid': 'max', 'is_auto_renew': lambda x:x.value_counts().index[0], 'transaction_date': 'min', 'membership_expire_date': 'max',
                                       'is_cancel': lambda x:x.value_counts().index[0]})

# returns the max value of date and number of unique songs
# returns the sum of other variables
user_logs_v2 = user_logs.groupby('msno', as_index = False).agg({'date': 'max', 'num_25': 'sum', 'num_50': 'sum', 'num_75': 'sum',
                                 'num_985': 'sum', 'num_100': 'sum', 'num_unq': 'max', 'total_secs': 'sum'})

# calculate the percentage of number of songs played within certain period
user_logs_v2['percent_25'] = user_logs_v2['num_25']/(user_logs_v2['num_25']+user_logs_v2['num_50']+user_logs_v2['num_75']+user_logs_v2['num_985']+user_logs_v2['num_100'])
user_logs_v2['percent_50'] = user_logs_v2['num_50']/(user_logs_v2['num_25']+user_logs_v2['num_50']+user_logs_v2['num_75']+user_logs_v2['num_985']+user_logs_v2['num_100'])
user_logs_v2['percent_100'] = (user_logs_v2['num_985']+user_logs_v2['num_100'])/(user_logs_v2['num_25']+user_logs_v2['num_50']+user_logs_v2['num_75']+user_logs_v2['num_985']+user_logs_v2['num_100'])

# drop useless variables
user_logs_v3 = user_logs_v2.drop(columns = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100'])

# merge between different tables for modelling purpose
dataset_train = train.merge(members, on = 'msno', how = 'left').merge(transactions_v2, on = 'msno', how = 'left').merge(user_logs_v3, on = 'msno', how = 'left')
dataset_train.dtypes

# date in csv will be recognized as float in python
# this value needs to be converted back to date
dataset_train['registration_init_time'] = pd.to_datetime(dataset_train['registration_init_time'], format = '%Y%m%d')
dataset_train['transaction_date'] = pd.to_datetime(dataset_train['transaction_date'], format = '%Y%m%d')
dataset_train['membership_expire_date'] = pd.to_datetime(dataset_train['membership_expire_date'], format = '%Y%m%d')
dataset_train['date'] = pd.to_datetime(dataset_train['date'], format = '%Y%m%d')

# check the maximum of datetime value
dataset_train.select_dtypes(include = ['datetime64[ns]']).max()

# create new day columns for modelling purpose
dataset_train['registration_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['registration_init_time']).astype('timedelta64[D]')
dataset_train['transaction_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['transaction_date']).astype('timedelta64[D]')
dataset_train['membership_expire_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['membership_expire_date']).astype('timedelta64[D]')
dataset_train['last_play_day'] = (dataset_train['membership_expire_date'].max() - dataset_train['date']).astype('timedelta64[D]')

# check the distribution of age due to the documentation explanation
dataset_train['bd'].value_counts()

# remove gender and age since missing value or incorrect value is over 50%
dataset_train_v2 = dataset_train.drop(columns = ['msno', 'gender', 'bd', 'registration_init_time', 'transaction_date', 'membership_expire_date', 'date'])
dataset_train_v2.dtypes

# check the number of missing values in each column
dataset_train_v2.isna().sum()

# Handle missing value of part of numeric columns by using mode
def replacemode(i):
    dataset_train_v2[i] = dataset_train_v2[i].fillna(dataset_train_v2[i].value_counts().index[0])
    return 

replacemode('city')
replacemode('registered_via')
replacemode('payment_method_id')
replacemode('payment_plan_days')
replacemode('is_auto_renew')
replacemode('is_cancel')

# Handle missing value of part of numeric columns by using mean
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
def replacemean(i):
    dataset_train_v2[i] = mean_imputer.fit_transform(dataset_train_v2[[i]])
    return 

replacemean('plan_list_price')
replacemean('actual_amount_paid')
replacemean('num_unq')
replacemean('total_secs')
replacemean('percent_25')
replacemean('percent_50')
replacemean('percent_100')
replacemean('registration_day')
replacemean('transaction_day')
replacemean('membership_expire_day')
replacemean('last_play_day')

# Handle outliers by using capping
def replaceoutlier(i):
    mean, std = np.mean(dataset_train_v2[i]), np.std(dataset_train_v2[i])
    cut_off = std*3
    lower, upper = mean - cut_off, mean + cut_off
    dataset_train_v2[i][dataset_train_v2[i] < lower] = lower
    dataset_train_v2[i][dataset_train_v2[i] > upper] = upper
    return

replaceoutlier('plan_list_price')
replaceoutlier('actual_amount_paid')
replaceoutlier('num_unq')
replaceoutlier('total_secs')
replaceoutlier('percent_25')
replaceoutlier('percent_50')
replaceoutlier('percent_100')
replaceoutlier('registration_day')
replaceoutlier('transaction_day')
replaceoutlier('membership_expire_day')
replaceoutlier('last_play_day')

dataset_train_v2.dtypes
dataset_train_v2.describe()

# convert categorical variables into string
dataset_train_v2.iloc[:, 1:4] = dataset_train_v2.iloc[:, 1:4].astype(str)

# replace discrete features with historical churn rate
city_mean = pd.DataFrame(dataset_train_v2.groupby('city')['is_churn'].mean().reset_index())
city_mean.rename(columns = {'is_churn': 'city_mean'}, inplace=True)
register_mean = pd.DataFrame(dataset_train_v2.groupby('registered_via')['is_churn'].mean().reset_index())
register_mean.rename(columns = {'is_churn': 'register_mean'}, inplace=True)
payment_mean = pd.DataFrame(dataset_train_v2.groupby('payment_method_id')['is_churn'].mean().reset_index())
payment_mean.rename(columns = {'is_churn': 'payment_mean'}, inplace=True)

dataset_train_v3 = dataset_train_v2.merge(city_mean, on = 'city', how = 'left').merge(register_mean, on = 'registered_via', how = 'left').merge(payment_mean, on = 'payment_method_id', how = 'left')
dataset_train_v3 = dataset_train_v3.drop(columns = ['city', 'registered_via', 'payment_method_id'])

# Feature Scaling for modelling purpose by using both min-max-scaling
from sklearn.preprocessing import MinMaxScaler
X = dataset_train_v3.drop(columns = ['is_churn'])
Y = dataset_train_v3['is_churn']
nm_X = pd.DataFrame(MinMaxScaler().fit_transform(X))
nm_X.columns = X.columns.values
nm_X.index = X.index.values

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2, f_classif
nm_col = ['is_auto_renew', 'is_cancel']
nm_X_v2 = nm_X.drop(columns = nm_col)
nm_X_v3 = pd.DataFrame(nm_X, columns = nm_col)
nm_X_v4 = pd.DataFrame(SelectKBest(score_func=chi2, k='all').fit(nm_X_v3, Y).pvalues_ <= 0.05, columns = ['importance'])
nm_X_v4.index = nm_X_v3.columns.values
nm_X_v5 = pd.DataFrame(SelectKBest(score_func=f_classif, k='all').fit(nm_X_v2, Y).pvalues_ <= 0.05, columns = ['importance'])
nm_X_v5.index = nm_X_v2.columns.values
nm_X_v6 = pd.concat([nm_X_v4, nm_X_v5])
nm_selected = list(pd.Series(nm_X_v6[nm_X_v6['importance'] == 1].index.values))
nm_X_v7 = pd.DataFrame(nm_X, columns = nm_selected)

# Dimension Reduction
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(nm_X_v7)
np.cumsum(pca.explained_variance_ratio_)
nm_X_v8 = PCA(n_components=10).fit_transform(nm_X_v7)

# Split into train and test Set
from sklearn.model_selection import train_test_split
nm_X_train, nm_X_test, nm_Y_train, nm_Y_test = train_test_split(nm_X_v8, Y, test_size = 0.3, random_state = 0)

# Fit training set into different algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import xgboost as xgb

nm_models = []
nm_models.append(('KNN', KNeighborsClassifier()))
nm_models.append(('LR', LogisticRegression()))
nm_models.append(('LDA', LinearDiscriminantAnalysis()))
nm_models.append(('QDA', QuadraticDiscriminantAnalysis()))
nm_models.append(('CART', DecisionTreeClassifier()))
nm_models.append(('NB', GaussianNB()))
# nm_models.append(('Linear SVM', SVC(kernel = 'linear')))
# nm_models.append(('Kernel SVM', SVC(kernel = 'rbf')))
ensembles = []
ensembles.append(('BC', BaggingClassifier(base_estimator=LogisticRegression())))
ensembles.append(('AB', AdaBoostClassifier(base_estimator=LogisticRegression())))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('XGB', XGBClassifier()))

from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in nm_models:
	nm_cv_results = cross_val_score(model, nm_X_train, nm_Y_train, cv=10, scoring='neg_log_loss', n_jobs = -1)
	results.append(nm_cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, nm_cv_results.mean(), nm_cv_results.std())
	print(msg)
    
results2 = []
names2 = []
for name2, model2 in ensembles:
	en_cv_results = cross_val_score(model2, nm_X_train, nm_Y_train, cv=10, scoring='neg_log_loss', n_jobs = -1)
	results2.append(en_cv_results)
	names2.append(name2)
	msg2 = "%s: %f (%f)" % (name2, en_cv_results.mean(), en_cv_results.std())
	print(msg2)

# Apply Grid Search on XGBoost since it returns the best result on Cross Validation among all models
from sklearn.model_selection import GridSearchCV
parameters = {"learning_rate": [0.1],
              "min_child_weight": [1],
              "max_depth": [9],
              "gamma": [0.1],
              "subsample": [0.8],
              "colsample_bytree": [0.8],
              "objective": ['binary:logistic']}
grid_search_XGB = GridSearchCV(estimator = XGBClassifier(), param_grid = parameters, scoring = "neg_log_loss", cv = 10, n_jobs = -1)
grid_result_XGB = grid_search_XGB.fit(nm_X_train, nm_Y_train)
print("Best: %f using %s" % (grid_result_XGB.best_score_, grid_result_XGB.best_params_)) 

# Evaluate tuned XGBoost model result on test dataset because it provides the best result
from sklearn.metrics import log_loss
nm_Y_predict = grid_result_XGB.predict_proba(nm_X_test)
logloss = log_loss(nm_Y_test, nm_Y_predict)  
print("Log Loss Score: %s" % (logloss))