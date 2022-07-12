# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import matplotlib
import re
from scipy import stats
matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['font.size'] = 12
import random
random.seed(1)
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

import pickle
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['font.size'] = 12
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_orig, news_train_orig) = env.get_training_data()
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
print('Market train shape: ',market_train_df.shape)
print('News train shape: ', news_train_df.shape)

# Sort data
market_train_df = market_train_df.sort_values('time')
market_train_df['date'] = market_train_df['time'].dt.date

# Fill nan
market_train_fill = market_train_df
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
for i in range(len(column_raw)):
    market_train_fill[column_market[i]] = market_train_fill[column_market[i]].fillna(market_train_fill[column_raw[i]])
market_train_orig = market_train_orig.sort_values('time')
news_train_orig = news_train_orig.sort_values('time')
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
del market_train_orig
del news_train_orig
market_train_df = market_train_df.loc[market_train_df['time'].dt.date>=datetime.date(2009,1,1)]
news_train_df = news_train_df.loc[news_train_df['time'].dt.date>=datetime.date(2009,1,1)]
market_train_df['close_open_ratio'] = np.abs(market_train_df['close']/market_train_df['open'])
threshold = 0.5
print('In %i lines price increases by 50%% or more in a day' %(market_train_df['close_open_ratio']>=1.5).sum())
print('In %i lines price decreases by 50%% or more in a day' %(market_train_df['close_open_ratio']<=0.5).sum())
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] < 1.5]
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] > 0.5]
market_train_df = market_train_df.drop(columns=['close_open_ratio'])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
#the top hundred words.
vectorizer = CountVectorizer(max_features=1000, stop_words={"english"})
#we do this with TF-IDF.
X = vectorizer.fit_transform(news_train_df['headline'].values)
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_train_tf = tf_transformer.transform(X)
X_train_vals = X_train_tf.mean(axis=1)


del vectorizer
del X
del X_train_tf

#mean tf-idf score for news article.
d = pd.DataFrame(data=X_train_vals)
news_train_df['tf_score'] = d

market_train_df = market_train_df.loc[market_train_df['time'].dt.date>=datetime.date(2009,1,1)]
news_train_df = news_train_df.loc[news_train_df['time'].dt.date>=datetime.date(2009,1,1)]

#add indicator features
market_train_df['rolling_average_close_mean'] = market_train_df.groupby('assetCode')['close'].transform('mean')
market_train_df['rolling_average_vol_mean'] = market_train_df.groupby('assetCode')['volume'].transform('mean')
market_train_df['rolling_average_close_std'] = market_train_df.groupby('assetCode')['close'].transform('std')
market_train_df['rolling_average_vol_std'] = market_train_df.groupby('assetCode')['volume'].transform('std')
#some more refined instruments
market_train_df['moving_average_7_day'] = market_train_df.groupby('assetCode')['close'].transform(lambda x: x.rolling(window=7).mean())
ewma = pd.Series.ewm
market_train_df['ewma'] =  market_train_df.groupby('assetCode')['close'].transform(lambda x : ewma(x, span=30).mean())
market_train_df['moving_average_7_day'] = market_train_df['moving_average_7_day'].fillna(0)
market_train_df['ewma'] = market_train_df['ewma'].fillna(0)
for i in range(len(column_raw)):
    market_train_df[column_market[i]] = market_train_df[column_market[i]].fillna(market_train_df[column_raw[i]])
    print('Removing outliers ...')
column_return = column_market + column_raw + ['returnsOpenNextMktres10']
orig_len = market_train_df.shape[0]
for column in column_return:
    market_train_df = market_train_df.loc[market_train_df[column]>=-2]
    market_train_df = market_train_df.loc[market_train_df[column]<=2]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
print('Removing strange data ...')
orig_len = market_train_df.shape[0]
market_train_df = market_train_df[~market_train_df['assetCode'].isin(['PGN.N','EBRYY.OB'])]
#market_train_df = market_train_df[~market_train_df['assetName'].isin(['Unknown'])]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
# Function to remove outliers
def remove_outliers(data_frame, column_list, low=0.02, high=0.98):
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        data_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return data_frame
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence','noveltyCount12H',\
                  'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H',\
                  'volumeCounts3D','volumeCounts5D','volumeCounts7D']
print('Clipping news outliers ...')
news_train_df = remove_outliers(news_train_df, columns_outlier)
asset_code_dict = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
drop_columns = [col for col in news_train_df.columns if col not in ['sourceTimestamp', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence', 'relevance','firstCreated', 'assetCodes']]
columns_news = ['firstCreated','relevance','sentimentClass','sentimentNegative','sentimentNeutral',
               'sentimentPositive','noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodes','sourceTimestamp',
               'assetName','audiences', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence','time', 'tf_score']
def data_prep(market_df,news_df):
    market_df['date'] = market_df.time.dt.date
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df.drop(['time'], axis=1, inplace=True)
    
    news_df = news_df[columns_news]
    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    news_df['firstCreated'] = news_df.firstCreated.dt.date
    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    news_df['len_audiences'] = news_train_df['audiences'].map(lambda x: len(eval(x)))
    kcol = ['firstCreated', 'assetCodes']
    news_df = news_df.groupby(kcol, as_index=False).mean()
    market_df = pd.merge(market_df, news_df, how='left', left_on=['date', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    del news_df
    market_df['assetCodeT'] = market_df['assetCode'].map(asset_code_dict)
    market_df = market_df.drop(columns = ['firstCreated','assetCodes','assetName']).fillna(0) 
    print(market_df.count)
    return market_df
print('Merging data ...')
market_train_df = data_prep(market_train_df, news_train_df)
market_train_df.head()
market_train_df = market_train_df.loc[market_train_df['date']>=datetime.date(2009,1,1)]
num_columns = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 
               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'close_to_open', 'rolling_average_close_mean', 'rolling_average_vol_mean', 'rolling_average_close_std', 'ewma', 'rolling_average_close_std', 'sourceTimestamp', 'urgency', 'companyCount', 'takeSequence', 'bodySize', 'sentenceCount',
               'moving_average_7_day','relevance', 'sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
               'noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodesLen', 'asset_sentiment_count', 'len_audiences', 'tf_score']
cat_columns = ['assetCodeT']
feature_columns = num_columns+cat_columns
# Scaling of data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_scaler = StandardScaler()
#market_train_df[num_columns] = data_scaler.fit_transform(market_train_df[num_columns])
#data_scaler = MinMaxScaler()
market_train_df[num_columns] = data_scaler.fit_transform(market_train_df[num_columns])
from sklearn.model_selection import train_test_split

market_train_df = market_train_df.reset_index()
market_train_df = market_train_df.drop(columns='index')

# Random train-test split
train_indices, val_indices = train_test_split(market_train_df.index.values,test_size=0.1, random_state=92)
# Extract X and Y
def get_input(market_train, indices):
    X = market_train.loc[indices, feature_columns].values
    y = market_train.loc[indices,'returnsOpenNextMktres10'].map(lambda x: 0 if x<0 else 1).values
    #y = market_train.loc[indices,'returnsOpenNextMktres10'].map(lambda x: convert_to_class(x)).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'date']
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train_df, train_indices)
X_val,y_val,r_val,u_val,d_val = get_input(market_train_df, val_indices)
# Set up decay learning rate
def learning_rate_power(current_round):
    base_learning_rate = 0.19000424246380565
    min_learning_rate = 0.01
    lr = base_learning_rate * np.power(0.995,current_round)
    return max(lr, min_learning_rate)


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

tune_params = {'n_estimators': [200,500,1000,2500,5000],
              'max_depth': sp_randint(4,12),
              'colsample_bytree':sp_uniform(loc=0.8, scale=0.15),
              'min_child_samples':sp_randint(60,120),
              'subsample': sp_uniform(loc=0.75, scale=0.25),
              'reg_lambda':[1e-3, 1e-2, 1e-1, 1]}

fit_params = {'early_stopping_rounds':40,
              'eval_metric': 'accuracy',
              'eval_set': [(X_train, y_train), (X_val, y_val)],
              'verbose': 20,
              'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_power)]}

lgb_clf = lgb.LGBMClassifier(n_jobs=4, objective='binary',random_state=1)
gs = RandomizedSearchCV(estimator=lgb_clf, 
                        param_distributions=tune_params, 
                        n_iter=40,
                        scoring='f1',
                        cv=5,
                        refit=True,
                        random_state=1,
                        verbose=True)
lgb_clf = lgb.LGBMClassifier(n_jobs=4,
                             objective='multiclass',
                            random_state=100)
opt_params = {'n_estimators':500,
              'boosting_type': 'dart',
              'objective': 'binary',
              'num_leaves':2452,
              'min_child_samples':212,
              'reg_lambda':0.01}
""" 
param_dist = {'objective':'binary:logistic','boosting_type': 'dart', 'n_estimators': 500, 'num_leaves':2452,
              'min_child_samples':212, 'reg_lambda':0.01}
"""
lgb_clf.set_params(**opt_params)
lgb_clf.fit(X_train, y_train,**fit_params)



print('Training accuracy: ', accuracy_score(y_train, lgb_clf.predict(X_train)))
print('F1 accuracy : ', f1_score(y_val, lgb_clf.predict(X_val)))
print('Validation accuracy: ', accuracy_score(y_val, lgb_clf.predict(X_val)))

"""
import shap
X = market_train_df.loc[:, feature_columns]
features_imp = pd.DataFrame()
features_imp['features'] = list(feature_columns)[:]
features_imp['importance'] = lgb_clf.feature_importances_
features_imp = features_imp.sort_values(by='importance', ascending=False).reset_index()

y_plot = -np.arange(15)
plt.figure(figsize=(10,6))
plt.barh(y_plot, features_imp.loc[:14,'importance'].values)
plt.yticks(y_plot,(features_imp.loc[:14,'features']))
plt.xlabel('Feature importance')
plt.title('Features importance')
plt.tight_layout()
shap_explainer = shap.TreeExplainer(lgb_clf)
sample = X.sample(frac=0.50, random_state=100)
shap_values = shap_explainer.shap_values(sample)
shap.summary_plot(shap_values, sample)
"""

# Rescale confidence
def rescale(data_in, data_ref):
    scaler_ref =  StandardScaler()
    scaler_ref.fit(data_ref.reshape(-1,1))
    scaler_in = StandardScaler()
    data_in = scaler_in.fit_transform(data_in.reshape(-1,1))
    data_in = scaler_ref.inverse_transform(data_in)[:,0]
    return data_in
    
y_pred_proba = lgb_clf.predict_proba(X_val)
predicted_return = y_pred_proba[:,1] - y_pred_proba[:,0]
#predicted_return = confidence_out(y_pred_proba)
predicted_return = rescale(predicted_return, r_train)

# distribution of confidence that will be used as submission
plt.hist(predicted_return, bins='auto', label='Predicted confidence')
plt.hist(r_val, bins='auto',alpha=0.8, label='True market return')
plt.title("predicted confidence")
plt.legend(loc='best')
plt.xlim(-1,1)
plt.show()

# calculation of actual metric that is used to calculate final score
r_val = r_val.clip(-1,1) # get rid of outliers.
x_t_i = predicted_return * r_val * u_val
data = {'day' : d_val, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print('Validation score', score_valid)

# This code is inspired from this kernel: https://www.kaggle.com/skooch/lgbm-w-random-split-2
#

#with 1 : 0.52, 0.54
#with 3 classifiers: 0.616 accuracy, 0.62 f1 score.
#with 5 : Accuracy score clfs: 0.622084 F1 score clfs: 0.637640
#with 7 : Accuracy score clfs: 0.624605 F1 score clfs: 0.639552
#with 9 :  Accuracy score clfs: 0.626605 F1 score clfs: 0.64552

clfs = []
for i in range(1):
    clf = lgb.LGBMClassifier(learning_rate=0.1, random_state=1200+i, silent=True,
                             n_jobs=4, n_estimators=500)
    clf.set_params(**opt_params)
    clfs.append(('lgbm%i'%i, clf))

def split_data(X, y, test_percentage=0.2, seed=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    return X_train, y_train, X_test, y_test 

def _parallel_fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    
    # randomly split the data so we have a test set for early stopping
    X_train, y_train, X_test, y_test = split_data(X, y, seed=1992)
    
    # update the fit params with our new split
    fit_params["eval_set"] = [(X_train,y_train), (X_test,y_test)]
    
    # fit the estimator
    if sample_weight is not None:
        estimator.fit(X_train, y_train, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    return estimator
    
    
class VotingClassifierLGBM(VotingClassifier):
    '''
    This implements the fit method of the VotingClassifier propagating fit_params
    '''
    def fit(self, X, y, sample_weight=None, **fit_params):
        
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        if sample_weight is not None:
            for name, step in self.estimators:
                if not has_fit_parameter(step, 'sample_weight'):
                    raise ValueError('Underlying estimator \'%s\' does not'
                                     ' support sample weights.' % name)
        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum([clf is None for _, clf in self.estimators])
        if n_isnone == len(self.estimators):
            raise ValueError('All estimators are None. At least one is '
                             'required to be a classifier!')

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []

        transformed_y = self.le_.transform(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
                                                 sample_weight=sample_weight, **fit_params)
                for clf in clfs if clf is not None)

        return self
        
vc = VotingClassifierLGBM(clfs, voting='soft')
vc.fit(X_train, y_train, **fit_params)
filename = 'VotingClassifierLGBM.sav'
pickle.dump(vc, open(filename, 'wb'))

vc = pickle.load(open(filename, 'rb'))
vc.voting = 'soft'
predicted_class = vc.predict(X_val)
predicted_return = vc.predict_proba(X_val)
#predicted_return = confidence_out(predicted_return)
predicted_return = vc.predict_proba(X_val)[:,1]*2-1
predicted_return = rescale(predicted_return, r_train)
plt.hist(predicted_class, bins='auto')

vc.voting = 'soft'
global_accuracy_soft = accuracy_score(y_val, predicted_class)
global_f1_soft = f1_score(y_val, predicted_class)
print('Accuracy score clfs: %f' % global_accuracy_soft)
print('F1 score clfs: %f' % global_f1_soft)
# distribution of confidence that will be used as submission
plt.hist(predicted_return, bins='auto', label='Prediciton')
plt.hist(r_val, bins='auto',alpha=0.8, label='True data')
plt.title("predicted confidence")
plt.legend(loc='best')
plt.xlim(-1,1)
plt.show()
# calculation of actual metric that is used to calculate final score
r_val = r_val.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = predicted_return * r_val * u_val
data = {'day' : d_val, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print('Validation score', score_valid)

days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if n_days % 50 == 0:
        print(n_days,end=' ')

    t = time.time()
    column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
    market_obs_df['close_open_ratio'] = np.abs(market_obs_df['close']/market_obs_df['open'])
    market_obs_df['rolling_average_close_mean'] = market_obs_df.groupby('assetCode')['close'].transform('mean')
    market_obs_df['rolling_average_vol_mean'] = market_obs_df.groupby('assetCode')['volume'].transform('mean')
    market_obs_df['rolling_average_close_std'] = market_obs_df.groupby('assetCode')['close'].transform('std')
    market_obs_df['rolling_average_vol_std'] = market_obs_df.groupby('assetCode')['volume'].transform('std')
    #some more refined instruments
    market_obs_df['moving_average_7_day'] = market_obs_df.groupby('assetCode')['close'].transform(lambda x: x.rolling(window=7).mean())
    ewma = pd.Series.ewm
    market_obs_df['ewma'] =  market_obs_df.groupby('assetCode')['close'].transform(lambda x : ewma(x, span=30).mean())
    market_obs_df['moving_average_7_day'] = market_obs_df['moving_average_7_day'].fillna(0)
    market_obs_df['ewma'] = market_obs_df['ewma'].fillna(0)
    for i in range(len(column_raw)):
        market_obs_df[column_market[i]] = market_obs_df[column_market[i]].fillna(market_obs_df[column_raw[i]])
    vectorizer = CountVectorizer(max_features=1000, stop_words={"english"})
#we do this with TF-IDF.
    X = vectorizer.fit_transform(news_obs_df['headline'].values)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X)
    X_train_tf = tf_transformer.transform(X)
    X_train_vals = X_train_tf.mean(axis=1)


    del vectorizer
    del X
    del X_train_tf

#mean tf-idf score for news article.
    d = pd.DataFrame(data=X_train_vals)
    news_obs_df['tf_score'] = d
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(asset_code_dict.keys())]
    market_obs = data_prep(market_obs_df, news_obs_df)
    market_obs[num_columns] = data_scaler.transform(market_obs[num_columns])

    X_live = market_obs[feature_columns].values
    prep_time += time.time() - t

    t = time.time()
    lp = vc.predict_proba(X_live)
    prediction_time += time.time() -t

    t = time.time()
    confidence = lp[:,1] - lp[:,0]
    #confidence = confidence_out(lp)
    confidence = rescale(confidence, r_train)
    preds = pd.DataFrame({'assetCode':market_obs['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
plt.hist(confidence, bins='auto')
plt.title("predicted confidence")
plt.show()