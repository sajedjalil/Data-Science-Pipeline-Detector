"""
This is my submission script. The model is composed of 3 blocks:

    - feature engineering: Transform data by Id, compose classic statistical measures (mean, std, ..)
                           as well as time varying variables (e.g X_t25 is the value of X at time = 0.25 
                           scaled between [0, 1]), and the same for diff_X, ratio_X etc..
                           I didn't have enough time to compose more sophisticated features (wavelets modeling,
                           ARIMA time series, correlation...)

    - Gaussian mixture model: classification of dataset in 3 groups by a GMM on the MP feature, in order to capture
                              some pattern. We can optimize the choice of number of clusters. I classify ids in the
                              test set in one of these clusters.

    - Learning: I use a xgboost regressor on each cluster. The advantage of xgboost is that it handle nan values.
                I also tried sklearn.GradientBoostingregressor using the 'lad' loss, which seems more powerful, but
                does not handle missing values. As a work around, I split my features in groups where each group
                has a full dataset then compose all. This gave merely the same score as xgboost with missing values.

The main score I use to validate my models is (score - score_median) / score_median, which gives an idea of the performence
compared to a single value prediction using the median

I tested a lot of approches, like trying to classify outliers, without much success (I got maximum 60% roc score). I also
tried to fill missing values by nearest neighbors but I was trapped in overfitting. I also tried some neural networks with
scikit-neural-network package, but no better performence. 

Anyway, this was a fun kaggle to play with, I wish I had more time to spend on and I am looking forward to your feedbacks
and to continue to learn.
"""

import random
import functools
import itertools
import copy
import multiprocessing
import concurrent.futures

import xgboost

from sklearn import ensemble, cluster
from scipy import stats

from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.grid_search import RandomizedSearchCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



columns = ['Id', 'minutes_past', 'radardist_km', 'Ref', 'Ref_5x5_10th', 'Ref_5x5_50th', 'Ref_5x5_90th',
           'RefComposite', 'RefComposite_5x5_10th', 'RefComposite_5x5_50th', 'RefComposite_5x5_90th',
           'RhoHV', 'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th',
           'Zdr','Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th',
           'Kdp', 'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th',
           'Expected']


var_columns = ['Ref', 'Ref_5x5_10th', 'Ref_5x5_50th', 'Ref_5x5_90th',
               'RefComposite', 'RefComposite_5x5_10th', 'RefComposite_5x5_50th', 'RefComposite_5x5_90th',
               'RhoHV', 'RhoHV_5x5_10th', 'RhoHV_5x5_50th', 'RhoHV_5x5_90th',
               'Zdr','Zdr_5x5_10th', 'Zdr_5x5_50th', 'Zdr_5x5_90th',
               'Kdp', 'Kdp_5x5_10th', 'Kdp_5x5_50th', 'Kdp_5x5_90th']

diff_columns = ['Ref',
               'RefComposite',
               'RhoHV',
               'Zdr',
               'Kdp']


def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in range(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)  # 0.53
            sum = sum + mmperhr * hours
    return sum

def transform_features(x):

    x = x.sort('minutes_past', ascending=True)
    est = marshall_palmer(x['Ref'], x['minutes_past'])
    x['est'] = est

    x['len'] = len(x)

    minutes = list(x['minutes_past'].values)

    x['min_q_25'] = minutes[int(len(minutes) / 4)]
    x['min_q_50'] = minutes[int(len(minutes) / 2)]
    x['min_q_75'] = minutes[int(3 * len(minutes) / 4)]
    x['min_q_100'] = minutes[-1]

    for feature in var_columns:

        values = np.array(x[feature]) - np.mean(x[feature])

        x[feature + '_mean'] = x[feature].mean()
        x[feature + '_std'] = x[feature].std() if len(x) > 1 else 0

        if feature in diff_columns:
            x[feature + '_t0'] = values[0]
            x[feature + '_t25'] = values[int(len(values) / 4)]
            x[feature + '_t50'] = values[int(len(values) / 2)]
            x[feature + '_t75'] = values[int(3 * len(values) / 4)]
            x[feature + '_t100'] = values[-1]

            x[feature + '_diff_25'] = sum(np.diff(values[:int(len(values) / 4)]))
            x[feature + '_diff_50'] = sum(np.diff(values[int(len(values) / 4):int(len(values) / 2)]))
            x[feature + '_diff_75'] = sum(np.diff(values[int(len(values) / 2):int(3 * len(values) / 4)]))
            x[feature + '_diff_100'] = sum(np.diff(values[int(3 * len(values) / 4):]))

            x[feature + '_ratio_25'] = values[int(len(values) / 4)] - values[0]
            x[feature + '_ratio_50'] = values[int(len(values) / 2)] - values[int(len(values) / 4)]
            x[feature + '_ratio_75'] = values[int(3 * len(values) / 4)] - values[int(len(values) / 2)]
            x[feature + '_ratio_100'] = values[-1] - values[int(3 * len(values) / 4)]

    x.drop_duplicates(subset='Id', inplace=True)

    return x

def group_by(x=None, feature=None, func=None):
    return x.groupby(feature, group_keys=False).apply(func)


###########################
##### TRANSFORM DATA ######
###########################

print('transform learn dataset...')

df_learn = pd.read_csv('../input/train.csv', usecols=columns)

# drop lines with no Ref values 
df_learn = df_learn[~df_learn['Ref'].isnull()]

df_learn = df_learn.reset_index()

df_learn = df_learn.groupby('Id', group_keys=False).apply(transform_features)

df_learn = df_learn.replace('inf', 1e5)
df_learn = df_learn.replace('-inf', -1e5)

df_learn.drop_duplicates(subset='Id', inplace=True)


###########################
######## FEATURES #########
###########################


# Compose learning features list

cols_org = ['radardist_km', 'len', 'est',
              'min_q_25', 'min_q_50', 'min_q_75', 'min_q_100',
              'Ref_mean', 'Ref_std',
              'Ref_t0', 'Ref_t25', 'Ref_t50', 'Ref_t75', 'Ref_t100',
              'RefComposite_mean', 'RefComposite_std',
              'RefComposite_t0', 'RefComposite_t25', 'RefComposite_t50', 'RefComposite_t75', 'RefComposite_t100',
              'RhoHV_mean', 'RhoHV_std',
              'RhoHV_t0','RhoHV_t25', 'RhoHV_t50', 'RhoHV_t75', 'RhoHV_t100',
              'Zdr_mean', 'Zdr_std',
              'Zdr_t0', 'Zdr_t25', 'Zdr_t50','Zdr_t75', 'Zdr_t100',
              'Kdp_mean', 'Kdp_std',
              'Kdp_t0', 'Kdp_t25', 'Kdp_t50', 'Kdp_t75', 'Kdp_t100',
              'RefComposite_5x5_10th_mean', 'RefComposite_5x5_50th_mean', 'RefComposite_5x5_90th_mean',
              'RefComposite_5x5_10th_std', 'RefComposite_5x5_50th_std', 'RefComposite_5x5_90th_std',
              'Ref_5x5_10th_std', 'Ref_5x5_50th_std', 'Ref_5x5_90th_std',
              'Ref_5x5_10th_mean', 'Ref_5x5_50th_mean', 'Ref_5x5_90th_mean',
              'RhoHV_5x5_10th_mean', 'RhoHV_5x5_50th_mean', 'RhoHV_5x5_90th_mean',
              'RhoHV_5x5_10th_std', 'RhoHV_5x5_50th_std', 'RhoHV_5x5_90th_std',
              'Zdr_5x5_10th_mean', 'Zdr_5x5_50th_mean', 'Zdr_5x5_90th_mean',
              'Zdr_5x5_10th_std', 'Zdr_5x5_50th_std', 'Zdr_5x5_90th_std',
              'Kdp_5x5_10th_mean', 'Kdp_5x5_50th_mean', 'Kdp_5x5_90th_mean',
              'Kdp_5x5_10th_std', 'Kdp_5x5_50th_std', 'Kdp_5x5_90th_std',]

diff_cols = ['Ref_diff_25', 'Ref_diff_50', 'Ref_diff_75', 'Ref_diff_100',
              'RefComposite_diff_25', 'RefComposite_diff_50', 'RefComposite_diff_75', 'RefComposite_diff_100',
              'RhoHV_diff_25','RhoHV_diff_50', 'RhoHV_diff_75', 'RhoHV_diff_100',
              'Zdr_diff_25', 'Zdr_diff_50', 'Zdr_diff_75','Zdr_diff_100',
              'Kdp_diff_25', 'Kdp_diff_50', 'Kdp_diff_75', 'Kdp_diff_100']

ratio_cols = ['Ref_ratio_25', 'Ref_ratio_50', 'Ref_ratio_75', 'Ref_ratio_100',
              'RefComposite_ratio_25', 'RefComposite_ratio_50', 'RefComposite_ratio_75', 'RefComposite_ratio_100',
              'RhoHV_ratio_25','RhoHV_ratio_50', 'RhoHV_ratio_75', 'RhoHV_ratio_100',
              'Zdr_ratio_25', 'Zdr_ratio_50', 'Zdr_ratio_75','Zdr_ratio_100',
              'Kdp_ratio_25', 'Kdp_ratio_50', 'Kdp_ratio_75', 'Kdp_ratio_100']

learn_cols = copy.copy(cols_org)
learn_cols.extend(diff_cols)
learn_cols.extend(ratio_cols)

###########################
######### LEARNING ########
###########################

# seperate learn ids and test ids

random.shuffle(all_learn_ids)

all_learn_ids = list(df_learn['Id'].unique())

learn_ids = all_learn_ids[:int(0.75 * len(all_learn_ids))]
test_ids = all_learn_ids[int(0.75 * len(all_learn_ids)):]

df_learn_local = df_learn[df_learn['Id'].isin(learn_ids)]
df_test_local = df_learn[df_learn['Id'].isin(test_ids)]

# drop outliers
df_learn_local = df_learn_local[df_learn_local['Expected'] < 70]

###########################
########## GMM ############
###########################

model = mixture.GMM(n_components=3)
model.fit(np.asmatrix(df_learn_local['est']).transpose())

labels = model.predict(np.asmatrix(df_learn_local['est']).transpose())
df_learn_local['est_cl']= labels
print(df_learn_local.groupby('est_cl').apply(lambda x: (np.min(x['est']), np.max(x['est']), len(x)) ) )

"""
# plot Expected values distribution in clusters

fig = plt.figure()
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    mask_cluster = df_learn_local['est_cl'] == k
    df_cluster = df_learn_local[mask_cluster]

    density = stats.kde.gaussian_kde(df_cluster['Expected'])
    x = np.arange(0., df_cluster['Expected'].max(), float(df_cluster['Expected'].max() / 100))
    plt.plot(x, density(x), c=col, label=str(int(k)))

    plt.legend()
    plt.hold(True)

fig.savefig('density_' + str(len(unique_labels)) + '.png')
"""

#########################
## LEARNING CLASSIFIER ##
#########################

X_learn_clf = df_learn_local[:int(0.75 * len(df_learn_local))]['est']
Y_learn_clf = df_learn_local[:int(0.75 * len(df_learn_local))]['est_cl']

X_test_clf = df_learn_local[int(0.75 * len(df_learn_local)):]['est']
Y_test_clf = df_learn_local[int(0.75 * len(df_learn_local)):]['est_cl']

"""
customized score function for multilabel classification. I thought of this as I wanted a simple real simple
metric that I can compare to the score of random guessing: 
s = |M - diag(M)| where M is the confusion matrix
"""
def func_score(est, X,y):
    y_hat = est.predict(X)
    cf_clf = metrics.confusion_matrix(y, y_hat)
    cf_clf = cf_clf / np.sum(cf_clf, 0)
    score_clf = np.sum(cf_clf) - np.sum(np.diag(cf_clf.diagonal()))
    return score_clf

# I simply use a DecisionTreeClassifier without any tunning.. this is not the most critical part I thinl
classifier = DecisionTreeClassifier()
classifier.fit(np.reshape(X_learn_clf,  (X_learn_clf.shape[0], 1)), Y_learn_clf)

Y_clf_hat = classifier.predict(np.reshape(X_test_clf,  (X_test_clf.shape[0], 1)))

cf_clf = metrics.confusion_matrix(Y_test_clf, Y_clf_hat)
cf_med = metrics.confusion_matrix(Y_test_clf, [random.choice([0, 1, 2]) for i in range(len(Y_test_clf))])

cf_clf = cf_clf / np.sum(cf_clf, 0)
cf_med = cf_med / np.sum(cf_med, 0)

score_clf = np.sum(cf_clf) - np.sum(np.diag(cf_clf.diagonal()))
score_med = np.sum(cf_med) - np.sum(np.diag(cf_med.diagonal()))

print("clf score {}".format(score_clf))
print("rand score {}".format(score_med))

###############################
#### LEARNING ON CLUSTERS #####
###############################

X_test_clf = df_test_local['est']

Y_clf_hat = classifier.predict((np.reshape(X_test_clf,  (X_test_clf.shape[0], 1))))
df_test_local['est_cl_hat'] = Y_clf_hat

# class_0
print('learning class 0')
df_learn_0 = df_learn_local[df_learn_local['est_cl'] == 0]
df_test_0 = df_test_local[df_test_local['est_cl_hat'] == 0]

X_learn_0 = df_learn_0[learn_cols]
Y_learn_0 = np.log(df_learn_0['Expected'])

X_test_0 = df_test_0[learn_cols]
Y_test_0 = np.log(df_test_0['Expected'])

print('shape learn, test:', len(X_learn_0), len(X_test_0) )

# Gridsearch the best parameters with RandomizedGridSearch
regressor_0 = xgboost.XGBRegressor(learning_rate=0.01, n_estimators=200)

regressor_0.fit(X_learn_0, Y_learn_0)

print(rs.best_params_)

Y_0_hat = np.exp(regressor_0.predict(X_test_0))
df_test_0['y_hat'] = Y_0_hat

score = metrics.mean_absolute_error(np.exp(Y_test_0), Y_0_hat)
score_median = metrics.mean_absolute_error(np.exp(Y_test_0), [np.median(np.exp(Y_test_0))] * len(np.exp(Y_test_0)))
print("CLASS 0: score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

# class_1
print('learning class 1')
df_learn_1 = df_learn_local[df_learn_local['est_cl'] == 1]
df_test_1 = df_test_local[df_test_local['est_cl_hat'] == 1]

X_learn_1 = df_learn_1[learn_cols]
Y_learn_1 = np.log(df_learn_1['Expected'])

X_test_1 = df_test_1[learn_cols]
Y_test_1 = np.log(df_test_1['Expected'])

print('shape learn, test:', len(X_learn_1), len(X_test_1) ) 
regressor_1 = xgboost.XGBRegressor(learning_rate=0.01, n_estimators=200)

regressor_1.fit(X_learn_1, Y_learn_1)

Y_1_hat = np.exp(regressor_1.predict(X_test_1))
df_test_1['y_hat'] = Y_1_hat

score = metrics.mean_absolute_error(np.exp(Y_test_1), Y_1_hat)
score_median = metrics.mean_absolute_error(np.exp(Y_test_1), [np.median(np.exp(Y_test_1))] * len(np.exp(Y_test_1)))
print("CLASS 1: score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

# class_2
print('learning class 2')
df_learn_2 = df_learn_local[df_learn_local['est_cl'] == 2]
df_test_2 = df_test_local[df_test_local['est_cl_hat'] == 2]

X_learn_2 = df_learn_2[learn_cols]
Y_learn_2 = np.log(df_learn_2['Expected'])

X_test_2 = df_test_2[learn_cols]
Y_test_2 = np.log(df_test_2['Expected'])

print('shape learn, test:', len(X_learn_2), len(X_test_2) ) 
regressor_2 = xgboost.XGBRegressor(learning_rate=0.01, n_estimators=200)

regressor_2.fit(X_learn_2, Y_learn_2)

Y_2_hat = regressor_2.predict(X_test_2)
df_test_2['y_hat'] = np.exp(Y_2_hat)

score = metrics.mean_absolute_error(np.exp(Y_test_2), np.exp(Y_2_hat))
score_median = metrics.mean_absolute_error(np.exp(Y_test_2), [np.exp(np.median(Y_test_2))] * len(Y_test_2))
print("CLASS 2: score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)


# bring it all together
df_test_local = pd.concat([df_test_0, df_test_1, df_test_2], axis=0)

############################
########### SCORES #########
############################

print('test ids ', df_test_local['Id'].nunique())
print("learn ids: ", df_learn_local['Id'].nunique())

print(' -- ALL -- ')
score = metrics.mean_absolute_error(df_test_local['Expected'], df_test_local['y_hat'])
score_median = metrics.mean_absolute_error(df_test_local['Expected'], [df_learn_local['Expected'].median()] * df_test_local['Id'].nunique())
print("score: ", score, " -- score median", score_median, " -- ratio ", (score_median - score) / score_median)

print(' -- OUTLIERS -- ')
score_high = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] > outlier]['Expected'], df_test_local[df_test_local['Expected'] > outlier]['y_hat'])
score_median_high = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] > outlier]['Expected'], [df_test_local[df_test_local['Expected'] > outlier]['Expected'].median()] * df_test_local[df_test_local['Expected'] > outlier]['Id'].nunique())
print("score: ", score_high, " -- score median", score_median_high, " -- ratio ", (score_median_high - score_high) / score_median_high, ' -- perc', round(100 * sum(df_test_local['Expected'] > outlier) / len(df_test_local), 2)  )

print(' -- NORMAL -- ')
score_normal = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] < outlier]['Expected'], df_test_local[df_test_local['Expected'] < outlier]['y_hat'])
score_median_normal = metrics.mean_absolute_error(df_test_local[df_test_local['Expected'] < outlier]['Expected'], [df_test_local[df_test_local['Expected'] < outlier]['Expected'].median()] * df_test_local[df_test_local['Expected'] < outlier]['Id'].nunique())
print("score: ", score_normal, " -- score median", score_median_normal, " -- ratio ", (score_median_normal - score_normal) / score_median_normal, ' -- perc', round(100 * sum(df_test_local['Expected'] < outlier) / len(df_test_local), 2))

