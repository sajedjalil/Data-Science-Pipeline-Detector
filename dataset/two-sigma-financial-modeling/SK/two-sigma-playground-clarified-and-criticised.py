# This script is a clarified and criticised version of Trottefox's acclaimed "Two sigma
# playground" kernel (half train holdout 0.020876, pub LB 0.0149618).
# 
# (It may very well have a long history of forks before that and contributions by others,
# I did not follow the source trail; apologies if I ought to add credits.)
# 
# I (SK) am not really competing in this competition, but want to contribute my two cents
# this way. In my opinion the original script is (1) unnecessarily unclear due mostly to
# lack of comments and unfortunate variable naming, and (2) is methodologically unsound,
# reinventing many wheels but making them square, triangle, and elliptic instead of
# circular. This does a disservice to novices and as a result to our fine community.
# 
# One general thing to keep in mind here is that there is very low SNR in this problem,
# and it echos loudly in the evaluation metric (small changes to model can wildly change
# even the half-trainset validation score). On top of that, nonstationarity is a big
# issue, and so behavior in the second half of the trainset, the public leaderboard, and
# (supposedly) the private leaderboard is different. So do yourselves a big favor and
# don't take results in validation / public too seriously as predictors of the private
# score. The public ranking of this script will have much more to do with the correlation
# between competitors' solutions than its intrinsic quality. All that glitters is not
# gold.
# 
# All of this is not to say that the original script doesn't include several nice
# elementary ideas that are good to know and use. Let's just try to make them more
# accessible.

import kagglegym
import numpy as np
import pandas as pd
import random
from sklearn import ensemble, linear_model, metrics

# Configuration
# ========================================================================================

add_na_indicators = True
add_diff_features = True
na_indicator_cols = ['technical_9', 'technical_0', 'technical_32', 'technical_16', 
    'technical_38', 'technical_44', 'technical_20', 'technical_30', 'technical_13'] 
    # allegedly selected by tree-based algorithms
diff_cols = ['technical_22', 'technical_20', 'technical_30', 'technical_13', 
    'technical_34'] # also allegedly selected by tree-based algorithms
univar_rlm_cols = ['technical_22', 'technical_20', 'technical_30_d1', 'technical_20_d1',
    'technical_30', 'technical_13', 'technical_34']
nr_l2_best_models = 10
wrlm_quant = 0.99
wrlm_min_trainset_fraction = 0.9
wslm_n_covar_sets = 30
wslm_max_nr_covars_per_set = 2
wslm_max_abs_y = 0.086
l1_et_n_estimators = 100
l1_et_max_depth = 4
l3_et_n_estimators = 100
l3_et_max_depth = 4
rnd_seed = 17

# Helper functions and objects
# ========================================================================================

# FIXME: This is a weird reinvention of the stepwise regression wheel: We are going to
# greedily choose n_covar_sets of covariate sets and build linear models based on each
# set. We are going to do this in a way that is quite arbitrary and irrational. We'd
# probably be better off with something like linear XGB or lasso.

class weird_stepwise_lm:
    def __init__(self, n_covar_sets = 30, max_nr_covars_per_set = 2, verbose = True):
        self.n_covar_sets = n_covar_sets
        self.max_nr_covars_per_set = max_nr_covars_per_set
        self.verbose = verbose
        
        if self.max_nr_covars_per_set == None:
            self.max_nr_covars_per_set = len(train.columns)
        self.chosen_covar_sets = []
        self.models = []
        
    def fit(self, train, y):
        # We initialize the sets such that each one includes a single random unique feature.
        
        covar_pool = list(train.columns)
        random.shuffle(covar_pool)
        for var in covar_pool[:self.n_covar_sets]:
            self.chosen_covar_sets.append([var])
        covar_pool = covar_pool[self.n_covar_sets:]
        
        # Now we are going to do greedy forward stepwise regression in a very strange way:
        # We go over the set of covariates not chosen initially, and see to which set will
        # adding this new covariate improve the training error the most. We then add this 
        # new covariate to that set. When the maximum number of covariates 
        # max_nr_covars_per_set is reached, that set is no longer considered for adding 
        # new variables.
        
        # FIXME: it doesn't make sense to look at training error when considering the 
        # improvement achieved by adding covariates (except in degenerate cases, adding 
        # will always improve the training error, but the question is how much is useful 
        # fitting and how much is just overfitting). We should use Cp/AIC/SURE/CV to 
        # estimate out of sample error! (note that when max_nr_covars_per_set = 2, 
        # Cp/AIC/SURE and in sample error selection are in fact equivalent for the purpose
        # of deciding to which set a given covariate should be added, but not whether it
        # should be added at all!)
        
        # FIXME: I don't understand the rationale behind only considering disjoint 
        # covariate sets.
        
        # initialize to a big value any model will beat
        best_mses_by_set = np.zeros(self.n_covar_sets) + 1e15
        
        for var in covar_pool:
            ci = 0
            mses_per_candidate_covar = []
            for covar_set in self.chosen_covar_sets:
                if len(covar_set) < self.max_nr_covars_per_set:
                    model = linear_model.LinearRegression(fit_intercept = False, 
                        normalize = True, copy_X = True, n_jobs = -1) 
                    model.fit(train[covar_set + [var]], y)
                    # FIXME this is not a good measure of model quality
                    mse = metrics.mean_squared_error(y, model.predict(train[covar_set +
                        [var]]))
                    mses_per_candidate_covar.append(mse)
                else:
                    mses_per_candidate_covar.append(best_mses_by_set[ci])
                ci += 1
            
            gains = best_mses_by_set - mses_per_candidate_covar
            # currently, this will always be true (except degenerate cases), and is meant
            # for when we maxed out the max_nr_covars_per_set
            if gains.max() > 0: 
                temp = gains.argmax()
                self.chosen_covar_sets[temp].append(var)
                best_mses_by_set[temp] = mses_per_candidate_covar[temp]
        
        # Now we have grown our covariate sets, and for some reason we train the models 
        # again and now bother to store them for later.
        
        csi = 0
        for covar_set in self.chosen_covar_sets:
            model = linear_model.LinearRegression(fit_intercept = False, normalize = True, 
                copy_X = True, n_jobs = -1) 
            model.fit(train[covar_set], y)
            self.models.append(model)
            if self.verbose:
                print('Covar set', csi, 'includes', covar_set, 'and achieves', 
                    best_mses_by_set[csi])
            csi += 1

    def predict(self, data):
        csi = 0
        for covar_set in self.chosen_covar_sets:
            # Original script comment: This line generates a warning. Could be avoided by 
            # working and returning with a copy of data. kept this way for memory management
            # TODO: need to verify this always works as claimed
            data['stacked_lm' + str(csi)] = self.models[csi].predict(data[covar_set])
            csi += 1
        return data

# FIXME: and this one is a weird reinvention of robust regression / m-estimation (a very well 
# studied area in statistics that keeps resurfacing in popularity every decade or so)

class weird_robust_lm:
    def __init__(self, quant = 0.999, min_trainset_fraction = 0.9):
        self.quant = quant
        self.min_trainset_fraction = min_trainset_fraction
        self.best_model = []
       
    def fit(self, train, y):
        tmp_model = linear_model.Ridge(fit_intercept = False)
        best_mse = 1e15
        better = True
        train_idxs = train.dropna().index
        min_trainset_fraction = len(train) * self.min_trainset_fraction
        while better:
            tmp_model.fit(train.ix[train_idxs], y.ix[train_idxs])
            mse = metrics.mean_squared_error(tmp_model.predict(train.ix[train_idxs]), 
                y.ix[train_idxs])
            if mse < best_mse:
                best_mse = mse
                self.best_model = tmp_model
                residuals = y.ix[train_idxs] - tmp_model.predict(train.ix[train_idxs])
                train_idxs = residuals[abs(residuals) <= 
                    abs(residuals).quantile(self.quant)].index
                if len(train_idxs) < min_trainset_fraction:
                    better = False
            else:
                better = False
                self.best_model = tmp_model
    
    def predict(self, test):
        return self.best_model.predict(test)

# Actually do stuff
# ========================================================================================

print('Initializing')
random.seed(rnd_seed)
env = kagglegym.make()
obs = env.reset()

# Batch supervised training part
# ----------------------------------------------------------------------------------------

train = obs.train

# Obtain overall train median per column. We will use this to impute missing values.
# FIXME: in the training sample, this will look much more useful than it is out of sample 
# because it uses all future values. It may very well be that we'd be better off to compute
# a rolling median (using recent past values for each observation).
train_median = train.median(axis = 0)

print('Adding missing value counts per row')
train['nr_missing'] = train.isnull().sum(axis = 1)

print('Adding missing value indicators')
if add_na_indicators:
    for col in na_indicator_cols:
        train[col + '_isna'] = pd.isnull(train[col]).apply(lambda x: 1 if x else 0)
        if len(train[col + '_isna'].unique()) == 1:
            print('Dropped constant missingness indicator:', col, '_isna')
            del train[col + '_isna']
            na_indicator_cols.remove(col)

print('Adding diff features')
if add_diff_features:
    train = train.sort_values(by = ['id', 'timestamp'])
    for col in diff_cols:
        # FIXME: why not group by (id, ts)? why only a lag of 1 sample?
        train[col + '_d1'] = train[col].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
    train = train[train.timestamp != 0] # Drop first timestamp that had no diffs 
    #(FIXME can be confusing; why not leave missing?)

# We're going to use all of these features for modeling
base_features = [x for x in train.columns if x not in ['id', 'timestamp', 'y']]

# FIXME: the following is a combination of models in a certain multilayer graph structure
# where one layer feeds the next (or sometimes multiple following layers). Normally, this
# kind of thing should be properly stacked, otherwise all layers but the base layer are
# biased because they are trained to think that the distribution of their inputs, which
# are fitted values (think: in sample error) of previous models represents the
# distribution of the same inputs when operating on holdout data (think: of out of sample
# error). This is not true because the previous layer model perform much better on the
# very examples they were trained on, relative to holdout examples, and ignoring this
# typically leads to nasty overfitting. Note though that stacking in this time-series
# context is more involved than the usual iid situation, just like cross validation is.

print('Fitting L0 weird robust univariate linear models')
l0_models = []
l0_columns = []
l0_residuals = []
for col in univar_rlm_cols:
    print('  working on', col)
    model = weird_robust_lm(quant = wrlm_quant, min_trainset_fraction = 
        wrlm_min_trainset_fraction)
    model.fit(train.loc[:, [col]], train.loc[:, 'y'])
    l0_models.append(model)
    l0_columns.append([col])
    # FIXME in sample, should be on holdout
    l0_residuals.append(abs(model.predict(train[[col]].fillna(train_median)) - train.y))

# Impute all missing values
# FIXME this is incredibly simplistic, aim for smarter imputation (I'm not saying in this 
# problem such a thing necessarily exists).
train = train.fillna(train_median)

print('Fitting L0 weird stepwise linear model')
l0_wslm = weird_stepwise_lm(n_covar_sets = wslm_n_covar_sets, max_nr_covars_per_set = 
    wslm_max_nr_covars_per_set, verbose = True)
# drop outlying response values from the trainset. 
# FIXME this is a very naive way to handle outliers, and probably required only because of
# the wheel-reinventing implementation of robust regression used in this script
train_idx = train[abs(train.y) < wslm_max_abs_y].index
l0_wslm.fit(train.ix[train_idx, base_features], train.ix[train_idx, 'y'])
# FIXME this "prediction" is on the same set the model was trained on, and so will look
# unrealistically good to the next layer compared to how it will be on the holdout...
l0_wslm_fitted_and_base_features = l0_wslm.predict(train[base_features])
l0_wslm_fitted_and_base_features_cols = l0_wslm_fitted_and_base_features.columns

print('Training L1 ETs')
model = ensemble.ExtraTreesRegressor(n_estimators = l1_et_n_estimators, 
    max_depth = l1_et_max_depth, n_jobs = -1, random_state = rnd_seed, verbose = 0)
model.fit(l0_wslm_fitted_and_base_features, train.y)
#print('NOTE: top 30 base features that ETs found useful:')
#print(pd.DataFrame(model.feature_importances_, 
#    index = l0_wslm_fitted_and_base_features_cols).sort_values(by = [0]).tail(30))
l1_models = []
l1_columns = []
l1_residuals = []
for extra_tree in model.estimators_:
    l1_models.append(extra_tree)
    l1_columns.append(l0_wslm_fitted_and_base_features_cols)
    # FIXME in sample, should be on holdout
    l1_residuals.append(abs(extra_tree.predict(l0_wslm_fitted_and_base_features) - train.y))

# We now connect some L0 and L1 outputs as inputs to an L2 (this is perfectly fine in theory;
# in this case it will create very correlated inputs, but that too should not be a problem 
# for the type of model we use in L2)
l01_models = l0_models + l1_models
l01_columns = l0_columns + l1_columns
l01_residuals = l0_residuals + l1_residuals

print('Training L2 select_best')
# FIXME oh my... the selection criterion here doesn't reflect the eval metric. What we want 
# is the model that gives the best average score, not the "highest number of best scores 
# observation-wise". Sometimes deviating from empirical risk minimization pays off (e.g., 
# regularization; a simplistic view of it anyway), but here the rationale is unclear, and
# we can probably do better.
midxs = np.argmin(np.array(l01_residuals).T, axis = 1)
midxs = pd.Series(midxs).value_counts().head(nr_l2_best_models).index
l2_best_models = []
l2_best_model_columns = []
l2_best_model_residuals = []
for midx in midxs:
    l2_best_models.append(l01_models[midx])
    l2_best_model_columns.append(l01_columns[midx])
    l2_best_model_residuals.append(l01_residuals[midx])

# FIXME again in sample predictions that should be stacked
l2_best_model_idx = np.argmin(np.array(l2_best_model_residuals).T, axis = 1)

print('Training L3 ET')
# FIXME: this is a VERY weird beast, it doesn't look like stacking, because it uses
# predictions form L2 to define a new *target* in L3. Sometimes weird ideas work well, but
# again I'd like to have some justification for this (theoretical, or empirical in the
# form of a hypothesis test versus the tried-and-true approach) we'd probably be much
# better off doing this by combining the actual predictions from L2 models the usual way. 
# On a fundamental level, note that in most realistic problems, we don't need to "learn to
# select models"? We can just select them as we go. In this competition we don't observe
# the targets or even individual model rewards as they unfold, so this strange idea is not
# totally without merit, at least at first glance.
l3_et = ensemble.ExtraTreesClassifier(n_estimators = l3_et_n_estimators, 
    max_depth = l3_et_max_depth, n_jobs = -1, random_state = rnd_seed, verbose = 0)
l3_et.fit(l0_wslm_fitted_and_base_features, l2_best_model_idx)
#print('NOTE: top 30 base features that ET found useful:')
#print(pd.DataFrame(l3_et.feature_importances_, 
#    index = l0_wslm_fitted_and_base_features_cols).sort_values(by = [0]).tail(30))

# Prediction / online training part
# ----------------------------------------------------------------------------------------

# TODO: this doesn't exploit the rewards we get as we go for learning (though those are 
# quite crude, tbh).

print('Predicting on holdout set')
oidx = 0
nr_positive_rewards = 0
holdout_rewards = []
prev_diff_cols_data = train[train.timestamp == max(train.timestamp)][['id'] + diff_cols].copy()

while True:
    oidx += 1
    test = obs.features
    
    # Preprocess
    test['nr_missing'] = test.isnull().sum(axis = 1)
    if add_na_indicators:
        for elt in na_indicator_cols:
            test[elt + '_isna'] = pd.isnull(test[elt]).apply(lambda x: 1 if x else 0)
    
    test = test.fillna(train_median)

    if add_diff_features:
        ids_with_prev = list(set(prev_diff_cols_data.id) & set(test.id))
        prev_diff_cols_data = pd.concat([
          test[test.id.isin(ids_with_prev)]['id'], 
          pd.DataFrame(
            test[diff_cols][test.id.isin(ids_with_prev)].values - 
                prev_diff_cols_data[diff_cols][prev_diff_cols_data.id.isin(ids_with_prev)].values, 
            columns = diff_cols, index = test[test.id.isin(ids_with_prev)].index
          )
        ], axis = 1)
        # FIXME why not ZOH missing values for which we have older values?
        test = test.merge(right = prev_diff_cols_data, how = 'left', on = 'id', 
            suffixes = ('', '_d1')).fillna(0)
        prev_diff_cols_data = test[['id'] + diff_cols].copy()
    
    # Pass the data through the stacked model to generate a prediction
    l0_wslm_fitted_and_base_features = l0_wslm.predict(test[base_features]) 
    l3_preds = l3_et.predict_proba(l0_wslm_fitted_and_base_features.loc[:, 
        l0_wslm_fitted_and_base_features_cols])
    pred = obs.target
    for idx, mdl in enumerate(l2_best_models):
        pred['y'] += (l3_preds[:, idx] * 
            mdl.predict(l0_wslm_fitted_and_base_features[l2_best_model_columns[idx]]))

    # One small step for man, one giant leap for mankind
    obs, reward, done, info = env.step(pred)
    
    holdout_rewards.append(reward)
    
    if reward > 0:
        nr_positive_rewards += 1
    
    if oidx % 100 == 0:
        print('Step', oidx, '#pos', nr_positive_rewards, 'curr', reward, 'mean so far', 
            np.mean(holdout_rewards))
        
    if done:
        print('Done. Public score:', info['public_score'])
        break
