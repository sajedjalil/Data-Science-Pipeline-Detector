# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Requirements Setup
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import gc
import xgboost as xgb
from sklearn.model_selection import KFold
import scipy as sp
from sklearn import metrics
from tsfresh.feature_extraction import feature_calculators

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import model_selection

#Features generator class
class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = '../input/train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('../input/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                del df
                yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values[-self.chunk_size:]
                del df
                yield seg_id, x, -999

    def features(self, x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # create features here
        # numpy
        feature_dict['mean'] = np.mean(x)
        feature_dict['max'] = np.max(x)
        feature_dict['min'] = np.min(x)
        feature_dict['std'] = np.std(x)
        feature_dict['var'] = np.var(x)
        feature_dict['ptp'] = np.ptp(x)
        feature_dict['percentile_10'] = np.percentile(x, 10)
        feature_dict['percentile_20'] = np.percentile(x, 20)
        feature_dict['percentile_30'] = np.percentile(x, 30)
        feature_dict['percentile_40'] = np.percentile(x, 40)
        feature_dict['percentile_50'] = np.percentile(x, 50)
        feature_dict['percentile_60'] = np.percentile(x, 60)
        feature_dict['percentile_70'] = np.percentile(x, 70)
        feature_dict['percentile_80'] = np.percentile(x, 80)
        feature_dict['percentile_90'] = np.percentile(x, 90)

        # scipy
        feature_dict['skew'] = sp.stats.skew(x)
        feature_dict['kurtosis'] = sp.stats.kurtosis(x)
        feature_dict['kstat_1'] = sp.stats.kstat(x, 1)
        feature_dict['kstat_2'] = sp.stats.kstat(x, 2)
        feature_dict['kstat_3'] = sp.stats.kstat(x, 3)
        feature_dict['kstat_4'] = sp.stats.kstat(x, 4)
        feature_dict['moment_1'] = sp.stats.moment(x, 1)
        feature_dict['moment_2'] = sp.stats.moment(x, 2)
        feature_dict['moment_3'] = sp.stats.moment(x, 3)
        feature_dict['moment_4'] = sp.stats.moment(x, 4)
        
        feature_dict['abs_energy'] = feature_calculators.abs_energy(x)
        feature_dict['abs_sum_of_changes'] = feature_calculators.absolute_sum_of_changes(x)
        feature_dict['count_above_mean'] = feature_calculators.count_above_mean(x)
        feature_dict['count_below_mean'] = feature_calculators.count_below_mean(x)
        feature_dict['mean_abs_change'] = feature_calculators.mean_abs_change(x)
        feature_dict['mean_change'] = feature_calculators.mean_change(x)
        feature_dict['var_larger_than_std_dev'] = feature_calculators.variance_larger_than_standard_deviation(x)
        feature_dict['range_minf_m4000'] = feature_calculators.range_count(x, -np.inf, -4000)
        feature_dict['range_m4000_m3000'] = feature_calculators.range_count(x, -4000, -3000)
        feature_dict['range_m3000_m2000'] = feature_calculators.range_count(x, -3000, -2000)
        feature_dict['range_m2000_m1000'] = feature_calculators.range_count(x, -2000, -1000)
        feature_dict['range_m1000_0'] = feature_calculators.range_count(x, -1000, 0)
        feature_dict['range_0_p1000'] = feature_calculators.range_count(x, 0, 1000)
        feature_dict['range_p1000_p2000'] = feature_calculators.range_count(x, 1000, 2000)
        feature_dict['range_p2000_p3000'] = feature_calculators.range_count(x, 2000, 3000)
        feature_dict['range_p3000_p4000'] = feature_calculators.range_count(x, 3000, 4000)
        feature_dict['range_p4000_pinf'] = feature_calculators.range_count(x, 4000, np.inf)

        feature_dict['ratio_unique_values'] = feature_calculators.ratio_value_number_to_time_series_length(x)
        feature_dict['first_loc_min'] = feature_calculators.first_location_of_minimum(x)
        feature_dict['first_loc_max'] = feature_calculators.first_location_of_maximum(x)
        feature_dict['last_loc_min'] = feature_calculators.last_location_of_minimum(x)
        feature_dict['last_loc_max'] = feature_calculators.last_location_of_maximum(x)
        feature_dict['time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
        feature_dict['time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)
        feature_dict['time_rev_asym_stat_1000'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1000)
        feature_dict['autocorrelation_5'] = feature_calculators.autocorrelation(x, 5)
        feature_dict['autocorrelation_10'] = feature_calculators.autocorrelation(x, 10)
        feature_dict['autocorrelation_50'] = feature_calculators.autocorrelation(x, 50)
        feature_dict['autocorrelation_100'] = feature_calculators.autocorrelation(x, 100)
        feature_dict['autocorrelation_1000'] = feature_calculators.autocorrelation(x, 1000)
        feature_dict['c3_5'] = feature_calculators.c3(x, 5)
        feature_dict['c3_10'] = feature_calculators.c3(x, 10)
        feature_dict['c3_100'] = feature_calculators.c3(x, 100)
        feature_dict['fft_1_real'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'real'}]))[0][1]
        feature_dict['fft_1_imag'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'imag'}]))[0][1]
        feature_dict['fft_1_ang'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 1, 'attr': 'angle'}]))[0][1]
        feature_dict['fft_2_real'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'real'}]))[0][1]
        feature_dict['fft_2_imag'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'imag'}]))[0][1]
        feature_dict['fft_2_ang'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 2, 'attr': 'angle'}]))[0][1]
        feature_dict['fft_3_real'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'real'}]))[0][1]
        feature_dict['fft_3_imag'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'imag'}]))[0][1]
        feature_dict['fft_3_ang'] = list(feature_calculators.fft_coefficient(x, [{'coeff': 3, 'attr': 'angle'}]))[0][1]
        feature_dict['long_strk_above_mean'] = feature_calculators.longest_strike_above_mean(x)
        feature_dict['long_strk_below_mean'] = feature_calculators.longest_strike_below_mean(x)
        feature_dict['cid_ce_0'] = feature_calculators.cid_ce(x, 0)
        feature_dict['cid_ce_1'] = feature_calculators.cid_ce(x, 1)
        feature_dict['binned_entropy_5'] = feature_calculators.binned_entropy(x, 5)
        feature_dict['binned_entropy_10'] = feature_calculators.binned_entropy(x, 10)
        feature_dict['binned_entropy_20'] = feature_calculators.binned_entropy(x, 20)
        feature_dict['binned_entropy_50'] = feature_calculators.binned_entropy(x, 50)
        feature_dict['binned_entropy_80'] = feature_calculators.binned_entropy(x, 80)
        feature_dict['binned_entropy_100'] = feature_calculators.binned_entropy(x, 100)

        feature_dict['num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)
        feature_dict['num_peaks_10'] = feature_calculators.number_peaks(x, 10)
        feature_dict['num_peaks_50'] = feature_calculators.number_peaks(x, 50)
        feature_dict['num_peaks_100'] = feature_calculators.number_peaks(x, 100)
        feature_dict['num_peaks_500'] = feature_calculators.number_peaks(x, 500)

        feature_dict['spkt_welch_density_1'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 1}]))[0][1]
        feature_dict['spkt_welch_density_10'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 10}]))[0][1]
        feature_dict['spkt_welch_density_50'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 50}]))[0][1]
        feature_dict['spkt_welch_density_100'] = list(feature_calculators.spkt_welch_density(x, [{'coeff': 100}]))[0][1]

        feature_dict['time_rev_asym_stat_1'] = feature_calculators.time_reversal_asymmetry_statistic(x, 1)
        feature_dict['time_rev_asym_stat_10'] = feature_calculators.time_reversal_asymmetry_statistic(x, 10)
        feature_dict['time_rev_asym_stat_100'] = feature_calculators.time_reversal_asymmetry_statistic(x, 100)        

        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs,
                       backend='threading')(delayed(self.features)(x, y, s)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)
        
# Generate Train data with Features
if (not os.path.exists("train.pkl")):
  training_fg = FeatureGenerator(dtype='train', n_jobs=10, chunk_size=150000)
  training_data = training_fg.generate()
  training_data.to_pickle("train.pkl")
else:
  training_data = pd.read_pickle("train.pkl")
  
# Generate Test data with Features
if (not os.path.exists("test.pkl")):
  test_fg = FeatureGenerator(dtype='test', n_jobs=10, chunk_size=150000)
  test_data = test_fg.generate()
  test_data.to_pickle("test.pkl")
else:
  test_data = pd.read_pickle("test.pkl")
  
# Extract X and y from train data
X = training_data.drop(['target', 'seg_id'], axis=1)
X_test = test_data.drop(['target', 'seg_id'], axis=1)
test_segs = test_data.seg_id
y = training_data.target

#Cross Validator Functions and Hyper Parameter Optimizations.
#import required packages
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
#optional but advised
import warnings
warnings.filterwarnings('ignore')

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 100 #number of hyperopt evaluation rounds
N_FOLDS = 3 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

#XGBOOST PARAMETERS
XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting
XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost
EVAL_METRIC_XGB_REG = 'mae' #XGBoost regression metric
EVAL_METRIC_XGB_CLASS = 'auc' #XGBoost classification metric

#CATBOOST PARAMETERS
CB_MAX_DEPTH = 6 #maximum tree depth in CatBoost
OBJECTIVE_CB_REG = 'RMSE' #CatBoost regression metric
OBJECTIVE_CB_CLASS = 'Logloss' #CatBoost classification metric

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False, cat_features=[]):
    
    #==========
    #LightGBM
    #==========
    
    if package=='lgbm':
        
        print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth',
                         'num_leaves',
                          'max_bin',
                         'min_data_in_leaf',
                         'min_data_in_bin']
        
        def objective(space_params):
            
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
            
            #extract nested conditional parameters
            if space_params['boosting']['boosting'] == 'goss':
                top_rate = space_params['boosting'].get('top_rate')
                other_rate = space_params['boosting'].get('other_rate')
                #0 <= top_rate + other_rate <= 1
                top_rate = max(top_rate, 0)
                top_rate = min(top_rate, 0.5)
                other_rate = max(other_rate, 0)
                other_rate = min(other_rate, 0.5)
                space_params['top_rate'] = top_rate
                space_params['other_rate'] = other_rate
            
            subsample = space_params['boosting'].get('subsample', 1.0)
            space_params['boosting'] = space_params['boosting']['boosting']
            space_params['subsample'] = subsample
            
            #for classification, set stratified=True and metrics=EVAL_METRIC_LGBM_CLASS
            cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                                early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, seed=42)
            
            best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['auc-mean'][-1]
            #if necessary, replace 'auc-mean' with '[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = lgb.Dataset(data, labels)
                
        #integer and string parameters, used with hp.choice()
        boosting_list = [{'boosting': 'gbdt',
                          'subsample': hp.uniform('subsample', 0.5, 1)},
                         {'boosting': 'goss',
                          'subsample': 1.0,
                         'top_rate': hp.uniform('top_rate', 0, 0.5),
                         'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        #for classification comment out the line above and uncomment the line below
        #modify as required for other classification metrics classification
        #metric_list = ['auc']
        objective_list_reg = ['huber', 'gamma', 'fair', 'tweedie']
        objective_list_class = ['logloss', 'cross_entropy']
        #for classification set objective_list = objective_list_class
        objective_list = objective_list_reg

        space ={'boosting' : hp.choice('boosting', boosting_list),
                'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
                'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
                'max_bin': hp.quniform('max_bin', 32, 255, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
                'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
                'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
                'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'metric' : hp.choice('metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
                'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)
            }
        
        #optional: activate GPU for LightGBM
        #follow compilation steps here:
        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
        #then uncomment lines below:
        #space['device'] = 'gpu'
        #space['gpu_platform_id'] = 0,
        #space['gpu_device_id'] =  0

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
                
        #fmin() will return the index of values chosen from the lists/arrays in 'space'
        #to obtain actual values, index values are used to subset the original lists/arrays
        best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
        best['metric'] = metric_list[best['metric']]
        best['objective'] = objective_list[best['objective']]
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
            
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #XGBoost
    #==========
    
    if package=='xgb':
        
        print('Running {} rounds of XGBoost parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth']
        
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract multiple nested tree_method conditional parameters
            #libera te tutemet ex inferis
            if space_params['tree_method']['tree_method'] == 'hist':
                max_bin = space_params['tree_method'].get('max_bin')
                space_params['max_bin'] = int(max_bin)
                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                    grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                    space_params['grow_policy'] = grow_policy
                    space_params['tree_method'] = 'hist'
                else:
                    max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                    space_params['grow_policy'] = 'lossguide'
                    space_params['max_leaves'] = int(max_leaves)
                    space_params['tree_method'] = 'hist'
            else:
                space_params['tree_method'] = space_params['tree_method'].get('tree_method')
                
            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS
            cv_results = xgb.cv(space_params, train, nfold=N_FOLDS, metrics=[EVAL_METRIC_XGB_REG],
                             early_stopping_rounds=100, stratified=False, seed=42)
            
            best_loss = cv_results['test-mae-mean'].iloc[-1] #or 'test-rmse-mean' if using RMSE
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = 1 - cv_results['test-auc-mean'].iloc[-1]
            #if necessary, replace 'test-auc-mean' with 'test-[your-preferred-metric]-mean'
            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = xgb.DMatrix(data, labels)
        
        #integer and string parameters, used with hp.choice()
        boosting_list = ['gbtree', 'gblinear'] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['mae', 'rmse'] 
        #for classification comment out the line above and uncomment the line below
        #metric_list = ['auc']
        #modify as required for other classification metrics classification
        
        tree_method = [{'tree_method' : 'exact'},
               {'tree_method' : 'approx'},
               {'tree_method' : 'hist',
                'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                'grow_policy' : {'grow_policy': {'grow_policy':'depthwise'},
                                'grow_policy' : {'grow_policy':'lossguide',
                                                  'max_leaves': hp.quniform('max_leaves', 32, XGB_MAX_LEAVES, 1)}}}]
        
        #if using GPU, replace 'exact' with 'gpu_exact' and 'hist' with
        #'gpu_hist' in the nested dictionary above
        
        objective_list_reg = ['reg:linear', 'reg:gamma', 'reg:tweedie']
        objective_list_class = ['reg:logistic', 'binary:logistic']
        #for classification change line below to 'objective_list = objective_list_class'
        objective_list = objective_list_reg
        
        space ={'boosting' : hp.choice('boosting', boosting_list),
                'tree_method' : hp.choice('tree_method', tree_method),
                'max_depth': hp.quniform('max_depth', 2, XGB_MAX_DEPTH, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
                'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
                'gamma' : hp.uniform('gamma', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'eval_metric' : hp.choice('eval_metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),
                'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                'nthread' : -1
            }
        
        #optional: activate GPU for XGBoost
        #uncomment line below
        #space['tree_method'] = 'gpu_hist'
        
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        best['tree_method'] = tree_method[best['tree_method']]['tree_method']
        best['boosting'] = boosting_list[best['boosting']]
        best['eval_metric'] = metric_list[best['eval_metric']]
        best['objective'] = objective_list[best['objective']]
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        if 'max_bin' in best:
            best['max_bin'] = int(best['max_bin'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    #==========
    #CatBoost
    #==========
    
    if package=='cb':
        
        print('Running {} rounds of CatBoost parameter optimisation:'.format(num_evals))
        
        #clear memory 
        gc.collect()
            
        integer_params = ['depth',
                          'one_hot_max_size', #for categorical data
                          'min_data_in_leaf',
                          'max_bin']
        
        def objective(space_params):
                        
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            #extract nested conditional parameters
            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                space_params['bagging_temperature'] = bagging_temp
                
            if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = space_params['grow_policy'].get('max_leaves')
                space_params['max_leaves'] = int(max_leaves)
                
            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']
                           
            #random_strength cannot be < 0
            space_params['random_strength'] = max(space_params['random_strength'], 0)
            #fold_len_multiplier cannot be < 1
            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)
                       
            #for classification set stratified=True
            cv_results = cb.cv(train, space_params, fold_count=N_FOLDS, 
                             early_stopping_rounds=25, stratified=False, partition_random_seed=42)
           
            #best_loss = cv_results['test-MAE-mean'].iloc[-1] 
            best_loss = cv_results['test-RMSE-mean'].iloc[-1] 
            
            #for classification, comment out the line above and uncomment the line below:
            #best_loss = cv_results['test-Logloss-mean'].iloc[-1]
            #if necessary, replace 'test-Logloss-mean' with 'test-[your-preferred-metric]-mean'
            
            return{'loss':best_loss, 'status': STATUS_OK}
        
        train = cb.Pool(data, labels.astype('float32'), cat_features=cat_features)
        
        #integer and string parameters, used with hp.choice()
        bootstrap_type = [
                          {'bootstrap_type':'Poisson'}, 
                          {'bootstrap_type':'Bayesian', 'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
                          {'bootstrap_type':'Bernoulli'}] 
        LEB = ['No', 'AnyImprovement', 'Armijo'] #remove 'Armijo' if not using GPU
        #score_function = ['Correlation', 'L2', 'NewtonCorrelation', 'NewtonL2']
        grow_policy = [{'grow_policy':'SymmetricTree'},
                       {'grow_policy':'Depthwise'},
                       {'grow_policy':'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]
        eval_metric_list_reg = ['MAE', 'RMSE', 'Poisson']
        eval_metric_list_class = ['Logloss', 'AUC', 'F1']
        #for classification change line below to 'eval_metric_list = eval_metric_list_class'
        eval_metric_list = eval_metric_list_reg
                
        space ={'depth': hp.quniform('depth', 2, CB_MAX_DEPTH, 1),
                'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254
                #'max_bin': 254,
                'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5),
                'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
                'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
                'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
                'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
                'learning_rate' : hp.uniform('learning_rate', 0.05, 0.25),
                'eval_metric' : hp.choice('eval_metric', eval_metric_list),
                'objective' : OBJECTIVE_CB_REG,
                #'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown
                'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
                'grow_policy': hp.choice('grow_policy', grow_policy),
                #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
                'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
                'od_type' : 'Iter',
                'od_wait' : 25,
                'task_type' : 'GPU',
                'verbose' : 0,
                'cat_features': cat_features
            }
        
        #optional: run CatBoost without GPU
        #uncomment line below
        #space['task_type'] = 'CPU'
            
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
        
        #unpack nested dicts first
        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
        best['eval_metric'] = eval_metric_list[best['eval_metric']]
        
        #best['score_function'] = score_function[best['score_function']] 
        #best['leaf_estimation_method'] = LEM[best['leaf_estimation_method']] #CPU only
        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    else:
        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "cb" for CatBoost.')

if not os.path.exists('lgbm_params.npy'):
    lgbm_params = quick_hyperopt(X, y, 'lgbm', 1500)
    np.save('lgbm_params.npy', lgbm_params)
else:
    lgbm_params = np.load('lgbm_params.npy', allow_pickle=True)
    lgbm_params = dict(lgbm_params.tolist())
    
print("lgbm_params", lgbm_params)

if not os.path.exists('xgb_params.npy'):
    xgb_params = quick_hyperopt(X, y, 'xgb', 2500)
    np.save('xgb_params.npy', xgb_params)
else:
    xgb_params = np.load('xgb_params.npy', allow_pickle=True)
    xgb_params = dict(xgb_params.tolist())
    
print("xgb_params", xgb_params)
    
if not os.path.exists('cb_params.npy'):
    cb_params = quick_hyperopt(X, y, 'cb', 15)
    np.save('cb_params.npy', cb_params)
else:
    cb_params = np.load('cb_params.npy', allow_pickle=True)
    cb_params = dict(cb_params.tolist())
    
print("cb_params", cb_params)

##TRAIN MODELS
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)

print("Train LGB")
model = lgb.train(lgbm_params, lgb.Dataset(X_train,y_train), num_boost_round=20_000, valid_sets=lgb.Dataset(X_valid,y_valid), early_stopping_rounds=2000, verbose_eval=True)
test_preds = model.predict(X_test)
submission = pd.DataFrame(columns=['seg_id', 'time_to_failure'])
submission.seg_id = test_segs
submission.time_to_failure = test_preds
submission.to_csv('submission_lgb.csv', index=False, line_terminator='\n')

print("Train XGB")
model = xgb.XGBModel(**xgb_params)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True)
test_preds = model.predict(X_test)
submission = pd.DataFrame(columns=['seg_id', 'time_to_failure'])
submission.seg_id = test_segs
submission.time_to_failure = test_preds
submission.to_csv('submission_xgb.csv', index=False, line_terminator='\n')

print("Train Catboost")
model = CatBoostRegressor(**cb_params, task_type='GPU')
model.fit(X_train, y_train) 
test_preds = model.predict(X_test)
submission = pd.DataFrame(columns=['seg_id', 'time_to_failure'])
submission.seg_id = test_segs
submission.time_to_failure = test_preds
submission.to_csv('submission_cat.csv', index=False, line_terminator='\n')