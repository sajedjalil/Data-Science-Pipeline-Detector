import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold



train_df = pd.read_csv('../input/train.csv', nrows = 50000)
X = train_df.copy().drop(['ID_code','target'], axis=1)
y = train_df.target


bayes_cv_tuner = BayesSearchCV(estimator = xgb.XGBClassifier(
                                n_jobs = -1,
                                objective = 'binary:logistic',
                                eval_metric = 'auc',
                                learning_rate = 0.1,
                                silent=1,
                                early_stopping = 200,
                                n_estimators = 8000,
                                tree_method='approx'),
    search_spaces = {
        'min_child_weight': (15, 20),
        'max_depth': (6, 8),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-2, 1000, 'log-uniform'),
        'reg_alpha': (1e-2, 1.0, 'log-uniform'),
        'gamma': (1e-2, 0.5, 'log-uniform'),
        'min_child_weight': (0, 20),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42),
    n_jobs = 3,
    n_iter = 10,   
    verbose = 500,
    refit = True,
    random_state = 786)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")
    
result = bayes_cv_tuner.fit(X.values, y.values, callback=status_print)

# Source OG  - https://www.kaggle.com/nanomathias/bayesian-optimization-of-xgboost-lb-0-9769