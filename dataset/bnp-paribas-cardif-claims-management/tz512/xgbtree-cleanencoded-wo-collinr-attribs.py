# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import  StratifiedShuffleSplit
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.


def clean_encode_data(train, test, attribs_todrop=None):

    float_imputer = Imputer(missing_values='NaN', strategy='mean')

    if attribs_todrop is not None:

        print('Dropping attributes with no influence on the target variable:\n')
        print('%s\n' % attribs_todrop)

        # dropping unimportant attributes from data sets
        train.drop(attribs_todrop, axis=1, inplace=True)
        test.drop(attribs_todrop, axis=1, inplace=True)

    # for nominal categorical variables:
    # create a pair of empty pandas DFs to keep dummies values
    train_dummies = pd.DataFrame(index=train.index)
    test_dummies = pd.DataFrame(index=test.index)

    print('Clean data...\n')
    print('Encoding the nominal (ordinal) categorical variables...\n')
    features = train.columns
    for col in features:
        # encoding nominal categorical variables 
        # OneHotEncoding
        if train[col].dtype == 'O':
            print('Encoding the nominal categorical variable \'%s\'.' % col)
            train_col_dummies = pd.get_dummies(train[col], prefix=col, prefix_sep='.', dummy_na=True)
            train_dummies = train_dummies.join(train_col_dummies, how='left')
            train.drop(col, axis=1, inplace=True)
            test_col_dummies = pd.get_dummies(test[col], prefix=col, prefix_sep='.', dummy_na=True)
            test_dummies = test_dummies.join(test_col_dummies, how='left')
            test.drop(col, axis=1, inplace=True)
            print('\n')
        else:
            # encoding ordinal categorical variables
            # pandas.factorize
            if train[col].dtype == 'int64':
                print('Encode the ordinal categorical variable \'%s\'.' % col)
                train[col], tmp_indexer = pd.factorize(train[col], na_sentinel=0)
                test[col] = tmp_indexer.get_indexer(test[col])
                print('\n')
            # imputing float means
            # sklearn-Imputer
            if train[col].dtype == 'float64':
                print('Imputing the \'na\' values of the float variable \'%s\' with its mean.' % col)
                train_col = np.array(train[[col]])
                train[col] = float_imputer.fit_transform(train_col)
                print('Imputing the \'na\' values of the float variable \'%s\' with its mean.' % col)
                test_col = np.array(test[[col]])
                test[col] = float_imputer.transform(test_col)
                print('\n')

    features = train_dummies.columns.join(test_dummies.columns, how='inner')
    train = pd.concat([train, train_dummies[features]], axis=1, copy=True)
    test = pd.concat([test, test_dummies[features]], axis=1, copy=True)

    print('The categorical variables of the data sets have been encoded successfully!\n')
    print('The \'na\' values of the float variables have been imputed (mean) successfully!\n')

    return train, test


def remove_collinear_predictors(train, test, attribs_of_interest, threshold=0.9):

    # REMOVE COLLINEAR PREDICTORS
    """
    :type train: pandas DataFrame
    :type test: pandas DataFrame
    """
    print('Determining the collinear float attributes...\n')
    print('Collinearity Threshold: %.2f\n' % threshold)

    # select the train/test subset of dtype floats
    train_float = train[attribs_of_interest].copy()
    test_float = test[attribs_of_interest].copy()

    # compute the pair-wise correlations for the attributes in data sets
    train_corrs = train_float.corr()
    test_corrs = test_float.corr()

    # determine the high correlated for the given threshold
    train_high_corrs = train_corrs[(abs(train_corrs) > threshold) & (train_corrs != 1)]
    test_high_corrs = test_corrs[(abs(test_corrs) > threshold) & (test_corrs != 1)]

    # verify that for the chosen collinearity threshold
    # the same subset of attributes will be removed from train and test data sets
    train_boolean_high_corrs = train_high_corrs.notnull()
    test_boolean_high_corrs = test_high_corrs.notnull()

    if train_boolean_high_corrs.equals(test_boolean_high_corrs):

        collinear_features = pd.Series()
        features = train_boolean_high_corrs.columns

        for col in features:
            collinear_features_tmp = (train_boolean_high_corrs.index[train_boolean_high_corrs[col]]).to_series()
            collinear_features = collinear_features.append(collinear_features_tmp)

        collinear_features.drop_duplicates(inplace=True)

        print('Collinear Predictors:')
        print('------------------------')
        print(list(collinear_features))
        print('\n')
        print('Removing the collinear predictors from both train and test data set...\n')
        train_float.drop(collinear_features, axis=1, inplace=True)
        test_float.drop(collinear_features, axis=1, inplace=True)
        # keep the non float variables of the original data sets
        # in a different pair of Pandas DataFrames
        features_non_float_train = train.columns.difference(attribs_of_interest)
        features_non_float_test = test.columns.difference(attribs_of_interest)
        train_non_float = train[features_non_float_train].copy()
        test_non_float = test[features_non_float_test].copy()
        # returns the original data sets without the observed collinear predictors
        train = pd.concat([train_float, train_non_float], axis=1)
        test = pd.concat([test_float, test_non_float], axis=1)
        print('The collinear predictors have been removed successfully!\n')

    else:

        print('A different subset of attributes will be removed in train and test set.')
        print('Please choose a different collinearity threshold!\n')
        pass

    return train, test


if __name__ == '__main__':

    print('Start\n')
    print('Loading Data...\n')
    # training data
    train = pd.read_csv('../input/train.csv')
    target = train['target'].values
    train = train.drop(['ID', 'target'], axis=1)
    # test data (awaiting predictions)
    test = pd.read_csv('../input/test.csv')
    test_ID = test['ID'].values
    test = test.drop(['ID'], axis=1)

    # keep the the attributes names of the various dtypes in separate lists
    features_categorical = list(train.select_dtypes(include=['O', 'int64']).columns)
    features_nominal_categorical = list(train.select_dtypes(include=['O']).columns)
    features_ordinal_categorical = list(train.select_dtypes(include=['int64']).columns)
    features_float = list(train.select_dtypes(include=['float64']).columns)

    # floats with no decisive influence on 'target' variable:
    attribs_todrop_float = ['v1', 'v5', 'v9', 'v11', 'v13', 'v15', 'v16', 
    'v23', 'v25', 'v28', 'v29', 'v32', 'v35', 'v42', 'v53', 'v54', 'v57',
    'v59', 'v60', 'v67', 'v70', 'v77', 'v78', 'v82', 'v86', 'v89', 'v90', 
    'v92', 'v94', 'v95', 'v96', 'v103', 'v104', 'v105', 'v111', 'v115', 
    'v117', 'v118', 'v120', 'v122', 'v124', 'v126', 'v127']
    attribs_todrop_categorical = ['v22', 'v62', 'v72', 'v129', 'v3', 'v52', 
    'v112', 'v125', 'v74']
    attribs_todrop = attribs_todrop_float+attribs_todrop_categorical

    print('Applying the clean_encode_data() function...\n')
    train, test = clean_encode_data(train, test, attribs_todrop=attribs_todrop)

    print('Applying the remove_collinear_predictors() function...\n')
    features_float_new = features_float
    for var in attribs_todrop_float:
        features_float_new.remove(var)
    train, test = remove_collinear_predictors(train, test,
                                              features_float_new, threshold=0.97)
                                              
    # stratifiedShuffleSplit OF THE TRAINING SET IN A TRAIN AND A VALIDATION PART
    # 90%-10% OF KNOWN EXAMPLES
    print('Stratified Shuffle Split of the training set in a train and a validation data set...\n')
    train_copy = np.array(train.copy())
    target_copy = np.array(target.copy())
    
    sss = StratifiedShuffleSplit(target_copy, n_iter=1, test_size=0.10, random_state=1)
    for train_idx, valdt_idx in sss:
        X_train, X_valdt = train_copy[train_idx], train_copy[valdt_idx]
        y_train, y_valdt = target_copy[train_idx], target_copy[valdt_idx]
    del train_copy, target_copy
    
    X_train = pd.DataFrame(X_train, columns=train.columns)
    X_valdt = pd.DataFrame(X_valdt, columns=train.columns)
    
    print('The train data set [90% of known Examples]: (X_train, y_train)\n')
    print('The validation data set [10% of known Examples]: (X_valdt, y_valdt)\n')
    
    # PREPARE THE TRAINING/VALIDATION/TEST DATA SET FOR XGBoost ALGO [XGB DMATRICES]
    X_train_csr = csr_matrix(np.array(X_train))
    y_train_np = np.array(y_train)
    dtrain = xgb.DMatrix(X_train_csr, y_train_np, silent=True)
    del X_train_csr, y_train_np
    
    X_valdt_csr = csr_matrix(np.array(X_valdt))
    y_valdt_np = np.array(y_valdt)
    dvaldt = xgb.DMatrix(X_valdt_csr, y_valdt_np, silent=True)
    del X_valdt_csr, y_valdt_np
    
    test_csr = csr_matrix(np.array(test))
    dtest = xgb.DMatrix(test_csr, silent=True)
    del test_csr

    # SET THE FIXED XGBoostTree PARAMETERS
    params = {}
    # general parameters
    params['booster'] = 'gbtree'
    params['silent'] = 0
    params['objective'] = 'binary:logistic'

    # gbtree parameters
    
    params['eta'] = 0.015 # 0.01 [exceeds max runtime limit (1200secs)]
    params['gamma'] = 1
    params['lambda'] = 1
    params['alpha'] = 0
    params['max_depth'] = 10
    params['max_delta_step'] = 1
    params['min_child_weight'] = 0.3
    params['colsample_bylevel'] = 0.5
    params['tree_method'] = 'auto'
    params['eval_metric'] = 'logloss'
    
    print('Set the XGBoostTree parameters:')
    print('----------------------------------')
    print(params)
    print('\n')

    # VALIDATE THE CHOSEN PARAMETERS AGAINST THE dvaldt DATA SET
    print('Training/Validating the XGBoostTree...\n')
    
    watchlist = [(dtrain, 'train'), (dvaldt, 'eval')]
    
    xgbtree = xgb.train(params, dtrain, num_boost_round=10000,
    evals=watchlist, early_stopping_rounds=50, evals_result=None,
    verbose_eval=True, learning_rates=None, xgb_model=None)

    # PROVIDE THE ACTUAL PREDICTIONS
    print('Providing the actual predictions...\n')
    test_pred = xgbtree.predict(dtest)
    
    print('Start Output\n')
    print('Preparing my submission file...\n')
    tgrammat_submission = pd.DataFrame({"ID": test_ID, "PredictedProb": test_pred})
    # create my submission .csv file
    tgrammat_submission_filename = str('tgrammat_submission.csv')
    tgrammat_submission.to_csv(tgrammat_submission_filename, index=False)
    print('The submission file has been stored.\n')
    
    print('Finish')