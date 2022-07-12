import csv
from math import sqrt
import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
#from datetime import datetime as dt


def CleanEncodeData(X_train, test, attribs_todrop=None):
    
    float_imputer = Imputer(missing_values='NaN', strategy='mean')
    
    X_train = X_train.copy()
    test = test.copy()
       
    if attribs_todrop is not None:
        
        print('Dropping attributes with no influence on the target variable:\n')
        print('%s\n' %attribs_todrop)
        
        ## droping unimportant attributes from data sets
        X_train.drop(attribs_todrop, axis=1, inplace=True)
        test.drop(attribs_todrop, axis=1, inplace=True)
    
    print('Clean data...\n')
    print('Encoding the nominal (ordinal) categorical variables...\n')
    features = X_train.columns
    for col in features:
        ## encoding nominal categorical variables
        ## fillna with '-1'
        if((X_train[col].dtype == 'O')):
            print('Encoding the nominal categorical variable \'%s\'.' %col)
            X_train[col], tmp_indexer = pd.factorize(X_train[col], na_sentinel=-1)
            test[col] = tmp_indexer.get_indexer(test[col])
            print('\n')
        ## encoding ordinal categorical variables
        ## fillna with '-1'
        if((X_train[col].dtype=='int64')):
            print('Encode the ordinal categorical variable \'%s\'.' %col)
            X_train[col], tmp_indexer = pd.factorize(X_train[col], na_sentinel=-1)
            test[col] = tmp_indexer.get_indexer(test[col])
            print('\n')
        ## imputing float means
        ## sklearn Imputer
        if((X_train[col].dtype=='float64')):
            print('Imputing the \'na\' values of the float variable \'%s\' with its mean: %.4f' %(col, X_train[col].mean()))
            X_train_col = np.array(X_train[[col]])
            X_train[col] = float_imputer.fit_transform(X_train_col)
            print('Imputing the \'na\' values of the float variable \'%s\' with its mean: %.4f' %(col, test[col].mean()))
            test_col = np.array(test[[col]])
            test[col] = float_imputer.transform(test_col)
            print('\n')

    print('The categorical variables of the data sets have been encoded successfully!\n')
    print('The \'na\' values of the float variables have been imputed (mean) succesfully!')
            
    return(X_train, test)


def RemoveCollinearPredictors(train, test, attribs_of_interest, threshold=0.9):
    
    ## REMOVE COLLINEAR PREDICTORS
    print('Determining the collinear float attributes...\n')
    print('Collinearity Threshold: %.2f\n' %threshold)
    
    ## select the train/test subset of dtype floats
    train_float = train[attribs_of_interest].copy()
    test_float = test[attribs_of_interest].copy()
    
    ## compute the pair-wise correlations for the attributes in data sets
    train_corrs = train_float.corr()
    test_corrs = test_float.corr()
    
    ## determine the high correlated for the given threshold
    train_HighCorrs = train_corrs[(abs(train_corrs) > threshold) & (train_corrs !=1)]
    test_HighCorrs = test_corrs[(abs(test_corrs) > threshold) & (test_corrs !=1)]
    
    ## verify that for the choosen collinearity threshold 
    ## the same subset of attributes will be removed from train and test data sets
    train_boolean_HighCorrs = train_HighCorrs.notnull()
    test_boolean_HighCorrs = test_HighCorrs.notnull()
    
    if train_boolean_HighCorrs.equals(test_boolean_HighCorrs):
        
        collinear_features = pd.Series()
        features = train_boolean_HighCorrs.columns
        
        for col in features:
            collinear_features_tmp = (train_boolean_HighCorrs.index[train_boolean_HighCorrs[col]]).to_series()
            collinear_features = collinear_features.append(collinear_features_tmp)
        
        collinear_features.drop_duplicates(inplace=True)
        
        print('Collinear Predictors:')
        print('------------------------')
        print(list(collinear_features))
        print('\n')
        print('Removing the collinear predictors from both train and test data set...\n')
        train_float.drop(collinear_features, axis=1, inplace=True)
        test_float.drop(collinear_features, axis=1, inplace=True)
        ## keep the non float variables of the original data sets 
        ## in a different pair of Pandas DataFrames
        features_non_float = (train.columns).difference(attribs_of_interest)
        train_non_float = train[features_non_float].copy()
        test_non_float = test[features_non_float].copy()
        ## returns the original data sets without the observed collinear predictors
        train = pd.concat([train_float, train_non_float], axis=1)
        test = pd.concat([test_float, test_non_float], axis=1)
        print('The collinear predictors have been removed successfully!\n')
                
    else:
        
        print('A different subset of attributes will be removed in train and test set.')
        print('Please choose a different collinearity threshold!')
        pass
    
    return(train, test)


def TrainPredictFunction(train, test, target):
    
    print('Creating an ExtraTreesClassifier object...\n')
    extrTrClassfr = ExtraTreesClassifier(n_estimators= 800,
                                         max_features= 40,
                                         criterion= 'entropy', 
                                         max_depth= None,
                                         min_samples_split= 2,
                                         min_samples_leaf= 1,
                                         class_weight='balanced_subsample',
                                         random_state=1,
                                         verbose=1)
    
    print('Training the ExtraTreesClassifier...\n')
    extrTrClassfr.fit(train, target)

    ## predictions on the test data set
    print('Providing the actual predictions...\n')
    y_pred = extrTrClassfr.predict_proba(test)
    
    return y_pred
    
    
if __name__ == '__main__':
    
    print('Start')
    print('Loading Data')
    ## training data
    train = pd.read_csv('../input/train.csv')
    target = train['target'].values
    train = train.drop(['ID','target'],axis=1)
    ## test data (awaiting predictions)
    test = pd.read_csv('../input/test.csv')
    test_ID = test['ID'].values
    test = test.drop(['ID'],axis=1)
    
    ## keep the the attributes names of the various dtypes in separate lists
    features_categorical = list(train.select_dtypes(include=['O', 'int64']).columns)
    features_nominal_categorical = list(train.select_dtypes(include=['O']).columns)
    features_ordinal_categorical = list(train.select_dtypes(include=['int64']).columns)
    features_float = list(train.select_dtypes(include=['float64']).columns)
    
    ## floats with no decisive infuence on 'target' variable:
    attribs_todrop_float = ['v1', 'v5', 'v9', 'v11', 'v13', 'v15',
    'v16', 'v23', 'v25', 'v28','v29', 'v32', 'v35', 'v42', 'v53',
    'v54', 'v57', 'v59', 'v60', 'v67', 'v70', 'v77', 'v78', 'v82',
    'v86', 'v89', 'v90', 'v92', 'v94', 'v95', 'v96', 'v103', 'v104',
    'v105', 'v111', 'v115', 'v117', 'v118', 'v120', 'v122', 'v124',
    'v126', 'v127']
    attribs_todrop_categorical = ['v22', 'v62', 'v72', 'v129', 'v3', 'v52',
    'v112', 'v125', 'v74']
    attribs_todrop = attribs_todrop_float+attribs_todrop_categorical
    
    print('Applying the CleanEncodeData() function...\n')
    train, test = CleanEncodeData(train, test, attribs_todrop=attribs_todrop)
    
    print('Applying the RemoveCollinearPredictors() function...\n')
    features_float_new = features_float
    for var in attribs_todrop_float:
          features_float_new.remove(var)
    train_copy = train.copy()
    test_copy = test.copy()
    train_copy, test_copy = RemoveCollinearPredictors(train_copy, test_copy, features_float_new, threshold = 0.97)
    
    
    print('Applying the TrainPredictFunction()...\n')
    y_pred = TrainPredictFunction(train_copy, test_copy, target)
    print('Start Output')
    tgrammat_submission = pd.DataFrame({"ID": test_ID, "PredictedProb": y_pred[:,1]})
    ## create my submission .csv file
    tgrammat_submission_filename = str('tgrammat_submission.csv')
    tgrammat_submission.to_csv(tgrammat_submission_filename, index=False)
    
    print('Finish')