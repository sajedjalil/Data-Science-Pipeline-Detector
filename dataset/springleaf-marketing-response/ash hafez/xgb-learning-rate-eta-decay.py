import numpy as np
import pandas as pd
import xgboost as xgb
import random

from sklearn.preprocessing      import LabelEncoder
from sklearn.preprocessing      import OneHotEncoder
import xgboost as xgb
from sklearn.cross_validation   import StratifiedKFold

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------------
#
def evalauc(preds, dtrain):
    
    labels = dtrain.get_label()
    
    return 'roc', - roc_auc_score(labels, preds)

# --------------------------------------------------------------------------------
#
def do_train(X, Y, initial_eta, min_eta, verbose=False):

    np.random.seed( 1 )
    random.seed(    1 )

    cv_scores    = []
    train_scores = []
    
    split = StratifiedKFold(Y, 5, shuffle=True )
    
    fold = 0
    
    for train_index, cv_index in split:
    
        fold = fold + 1
                    
        y_pred              = []
    
        X_train, X_valid    = X[train_index,:], X[cv_index,:]
        y_train, y_valid    = Y[train_index],   Y[cv_index]
    
        params = {
                "max_depth"             : 5, 
                "eta"                   : initial_eta,
                "min_eta"               : min_eta,
                "eta_decay"             : 0.5,
                "max_fails"             : 3,
                "early_stopping_rounds" : 20,
                "objective"             : 'binary:logistic',
                "subsample"             : 0.8, 
                "colsample_bytree"      : 1.0,
                "n_jobs"                : -1,
                "n_estimators"          : 5000, 
                "silent"                : 1
        }
    
        num_round       = params["n_estimators"]
        eta             = params["eta"]
        min_eta         = params["min_eta"]
        eta_decay       = params["eta_decay"]
        early_stop      = params["early_stopping_rounds"]
        max_fails       = params["max_fails"]
        
        params_copy     = dict(params)
        
        dtrain          = xgb.DMatrix( X_train, label=y_train ) 
        dvalid          = xgb.DMatrix( X_valid, label=y_valid )  
    
        total_rounds        = 0
        best_rounds         = 0
        pvalid              = None
        model               = None
        best_train_score    = None
        best_cv_score       = None
        fail_count          = 0
        best_rounds         = 0
        best_model          = None
        
        while eta >= min_eta:           
            
            model        = xgb.train( params_copy.items(), 
                                      dtrain, 
                                      num_round, 
                                      [(dtrain, 'train'), (dvalid,'valid')], 
                                      early_stopping_rounds=early_stop,
                                      feval=evalauc )
    
            rounds          = model.best_iteration + 1
            total_rounds   += rounds
            
            train_score = roc_auc_score( y_train, model.predict(dtrain, ntree_limit=rounds) )
            cv_score    = roc_auc_score( y_valid, model.predict(dvalid, ntree_limit=rounds) )
    
            if best_cv_score is None or cv_score > best_cv_score:
                fail_count = 0
                best_train_score = train_score
                best_cv_score    = cv_score
                best_rounds      = rounds
                best_model       = model

                ptrain           = best_model.predict(dtrain, ntree_limit=rounds, output_margin=True)
                pvalid           = best_model.predict(dvalid, ntree_limit=rounds, output_margin=True)
                
                dtrain.set_base_margin(ptrain)
                dvalid.set_base_margin(pvalid)
            else:
                fail_count += 1

                if fail_count >= max_fails:
                    break
    
            eta                 = eta_decay * eta
            params_copy["eta"]  = eta
    
        train_scores.append(best_train_score)
        cv_scores.append(best_cv_score)

        print("Fold [%2d] %9.6f : %9.6f" % ( fold, best_train_score, best_cv_score ))
        
    print("-------------------------------")
    print("Mean      %9.6f : %9.6f" % ( np.mean(train_scores), np.mean(cv_scores) ) )
    print("Stds      %9.6f : %9.6f" % ( np.std(train_scores),  np.std(cv_scores) ) )
    print("-------------------------------")
        
# ----------------------------f----------------------------------------------------
#
def main():
    
    data_path = "../input/"
    
    train = pd.read_csv( data_path + "train.csv", dtype=str, nrows=20000)
    
    Y = train.target.values.astype(np.int32)
    X = train[ [ "VAR_0001", "VAR_0005", "VAR_0006", "VAR_0226"] ].values
    
    for c in range(X.shape[1]):
        encoder = { }
        for r in range(X.shape[0]):
            x = X[r,c]
            if x in encoder:
                mapping = encoder[ X[r,c] ]
            else:
                mapping = 1 + len(encoder)
                encoder[ X[r,c] ] = mapping
            X[r,c] = mapping

#        enc = LabelEncoder()
#        enc.fit(X[:,c])
#        X[:,c] = enc.transform(X[:,c])
    
    # enc = OneHotEncoder()
    # enc.fit(X)
    # X  = enc.transform(X)
    
    print(X.shape)

    print("\nWithout decay ...\n")
    do_train(X, Y, 0.1, 0.1)

    print("\nWith decay ...\n")   
    do_train(X, Y, 0.1, 0.00001)

# --------------------------------------------------------------------------------
#
if __name__ == '__main__':

    main()
                