""" Benchmark for Santander competition

Author: Andrew Stier based heavily off of code by Marios Michaildis

script for Santander Customer Satisfaction competition 2016

"""
import numpy as np # numpy has various stat's helpers. Also contains functions to load files.
import pandas as pd
from sklearn.preprocessing import StandardScaler # we will use this to standardize the data
from sklearn.metrics import roc_auc_score # the metric we will be tested on . You can find more here :  https://www.kaggle.com/wiki/AreaUnderCurve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold,cross_val_score # the cross validation method we are going to use
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import rankdata
from sklearn.cross_validation import train_test_split

import xgboost as xgb

# from sklearn.grid_search import GridSearchCV
SEED = 12  # seed to replicate results

#If you want to use just a part of the data to make sure your code works before you run it on your entire dataset, you can set the percentage here and set USE_PARTIAL_DATA to 1.
PERCENT_DATA = 0.05
USE_PARTIAL_DATA = 0

def save_results(predictions,IDs,  filename):
    with open(filename, 'w') as f:
        f.write("id,TARGET\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (IDs[i], pred))
            
def convert_from_prob_to_order(preds):
    preds_df=pd.DataFrame(preds,index=range(len(preds)),columns=['preds'])
    preds_df.sort_values(by='preds',inplace=True)
    preds_df['new_order']=np.array(range(len(preds)))
    preds_df.sort_index(inplace=True)
    tmp=np.array(preds_df.new_order)
    tmp=tmp*1.0/len(tmp)
    return tmp

# using numpy to print results
def printfilcsve(X, filename):

    np.savetxt(filename,X) 
                

def main():

    #Remember to change the filename for each of your different models
    filename="tuned_xgb_seed_1"
    

    model=xgb.XGBClassifier(max_depth=5,colsample_bytree=0.7,n_estimators=500, seed=SEED,learning_rate=0.02,subsample=0.7)
    
    # === load data into numpy arrays === #
    print ("loading training data")
    X=np.loadtxt("train.csv",skiprows=1, delimiter=",", usecols=range (1, 370)) # 1 is inclusive and 22 exclusive. so it is [1,21] basically. We skip 1 row since the first one is headers
    print ("loading labels")    
    y=np.loadtxt("train.csv",skiprows=1, delimiter=",", usecols=[370]) # The last one - the respnse variable   
    print ("loading test data")
    X_test=np.loadtxt("test.csv",skiprows=1, delimiter=",", usecols=range (1, 370)) # 1 is inclusive and 22 exclusive. so it is [1,21] basically
    print ("loading ids")    
    ids=np.loadtxt("test.csv",skiprows=1, delimiter=",", usecols=[0]) # The first column is the id
    


    if USE_PARTIAL_DATA == 1:
        X, X_discard, y, y_discard = train_test_split(X, y, train_size=PERCENT_DATA, random_state=42)
    
    model.fit(X,y)
    
    #remove "unimportant" features
    feat_imps=np.array(model.feature_importances_)
    
    inds_col=feat_imps>0.001
    
    X_all=X.copy()
    X_test_all=X_test.copy()
    X=X_all[:,inds_col]
    X_test=X_test_all[:,inds_col]

    print("running cross validation and creating train predictions for stacking")
    #allocate space to create an array of training predictions to be used for stacking later
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] 
    # === training & metrics === #
    mean_auc = 0.0
    n = 5  # number of folds in strattified cv
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=123)     
    i=0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # cleaning and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))
#        
#        # do scalling
##        scaler=StandardScaler()
##        X_train = scaler.fit_transform(X_train)
##        X_cv = scaler.transform(X_cv)
#
#       
#        
        model.fit(X_train,y_train)
        #  make predictions in probabilities
        preds=model.predict_proba(X_cv)[:,1]
        
        
        roc_auc = roc_auc_score(y_cv, preds)
        # print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

        #Store your train predictions in the train stacker.
        no=0
        for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1

        i+=1
      
    
    mean_auc/=n
    print (" Average AUC: %f" % (mean_auc))

#    make final model
    
    preds_final=model.predict_proba(X_test)[:,1] 
    
    
    #create training predictions file to be used in stacking 
    printfilcsve(np.array(train_stacker), filename + ".train.csv")

    #create test predictions file to be used in stacking
    printfilcsve(np.array(preds_final), filename+ ".test.csv")

    #create submission file
    save_results(preds_final, ids,  filename+"_submission_" +str(mean_auc).replace(".","_") + ".csv") # putting the actuall AUC (of cv) in the file's name for reference
    
if __name__ == '__main__':
    main()
