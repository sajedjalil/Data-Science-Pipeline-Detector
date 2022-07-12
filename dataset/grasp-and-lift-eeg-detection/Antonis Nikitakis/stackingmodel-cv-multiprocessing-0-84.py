# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12, 2015

@author: Antonis Nikitakis

A unified framework utilizing the so-far published solutions of the Grasp-and-Lift EEG Detection challenge in the context of a stacking classifier and 
an ensemble classifier. An accurate CV scheme (modified leave-one-out) based on the public score evaluation is also presented.

Code is based on scripts  of Elena Cuoco and alexandrebarachant
and on some other forked scirpts such as  Adam Gągol's script which is also based on Elena Cuoco's script

I demonstrate two different classification schemes:
    1) an ensemble classifier with predefined weights
    2) a two-level stacking classifier

Both schemes performed about the same at this level with the ensemble classifier taking the lead. 
If more models come into play things may change. Public scores are a little-bit higher than the local CV,
they are strongly correlated though.

The code is further divided in the CV phase that helps designer to estimate its public score
and in the Submission phase which process the provided test data for the public leaderboard score evaluation.
 
The cross validation process is performed in the exact manner that the public leaderboard score is calculated.
That is a leave-one-out scheme based on subject series. The same CV scheme applied for both classifiers. 

Things are getting a little bit tricky in the stacking classifier due to the bookkeeping of the subject-series across folds.
Maybe simpler solutions are possible. The total CV score is the mean over the folds.
As 8 folds are performed per model the whole process is slow. Thus a multiprocessing scheme is also applied to improve performance.

Moving to the predictions of the provided test data, those are utilized as follows: 
I am using the the whole train series to refit the classifiers separately from the CV training. 
It could be another approach. That is to average all the "folded" predictions on the test set to make a single prediction 
implicitly using all the available training data. This also saves computation time but it didn't work in this case. 
If you have an idea why please inform me. 

Some extra features added:
    -user can define different subsampling between models
    -user can define different cutoff frequencies between models
    -user can duplicate the predictions of a model to increase its weight in the stacking classifier.
    This approach is also applied in the ensemble classifier (instead of soft-weighting models) to have comparable results

The code is probably a mess and needs a cleanup; use it at your own risk :). I had great difficulty in expressing the algorithm in a simpler way.
If you find any bugs, have any suggestions or if you want to team up, please email me: a.s.nikitakis@gmail.com

Thnx
Antonis.


"""
from sklearn.lda import LDA
import numpy as np
import pandas as pd
from sklearn.linear_model import (LogisticRegression, LinearRegression,SGDClassifier)
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import ( preprocessing, cross_validation   )
from scipy.fftpack import fft, ifft        
from sklearn import metrics
from scipy.optimize import minimize
from glob import glob
import os

from sklearn.preprocessing import StandardScaler
 
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

      
# training subsample.if you want to downsample the training data
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

idx_tot = []
stacking_submission_preds_tot = []
ensemble_submission_preds_tot = []


#######number of subjects###############


###loop on subjects and 8 series for train data + 2 series for test data
def subject_processing(subject,MAX_SERIES=9): # change to save time when debugging
    FNAME = "../input/{0}/subj{1}_series{2}_{3}.csv"
    def load_data(subj, series=range(1,MAX_SERIES), prefix = 'train'):
        data = [pd.read_csv(FNAME.format(prefix,subject,s,'data'), index_col=0) for s in series]
        idx = [d.index for d in data]
        data = [d.values.astype(float) for d in data]
        if prefix == 'train':
            events = [pd.read_csv(FNAME.format(prefix,subject,s,'events'), index_col=0).values for s in series]
            events = np.concatenate(events,axis=0)        
            return data, events
        else:
            return data, idx 
            
            
    def compute_features(X, scale=None,cutoff=250.0):
        from scipy.signal import butter, lfilter
        
        X0 = [x[:,0] for x in X]
        X = np.concatenate(X, axis=0)
        F = [];
        for fc in np.linspace(0,1,11)[1:]:
            b,a = butter(3,fc/cutoff,btype='lowpass')
            F.append(np.concatenate([lfilter(b,a,x0) for x0 in X0], axis=0)[:,np.newaxis])
        F = np.concatenate(F, axis=1)
        F = np.concatenate((X,F,F**2), axis=1)
            
        if scale is None:    
            scale = StandardScaler()
            F = scale.fit_transform(F)
            return F, scale
        else:
            F = scale.transform(F)
            return F
        
    X_train_, y = load_data(subject)     
    X_test_, idx = load_data(subject,[9,10],'test')     
   
    #%%########### some helper functions ######################################    
    print ('CV Train subject', subject)
    def train_predict(lr,X_train,X_test,y,subsample):

        pred = np.empty((X_test.shape[0],6))
        for i in range(6):
            y_train= y[:,i]
            lr.fit(X_train[::subsample,:],y_train[::subsample])
            pred[:,i] = lr.predict_proba(X_test)[:,1]
        return pred
    

    def predict(lr,X_test):
#        print 'Making predictions on test set for current fold'
        pred = np.empty((X_test.shape[0],6))
        for i in range(6):
            pred[:,i]=lr.predict_proba(X_test)[:,1]
        return pred


    
    def ensemble_preds(preds):#,weights=ensemble_weights):
#        weighted_preds=[]
        
#        for i, pred in enumerate(preds):
#            w_pred=pred+weights[i]
#            weighted_preds.append(w_pred)
#        final_preds=np.sum(weighted_preds,axis=0)
#        
#        weighting is implied by adding a model twice
        final_preds=np.mean(preds,axis=0)
        return final_preds
    
    cv_model_preds=[]
    t_preds_refit_model=[]

#%%########### 1st stage classifiers (generalizer) ############################
 
    for wi, (subsample_, m_name, model) in enumerate(zip(m_subsample,models_name,models)):
        cv_preds_series=[]     
        subject_scores=[]
        series_index=[] #holds the indexes for the "unrolled" series predictions for the stacking classifier 
        index=0
           
        X_train, scaler = compute_features(X_train_,None,c_freq[wi])
        X_test = compute_features(X_test_, scaler,c_freq[wi])  
     
        for i in range(1,MAX_SERIES):
                      
            series_train=[j for j  in range(1,MAX_SERIES) if j!=i ]
            series_test= [j for j  in range(1,MAX_SERIES) if j==i ]
            
            X_trCV,y_trCV  = load_data(subject,series_train)
            X_trCV = compute_features(X_trCV,scaler,c_freq[wi])
            X_valCV,y_valCV = load_data(subject,series_test)
            X_valCV = compute_features(X_valCV,scaler,c_freq[wi])            
            
            series_index.append( (index,index+y_valCV.shape[0]) ) # we do this to track the series to make the series-based CV scheme
            index=index+y_valCV.shape[0]

            preds=train_predict(model,X_trCV,X_valCV,y_trCV,subsample_)
            cv_preds_series.extend(preds) #extend series by series, the same way y is built, series_index=[] helps to rebuild series in the 2nd stage
            #I extend the series to a single array instead of appending to multiple list
            #this way is easiear to stack multiple model predictions to form the feature vectors for the 2nd stage classifier
                       
            scores=np.zeros(6)
            for k in range(6):
                scores[k]=metrics.roc_auc_score(y_valCV[:,k], preds[:,k])       
            
#            print("Fold %d ROC score= %3.4f"% (i,np.mean(scores)) )
            subject_scores.append(np.mean(scores))
            pass
        
        cv_preds_series=np.asarray(cv_preds_series)
        cv_model_preds.append(cv_preds_series)
        if (stacking_weights[wi]==2): cv_model_preds.append(cv_preds_series) #add for a second time to increase model weight
        
        print("-->Subject: %d, Model: %s, MEAN CV ROC score= %3.4f"% (subject,m_name,np.mean(subject_scores)) )
        
        print ("Refitting the whole series for the 1st stage classifiers")
        t_preds_refit=train_predict(model,X_train,X_test,y,subsample_)
        t_preds_refit_model.append(t_preds_refit)
        if (stacking_weights[wi]==2): t_preds_refit_model.append(t_preds_refit) #add for a second time to increase weight
           
    pass
    
    #------- For the Cross Validation ---------
    stacking_cv_features=np.concatenate(cv_model_preds ,axis=1) #stacking predictios to form feature vectors
    ensemble_cv_preds = ensemble_preds(cv_model_preds)  # weighting models for the Cross Validation of the ensemble model                                         
    
    #calculate and print CV score
    scores_ensemble=[]        
    for i in range(6):
        scores_ensemble.append(metrics.roc_auc_score(y, ensemble_cv_preds) )
    ensemble_score=  np.mean(scores_ensemble)  
    print("-->Subject: %d ....Ensemble Classifier, Mean CV score %0.4f" % (subject,ensemble_score))  

    #------- For the Submission ------------------
    #extracting features from re-fitted classifier and forwarded to the second stage classifier
    stacking_submission_features=np.concatenate(t_preds_refit_model ,axis=1)
    
    #final predictions for the ensemble model (there is no second stage)
    ensemble_submission_preds=ensemble_preds(t_preds_refit_model)
    
#%%########### 2nd stage classifier (model stacker) ########################### 
   
    subsample=100 #stacker classifier subsumpling (larger values seem to work better)
    for k, clf in enumerate(clfs):
        fold_scores=[]
        
        for idx1, dump in enumerate( series_index): #fold counting
            X_trCV=[]
            y_trCV=[]
            
            for idx2, current in enumerate(series_index): #rebuilding leave-one-out CV series                
                if (idx1!=idx2):
                    X_trCV.extend( stacking_cv_features[current[0]:current[1],:])
                    y_trCV.extend( y[current[0]:current[1],:] )                    
                else:    
                    X_valCV = stacking_cv_features[current[0]:current[1],:]
                    y_valCV = y[current[0]:current[1],:]
                           
            #current fold is rebuilt, use this to train 2nd stage classifier
            X_trCV=np.asarray(X_trCV)
            y_trCV=np.asarray(y_trCV)
   
            scores=[]
            #per class evaluation
            for i in range(6):
               y_trCVi= y_trCV[:,i]
               y_valCVi= y_valCV[:,i]
                
               clf.fit( X_trCV[::subsample,:],y_trCVi[::subsample])
               preds=clf.predict_proba(X_valCV)[:,1]
               
               scores.append(metrics.roc_auc_score(y_valCVi, preds) )
              
            #  print("....Stacker Model=%d, Fold:%d, Stacking CV score %0.4f" % (k,idx1,np.mean(scores)) )
            fold_scores.append(np.mean(scores) )

        #--- Stacker model CV predictions
        stacking_score=np.mean(fold_scores)        
        print("-->Subject: %d ....Stacking Classifier:%d, Mean CV score %0.4f" % (subject,k,stacking_score) )
   
    #-----For the Submission ------
    # Remember: submission is based on the last examined classifier
 
    # refit stacking classifier in the whole series, and get 
    # final submission preds from the 1st stage features  
    stacking_submission_preds=train_predict(clf,stacking_cv_features,stacking_submission_features,y,subsample)

    # prepare data to pickle   
    data_pack=(stacking_submission_preds, ensemble_submission_preds, idx,stacking_score, ensemble_score)
    import cPickle
    import gzip
    with gzip.GzipFile('./output/subject%d.dump.gz'%subject, 'w') as f:
        cPickle.dump(data_pack,f)
    print ('Finished storing subject%d dump data.... '%subject)
    

#%%########### set parameters module ##############################################

do_process=True #False to load from files
 
##--------Define your models and parameters -------------
subjects = range(1,13)
 
#------1st stage models
models=[LogisticRegression(),
        LDA(),
        RandomForestClassifier(n_estimators=200, criterion="entropy",random_state=1),
        LDA() ]
models_name=['LRegression(sub40)', 'LDA(sub40)','RF(sub40)', 'LDA_fq_500(sub40)']

#different subsampling across models
m_subsample=[40,40,40,40]
# we stack twice the predictions of a model to increase its weight in the stacking classifier
stacking_weights=[1, 2, 1, 1]
# differente filter responses across models 
c_freq=[250.0,250.0,250.0,500.0]
   
#------2nd stage models

clfs = [  LogisticRegression()  ] # LogisticRegression  worked best for me       

#%%########### multiprocessing module #########################################

#subject_processing(9)

import multiprocessing
import time

if do_process==True:
    def start_process():
        pass
  
    start = time.time()

    # do not add more parallelism than the pysical cores of your machine (i.e hyperthreading won't improve things)     
    pool_size =4 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size,initializer=start_process, maxtasksperchild=2)
    pool_outputs = pool.map(subject_processing, subjects)
    pool.close()
    pool.join()    
    
    end = time.time()
    print ('total processing time in minutes=%f'%(end - start)/60 )
          
    print("All threads finished ")
else:
    print("Skipping processing, loading from files")
    
#%%########### results aggregation module ##################################### 
#loads the individual subject files to form a single submission file 
#implicitly linearizes the asynchronously produced results
    
print("Aggregating results ")
import cPickle
import gzip
stacking_scores=[]
ensemble_scores=[]

subjects = range(1,13)

for subject in subjects:
    print ('Loading dump data for subject:%d.... '%subject) 
    with  gzip.GzipFile('./output/subject%d.dump.gz'%subject, 'r') as f:
        data_pack = cPickle.load(f)
    stacking_submission_preds, ensemble_submission_preds, idx,stacking_score, ensemble_score= data_pack
    
    stacking_submission_preds_tot.append(stacking_submission_preds)
    ensemble_submission_preds_tot.append(ensemble_submission_preds) 
    idx_tot.append(np.concatenate(idx))
    
    stacking_scores.append(stacking_score)
    ensemble_scores.append(ensemble_score)

print("TOTAL....Stacking Classifier CV score %0.4f" % (np.mean(stacking_scores)) )
print("TOTAL....Ensemble Classifier CV score %0.4f" % (np.mean(ensemble_scores)) )

#%%########### submission file module #########################################
print("Making submission ") 
# create pandas object for submission
submission_ensemble = pd.DataFrame(index=np.concatenate(idx_tot),
                          columns=cols,
                          data=np.concatenate(ensemble_submission_preds_tot))

submission_stacking = pd.DataFrame(index=np.concatenate(idx_tot),
                          columns=cols,
                          data=np.concatenate(stacking_submission_preds_tot))

# write files

import gzip
 
submission_ensemble.to_csv(gzip.open('submission_ensemble.csv.gz',"w"),index_label='id',float_format='%.3f')
submission_stacking.to_csv(gzip.open('submission_stacking.csv.gz',"w"),index_label='id',float_format='%.3f')
 