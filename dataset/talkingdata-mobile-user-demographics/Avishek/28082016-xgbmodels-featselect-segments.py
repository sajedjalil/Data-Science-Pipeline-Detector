print('')
#
## coding: utf-8
#
## In[1]:
#
#from __future__ import print_function
#
#"""
#Created on Tue Oct 13 00:36:13 2015
#
#@author: Administartor
#"""
#
#import os
#import numpy as np
#import sys
#import pickle as pkl
#from multiprocessing import cpu_count
#from sklearn.metrics import log_loss as accuracy_score
##import xgboost as xgb # import XGBClassifier
#from xgblinear import XGBLinearClassif
#from segment_models import SegmentModels
#import gc
#import itertools
#import random
##from sklearn.externals import joblib
#from sklearn.feature_selection import chi2
#from sklearn.feature_selection import SelectPercentile
#import pandas as pd
#
#
## In[2]:
#
##from scipy.stats import gmean #, hmean
#
#random.seed(1005)
#
#if 'nt' in os.name:
#    print('Ruinning on a Windows machine')
#    os.chdir('E:/tdata')
#    nCores = 1
#else:
#    print('Running on a Linux machine')
#    os.chdir('/data/tdata')
#    nCores = int(0.75*cpu_count())
#
#
## In[3]:
#
#print('Running on: ' + str(nCores))
#
#sys.path.append('code')
#import mrd
#import utils as ut
#from generic_model import GenericModel
#
#
## In[4]:
#
#"""
#    set params to run this is script starts here
#"""
#desc = 'xgbboost-chi2-feat-selection'
#k = 5
#kFoldsFile = '5fold.pkl'
#params = ['percentile']
#measureType = 'mlogloss'
#grid = [i for i in itertools.product(range(5,105,5))
#       ]
#"""
#    set params to run this is script ends here
#"""
#
#
## In[5]:
#
#print('Desc:', desc)
#
#print('Preparing mrd...')
#
#trainX, trainY, testX, testIds, le_grp, train_segment_ids, test_segment_ids, nb_segments = mrd.get_mrd('data/D_all.csv.gzip', sids = True)
#
#print('Shapes:', trainX.shape, testX.shape)
#
#nTrain = trainX.shape[0]
#
#nTest = testX.shape[0]
#
#nClasses = len(np.unique(trainY))
#
#cvTrainIdx, cvTestIdx = pkl.load(open(kFoldsFile,'rb'))
#
#
## In[6]:
#
##trainY_cv_pred = None
#train_cv_pred_prob = None
##testY_cv_pred = None
#testY_cv_pred_prob = None
#cv_measure = []
#best_cv_measure = None
#best_params = None
#best_foldwise_measure = None
#best_iteration = []
#
#
## In[7]:
#
##-----declare feat selection and learning params
#feat_selector_params = {}
#feat_selector_params[0] = {'score_func':chi2, 'percentile':50}                     
#feat_selector_params[1] = {'score_func':chi2, 'percentile':50}
#learner_params = {}
#learner_params[0] = {'nb_classes':nClasses, 'alpha':0, 'lambda_': 0,
#                     'lambda_bias': 0, 'eta':0.01, 'nb_rounds':75}
#learner_params[1] = {'nb_classes':nClasses, 'alpha':0, 'lambda_': 0,
#                     'lambda_bias': 0, 'eta':0.01, 'nb_rounds':75}
##------
#
#
## In[8]:
#
#print('Running cv...')
#
#for pv in grid:
#    
#    tmp_measure = []
#    
##    tmp_trainY_cv_pred = np.zeros((nTrain,))
#    tmp_trainY_cv_pred_prob = np.zeros((nTrain,nClasses)) if nClasses > 2 else np.zeros((nTrain,))
##    tmp_testY_cv_pred = np.zeros((nTest,k)) 
#    tmp_testY_cv_pred_prob = []
#    
#    for fold in range(0,k):
#        print(str(params) + ':' + str(pv) + ' fold:' + str(fold))
#        
#        trn = cvTrainIdx[fold]
#        tst = cvTestIdx[fold]
#        
#        feat_selector_params[0]['percentile'] = pv[0]
#        feat_selector_params[1]['percentile'] = pv[0]
#        
#        model = SegmentModels(nb_segments=nb_segments,learner=XGBLinearClassif,learner_params=learner_params,
#                              feat_selector=SelectPercentile,feat_selector_params=feat_selector_params)
#        
#        model.feature_selection(trainX, trainY, train_segment_ids)
#        
#        model.fit(trainX[trn,:], trainY[trn], train_segment_ids[trn])
#        
#        pred_prob = model.predict_proba(trainX[tst,:], train_segment_ids[tst])
#        
##        tmp_trainY_cv_pred[tst] = pred
#        
#        tmp_trainY_cv_pred_prob[tst] = pred_prob
#        
#        tmp_measure.append(model.score(trainY[tst], pred_prob, accuracy_score, train_segment_ids[tst])[0])
#        
##        tmp_testY_cv_pred[:,fold] = model.predict(testX, ntree_limit = model.best_iteration)
#        
#        tmp_testY_cv_pred_prob.append(model.predict_proba(testX, test_segment_ids))
#    
#    tmp_measure = [round(np.mean(list(s)),5) for s in zip(*tmp_measure)]
#    cv_measure.append(tmp_measure)
#    
#    print('cv measure:', cv_measure)
#    
#    if best_cv_measure is None:
#        best_cv_measure = tmp_measure
#        best_feat_selector_params = feat_selector_params
#        best_learner_params = learner_params
##        trainY_cv_pred = tmp_trainY_cv_pred
#        train_cv_pred_prob = tmp_trainY_cv_pred_prob
##        testY_cv_pred = tmp_testY_cv_pred
#        testY_cv_pred_prob = tmp_testY_cv_pred_prob
#        best_foldwise_measure = tmp_measure
#    for s in range(nb_segments):
#        if best_cv_measure[s] >  tmp_measure[s]:
#            best_cv_measure[s] = tmp_measure[s]
#            best_feat_selector_params[s] = feat_selector_params[s]
#            best_learner_params[s] = learner_params[s]
#            #        trainY_cv_pred = tmp_trainY_cv_pred
#            train_cv_pred_prob = tmp_trainY_cv_pred_prob
#            #        testY_cv_pred = tmp_testY_cv_pred
#            testY_cv_pred_prob = tmp_testY_cv_pred_prob
#            best_foldwise_measure = tmp_measure
#    
#    print('Best cv measure:', best_cv_measure, 'Avg. best cv measure:', np.mean(best_cv_measure))
#    print('Best param--', best_feat_selector_params, best_learner_params)
#    
#    gc.collect()
#
#
## In[9]:
#
#gm = GenericModel(desc = desc, params = params, grid = grid)
#gm.cv_measure = cv_measure
#gm.measureType = measureType
##gm.testY_cv_pred = testY_cv_pred
#gm.testY_cv_pred_prob = testY_cv_pred_prob
#gm.train_cv_pred_prob = train_cv_pred_prob
##gm.trainY_cv_pred = trainY_cv_pred
#gm.best_cv_measure = best_cv_measure
#gm.best_params = best_params
#gm.best_foldwise_measure = best_foldwise_measure
#
#
## In[10]:
#
#print('Building full models')
#model = SegmentModels(nb_segments=nb_segments,learner=XGBLinearClassif,learner_params=best_learner_params,
#                      feat_selector=SelectPercentile,
#                      feat_selector_params=best_feat_selector_params)
#model.feature_selection(trainX, trainY, train_segment_ids)
#model.fit(trainX, trainY, train_segment_ids)
#gc.collect()
#
#
## In[11]:
#
##trainY_all_pred = model.predict(trainX, ntree_limit = 1500)
#trainY_all_pred_prob = model.predict_proba(trainX,train_segment_ids)
#
#gc.collect()
#
##testY_all_pred = model.predict(testX, ntree_limit = model.best_iteration)
#testY_all_pred_prob = model.predict_proba(testX,test_segment_ids)
#
##gm.model = model
##gm.trainY_all_pred = trainY_all_pred
#gm.trainY_all_pred_prob = trainY_all_pred_prob
##gm.testY_all_pred = testY_all_pred
#gm.testY_all_pred_prob = testY_all_pred_prob
#
#gm.save() 
#
#
## In[13]:
#
##-----the following section should remain same for all models/methods----
#
##calib probas if improves
#gm.trainY_all_pred_prob, gm.testY_all_pred_prob,gm.train_cv_pred_prob, gm.testY_cv_pred_prob, gm.best_cv_measure, gm.calibrated = ut.calibrateProbasIfImproves(trainY,
#                             gm.trainY_all_pred_prob, gm.testY_all_pred_prob,
#                             gm.train_cv_pred_prob, gm.testY_cv_pred_prob,
#                             accuracy_score, np.mean(gm.best_cv_measure))
##print 'Saving GenericModel after prob calibration'
##gm.save()
##-------
#
##---rescaling probabilites so that that each row sum upto 1
##gm = ut.rescale_probas(gm)
##gm.save()
#
#
## In[14]:
#
##create submission file from all pred
##testIds = pkl.load(open('testIds.pkl','rb'))
#import pandas as pd
#sub = pd.DataFrame(np.column_stack((testIds,gm.testY_all_pred_prob)))
#sub.columns = ['device_id'] + list(le_grp.classes_)
##sub.id = sub.id.astype(int)
#sub = ut.clip_values(sub)
#sub.to_csv('sub_all_' + desc + '_' + str(round(gm.best_cv_measure,4)) +'_.csv', index =  False,
#           columns = ['device_id'] + list(le_grp.classes_))
#print('Submission full data based ready')
#
##create submission file from cv pred
#weights = np.array(gm.best_foldwise_measure)/float(np.sum(gm.best_foldwise_measure))
#sub = pd.DataFrame(np.column_stack((testIds,ut.weighted_mean(gm.testY_cv_pred_prob, weights))))
#sub.columns = ['device_id'] + list(le_grp.classes_)
##sub.id = sub.id.astype(int)
#sub = ut.clip_values(sub)
#sub.to_csv('sub_wa_' + desc + '_' + str(round(gm.best_cv_measure,4)) +'_.csv', index =  False,
#           columns = ['device_id'] + list(le_grp.classes_))
#
#
#sub = pd.DataFrame(np.column_stack((testIds,ut.arithmetic_mean(gm.testY_cv_pred_prob))))
#sub.columns = ['device_id'] + list(le_grp.classes_)
##sub.id = sub.id.astype(int)
#sub = ut.clip_values(sub)
#sub.to_csv('sub_am_' + desc + '_' + str(round(gm.best_cv_measure,4)) +'_.csv', index =  False,
#           columns = ['device_id'] + list(le_grp.classes_))
#
#sub = pd.DataFrame(np.column_stack((testIds,ut.geometric_mean(gm.testY_cv_pred_prob))))
#sub.columns = ['device_id'] + list(le_grp.classes_)
##sub.id = sub.id.astype(int)
#sub = ut.clip_values(sub)
#sub.to_csv('sub_gm_' + desc + '_' + str(round(gm.best_cv_measure,4)) +'_.csv', index =  False,
#           columns = ['device_id'] + list(le_grp.classes_))
#
#print('Submission cv data based ready')
#
#
## In[16]:
#
##--save feat imp
#feat_imp = model.segments[0]['model'].feature_importances_
#pkl.dump([feat_imp,trainX,testX], open('feat_select/seg0_xgb_feat_imp_'+desc, 'wb'))
#
#feat_imp = model.segments[0]['model'].feature_importances_
#pkl.dump([feat_imp,trainX,testX], open('feat_select/seg1_xgb_feat_imp_'+desc, 'wb'))
#print('Feat imp written to disk' )
#
#
## In[ ]:
#
#
#
