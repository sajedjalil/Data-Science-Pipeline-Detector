import numpy as np, pandas as pd
import gc

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning,DataConversionWarning
warnings.filterwarnings("ignore",category=ConvergenceWarning)
warnings.filterwarnings("ignore",category=DataConversionWarning)
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import sklearn
print('SKLearn Version:',sklearn.__version__)
Local=False

if (Local):
    fDir=''
else:    
    fDir='../input/'

if (not Local):
    train = pd.read_csv(fDir+'train.csv')
    test = pd.read_csv(fDir+'test.csv')
else:
    train=joblib.load('train.pkl')    
    test=joblib.load('test.pkl')    
    
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')

gc.collect();
        
oof = pd.Series(np.zeros(len(train)))
preds = np.zeros(len(test))
    

IDX=[]
IDX=np.array(IDX)

for i in np.sort(np.unique(train['wheezy-copper-turtle-magic'])):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    IDX=np.append(IDX,idx1)
    IDX=IDX.ravel()
    train2.reset_index(drop=True,inplace=True)
    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    sel = VarianceThreshold(threshold=1.5).fit(data[cols])
    train3 = np.array(pd.DataFrame(sel.transform(train2[cols])).reset_index(drop=True))
    test3 = np.array(pd.DataFrame(sel.transform(test2[cols])).reset_index(drop=True))
    train33=train3.copy();test33=test3.copy();        
    nc=train33.shape[1]
    
    if (1):    
        tt=pd.concat([pd.DataFrame(train33),pd.DataFrame(test33)]).reset_index(drop=True)
        tt=MaxAbsScaler().fit_transform(tt)
        gm0=GaussianMixture(n_components=6, covariance_type='full',
                           tol=0.00001, reg_covar=0.01, max_iter=2000, 
                           n_init= 6, init_params='kmeans', 
                           weights_init=None, means_init=None, 
                           precisions_init=None , random_state=1, 
                           warm_start=False, verbose=0, verbose_interval=10)
        np.random.seed(1)
        gm0.fit(tt);ttU=gm0.predict_proba(tt);
        meansC=gm0.means_;precC=gm0.precisions_;
        #ttU=np.sign(ttU-0.5);
        ttU=pd.DataFrame(StandardScaler().fit_transform(ttU)).reset_index(drop=True);
        ttU=ttU.loc[:,ttU.apply(pd.Series.nunique) != 1]
        ttU=ttU.add_prefix('gm1_')
        trainU=(ttU[:train3.shape[0]]).reset_index(drop=True);
        testU=(ttU[train3.shape[0]:]).reset_index(drop=True);
        train3=np.array(pd.concat([pd.DataFrame(train3),trainU],axis=1).reset_index(drop=True))
        test3=np.array(pd.concat([pd.DataFrame(test3),testU],axis=1).reset_index(drop=True))        
    
    testX=pd.DataFrame(test3).reset_index(drop=True)
        
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    clf1 = QuadraticDiscriminantAnalysis(reg_param=0.5)    
    for train_index, test_index in skf.split(train3, train2['target']):        
        trainX=pd.DataFrame(train3[train_index,:]).reset_index(drop=True);trainY=train2.loc[train_index]['target'].reset_index(drop=True);
        validX=pd.DataFrame(train3[test_index,:]).reset_index(drop=True);validY=train2.loc[test_index]['target'];
        trainX.columns=trainX.columns.astype('str');validX.columns=validX.columns.astype('str');testX.columns=testX.columns.astype('str');
        testX.drop([x for x in testX.columns if 'ggmm' in x],axis=1,inplace=True)
        testX.drop([x for x in testX.columns if 'GM' in x],axis=1,inplace=True)
        useSL=True
        useFl=True
        useEr=True
        use2=True
             
        if (1):
            st1=MaxAbsScaler()            
            st2=StandardScaler()            
            st1.fit(pd.concat([trainX.iloc[:,:nc],validX.iloc[:,:nc],testX.iloc[:,:nc]]))
            gmP=GaussianMixture(n_components=3, covariance_type='full',
                                   tol=0.001, reg_covar=0.01, max_iter=1000, 
                                   n_init= 3, init_params='kmeans', 
                                   weights_init=None, means_init=None, 
                                   precisions_init=None , random_state=1, 
                                   warm_start=False, verbose=0, verbose_interval=10)  
            gmP.fit(st1.transform(trainX[trainY==1].iloc[:,:nc].reset_index(drop=True)))
            means1=gmP.means_;prec1=gmP.precisions_;wt1=gmP.weights_;
            trainP=pd.DataFrame(gmP.predict_proba(st1.transform(trainX.iloc[:,:nc])));validP=pd.DataFrame(gmP.predict_proba(st1.transform(validX.iloc[:,:nc])));testP=pd.DataFrame(gmP.predict_proba(st1.transform(testX.iloc[:,:nc])));
            st2.fit(pd.concat([trainP,validP,testP]));trainP=pd.DataFrame(st2.transform(trainP));validP=pd.DataFrame(st2.transform(validP));testP=pd.DataFrame(st2.transform(testP));
            trainP=trainP.add_prefix('ggmmpp_');validP=validP.add_prefix('ggmmpp_');testP=testP.add_prefix('ggmmpp_');
            gmN=GaussianMixture(n_components=3, covariance_type='full',
                                   tol=0.001, reg_covar=0.01, max_iter=1000, 
                                   n_init= 3, init_params='kmeans', 
                                   weights_init=None, means_init=None, 
                                   precisions_init=None , random_state=1, 
                                   warm_start=False, verbose=0, verbose_interval=10)  
            gmN.fit(st1.transform(trainX[trainY==0].iloc[:,:nc].reset_index(drop=True)))
            means0=gmN.means_;prec0=gmN.precisions_;wt0=gmN.weights_;
            trainN=pd.DataFrame(gmP.predict_proba(st1.transform(trainX.iloc[:,:nc])));validN=pd.DataFrame(gmP.predict_proba(st1.transform(validX.iloc[:,:nc])));testN=pd.DataFrame(gmP.predict_proba(st1.transform(testX.iloc[:,:nc])));
            st2.fit(pd.concat([trainN,validN,testN]));trainN=pd.DataFrame(st2.transform(trainN));validN=pd.DataFrame(st2.transform(validN));testN=pd.DataFrame(st2.transform(testN));
            trainN=trainN.add_prefix('ggmmnn_');validN=validN.add_prefix('ggmmnn_');testN=testN.add_prefix('ggmmnn_');
            trainX=pd.concat([trainX,trainP],axis=1).reset_index(drop=True);trainX=pd.concat([trainX,trainN],axis=1).reset_index(drop=True);
            validX=pd.concat([validX,validP],axis=1).reset_index(drop=True);validX=pd.concat([validX,validN],axis=1).reset_index(drop=True);
            testX=pd.concat([testX,testP],axis=1).reset_index(drop=True);testX=pd.concat([testX,testN],axis=1).reset_index(drop=True);
        clf1.fit(trainX,trainY)
        if (useFl):
            pT=clf1.predict_proba(trainX)[:,1]
            er=np.abs(pT-trainY)
            trainY[er>0.9]=1.0-trainY[er>0.9]        
        '''if (useSL):            
            clf1.fit(trainX,trainY)
            pTs=(clf1.predict_proba(testX)[:,1]) 
            sMax=0.999;sMin=0.001;
            testX2=testX[(pTs>=sMax) | (pTs<=sMin)].reset_index(drop=True)
            testX3=testX.copy()[:]+0.0#[(pTs<sMax) & (pTs>sMin)].reset_index(drop=True)
            testY2=np.zeros(testX.shape[0]);testY2[(pTs>=sMax) ]=1.0;
            testY2=testY2[(pTs>=sMax) | (pTs<=sMin) ]; 
            trainX=pd.concat([trainX,testX2]).reset_index(drop=True)
            trainY=pd.concat([trainY,pd.Series(testY2).reset_index(drop=True)]).reset_index(drop=True)'''
        clf1.fit(trainX,trainY)
        '''if (useFl):
            pT=clf1.predict_proba(trainX)[:,1]
            er=np.abs(pT-trainY)
            trainY[er>0.9]=1.0-trainY[er>0.9] '''       
        if (useEr):    
            #clf1.fit(trainX,trainY)
            pT=clf1.predict_proba(trainX)[:,1]
            er=np.abs(pT-trainY)
            clf1.fit(trainX[(er<0.75)],trainY[(er<0.75) ])
        if (use2):
            init_means=np.concatenate([means0,means1])
            init_precs=np.concatenate([prec0,prec1])            
            init_weights=np.concatenate([wt0,wt1])     
            init_weights=init_weights/(np.sum(init_weights))
            np.random.seed(1)
            gmC = GaussianMixture(n_components=6, covariance_type='full',
                                   tol=0.001, reg_covar=0.001, max_iter=500, 
                                   n_init= 1, init_params='kmeans', 
                                   weights_init=None, means_init=init_means, 
                                   precisions_init=init_precs , random_state=1, 
                                   warm_start=False, verbose=0, verbose_interval=10)
            gmC.fit(pd.concat([trainX.iloc[:,:nc],validX.iloc[:,:nc],testX.iloc[:,:nc]],axis = 0))
            ppTrain=gmC.predict_proba(trainX.iloc[:,:nc]);
            ppValid=gmC.predict_proba(validX.iloc[:,:nc]);
            ppTest=gmC.predict_proba(testX.iloc[:,:nc]);
            pValid0=(1.0*gmC.predict_proba(validX.iloc[:,:nc])[:,3:].max(axis=1)-1.0*gmC.predict_proba(validX.iloc[:,:nc])[:,:3].max(axis=1))
            pTest0=(1.0*gmC.predict_proba(testX.iloc[:,:nc])[:,3:].max(axis=1)-1.0*gmC.predict_proba(testX.iloc[:,:nc])[:,:3].max(axis=1))
            if (0):
                trainX['GM0']=ppTrain[:,3:].max(axis=1);
                validX['GM0']=ppValid[:,3:].max(axis=1);
                testX['GM0']=ppTest[:,3:].max(axis=1);
                trainX['GM1']=ppTrain.max(axis=1);
                validX['GM1']=ppValid[:,:3].max(axis=1);
                testX['GM1']=ppTest[:,:3].max(axis=1);
                if (1):
                    GMCols=[x for x in trainX.columns if 'GM' in x]
                    st=MaxAbsScaler();st.fit(pd.concat([trainX[GMCols],validX[GMCols],testX[GMCols]]));
                    trainX[GMCols]=st.transform(trainX[GMCols]);
                    validX[GMCols]=st.transform(validX[GMCols]);
                    testX[GMCols]=st.transform(testX[GMCols]);
                clf1.fit(trainX,trainY)        
                pT=clf1.predict_proba(trainX)[:,1]
                er=np.abs(pT-trainY)
                clf1.fit(trainX[(er<0.75)],trainY[(er<0.75) ])   
            pValid1= clf1.predict_proba(validX)[:,1]
            pTest1=clf1.predict_proba(testX)[:,1]
            pValid=0.1*(1.0*pValid1+0.5*pValid0)
            pTest=0.1*(1.0*pTest1*+0.5*pTest0)
            oof[idx1[test_index]] =pValid
            preds[idx2] +=  pTest/ skf.n_splits
        else:
            oof[idx1[test_index]] = clf1.predict_proba(validX)[:,1]
            preds[idx2] += clf1.predict_proba(testX)[:,1]/ skf.n_splits                   
    if i%10==0:
        print('Iter',i,' This:',np.round(roc_auc_score(train['target'][idx1],oof[idx1]),5),', Overall:',np.round(roc_auc_score(train['target'][IDX],oof[IDX]),5))
        
auc = roc_auc_score(train['target'],oof)
print('CV Score :',round(auc,5))

sub = pd.read_csv(fDir+'sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv', index=False)