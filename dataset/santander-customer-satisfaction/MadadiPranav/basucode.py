
import pandas as pd
import sys
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
import numpy as np
import heapq, random 
from collections import Counter
from math import ceil
from sklearn.ensemble import  RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import  LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import  VarianceThreshold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

#%%
def preData(data, M=1000):
    """
    Make data categorical by removing noise using rounding
    
    """
    new_data=[]
    m,n =data.shape
    for i in xrange(n):
        C=data[:,i]
        m=min([x for x in C if x!=0])
        new_data.append([ int(round(M*v)) for v in C])
        new_data.append([ int(round(v/m)) for v in C])
    return np.array(new_data).T
#%%
#removing constant features

def removeConst(A):
    selector = VarianceThreshold(0)
    selector.fit(A)
    return selector, selector.transform(A)
#%%
#removing identical features
def removeIden(A):
    r, c = A.shape
    import itertools
    col_identical = set()
    for pair in itertools.combinations(range(np.shape(A)[1]),2):
        if np.array_equal(A[:,pair[0]],A[:,pair[1]]): #compare columns
            col_identical.add(pair[0]) 
    I=list(set(range(c))-set(col_identical))
    return I, A[:,I]  
#%%

def addFeature(A):
    """
    add high dimensional feature (squared variables)    
    
    """
    I=np.array(range(np.shape(A)[1]))
    newMat=A
    for i in range(int(ceil(np.shape(A)[1]/2.0))):
        newMat=np.hstack((newMat, A*A[:,np.roll(I,i)]))   
    return newMat
#%%
def freqFeature(A):
   newCol=np.zeros(np.shape(A),dtype = int)
   for i in range(np.shape(A)[1]):
       D=Counter([ str(x) for x in A[:,i]])
       temp=np.array([D[str(x)] for x in A[:,i]])
       newCol1 = np.reshape(temp,(np.shape(temp)[0],))
       newCol[:,i]=newCol1
   return newCol
#%%
#%%
def hardCode(yPred, H):
    yPred[list(np.nonzero(H)[0])]=0
    return yPred   
#%%

#loading data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

#%%
numCol =[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 175, 176, 177, 178, 181, 182, 183, 184, 185, 187, 188, 190, 191, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 223, 224, 225, 226, 227, 230, 231, 232, 233, 235, 236, 237, 240, 241, 242, 243, 245, 246, 247, 250, 251, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368]
catCol=[0, 1, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 170, 174, 179, 180, 186, 189, 192, 193, 220, 222, 228, 229, 234, 238, 239, 244, 248, 249, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 349, 350]
#%%
xTrain = train_df.drop(['ID', 'TARGET'], axis = 1)
xTrain = np.array(xTrain)


yTrain = train_df['TARGET']
yTrain = np.array(yTrain)

xTest = test_df.drop('ID',axis = 1)
xTest = np.array(xTest)


xID = test_df['ID']
#%%
xTrainZero = (xTrain == 0).sum(1).reshape(xTrain.shape[0],1)
xTrain = np.hstack((xTrain, xTrainZero))

xTestZero = (xTest == 0).sum(1).reshape(xTest.shape[0],1)
xTest = np.hstack((xTest, xTestZero))
#%% Hardcoding output
var15 = test_df['var15']
saldo_medio_var5_hace2 = test_df['saldo_medio_var5_hace2']
saldo_var33 = test_df['saldo_var33']
var38 = test_df['var38']
var21 = test_df['var21']
H=np.add(np.array(var15<23),np.array(saldo_medio_var5_hace2 > 160000), np.array(var38 > 398859))
H=np.add(H, np.array(saldo_var33 > 0), np.array(var21>7500))
#%%
# check basic stats 
eventRate=sum(yTrain)/float(len(yTrain))
print (eventRate)
#%%
#%%
# feature engineering
#%%      
xTrainOrg=xTrain
xTestOrg=xTest
#%%
xTrainNum=xTrain[:,numCol]
xTrainCat=xTrain[:,catCol]
xTestCat=xTest[:,catCol]
xTestNum=xTest[:,numCol]
#%%    Eliminating rows
selectorNum, xTrainNum = removeConst(xTrainNum)
INum, xTrainNum = removeIden(xTrainNum)
selectorCat, xTrainCat = removeConst(xTrainCat)
ICat, xTrainCat = removeIden(xTrainCat)
xTestCat=selectorCat.transform(xTestCat)[:,ICat]
xTestNum=selectorNum.transform(xTestNum)[:,INum]
#%% Clipping data
trainCatMin, trainCatMax= np.min(xTrainCat, axis=0), np.max(xTrainCat, axis=0)    
trainNumMin, trainNumMax= np.min(xTrainNum, axis=0), np.max(xTrainNum, axis=0) 
xTestCatClip=np.clip(xTestCat, trainCatMin, trainCatMax)
xTestNumClip=np.clip(xTestNum, trainNumMin, trainNumMax)
#%%
#scale numerical data
scaler=StandardScaler()
xTrainNum = scaler.fit_transform(xTrainNum)
xTestNum=scaler.transform(xTestNum)
xTestNumClip=scaler.transform(xTestNumClip)
#%%
xTest=np.hstack((xTestNum,xTestCat))
xTrain=np.hstack((xTrainNum,xTrainCat))
#%%
xTestClip=np.hstack((xTestNumClip,xTestCatClip))
#%%
    
#%%
#xTrainFreq = freqFeature(xTrain)
#xTestFreq = freqFeature(xTest)
#if holdFlag:
#    xHoldFreq = freqFeature(xHold)
##%%
#xTrainAdd = addFeature(xTrain)
#xTestAdd = addFeature(np.array(xTest))
#if holdFlag:
#    xHoldAdd = addFeature(xHold)
##%%
#xTrainQ = preData(xTrain)
#xTestQ = preData(xTest)
#if holdFlag:
#    xHoldQ = preData(xHold)
##%%
#ohe = OneHotEncoder(n_values='auto', handle_unknown= 'ignore' )
#ohe.fit(xTrainQ)
#xTrainOHE = ohe.transform(xTrainQ)
#xTestOHE = ohe.transform(xTestQ)
#if holdFlag:
#    xHoldOHE = ohe.transform(xHoldQ)
##%%
#ohe1 = OneHotEncoder(n_values='auto', handle_unknown= 'ignore' )
#ohe1.fit(xTrain)
#xTrainOHE1 = ohe1.transform(xTrain)
#xTestOHE1 = ohe1.transform(xTest)
#if holdFlag:
#    xHoldOHE1 = ohe1.transform(xHold)
##%%
#model=xgb.XGBClassifier( min_child_weight = 0.5,  colsample_bytree = 0.3, learning_rate=0.25, n_estimators=50, max_depth=5, random_state=2591).fit(xTrainAdd,yTrain)
#featsXGB=model.feature_importances_
#nFeature=100
#I=sorted(heapq.nlargest(nFeature, range(len(featsXGB)), featsXGB.take))
#xTrainXGB=xTrain[:,I]
#xTestXGB=xTest[:,I]
#if holdFlag:
#    xHoldXGB=xHold[:,I]
#%%
##Various classifier 
#########Logreg CV
#print 'inside logreg..'
#if cvFlag:
#    clflr = GridSearchCV(LogisticRegression(), cv=5, 
#                         param_grid={'penalty': ['l2'], 'C':[5e-2,1e-1,0.5e0]}, scoring='roc_auc')
#else:
#    clflr = LogisticRegression(C=1e-1, penalty = 'l2')
#clflr.fit(xTrainOHE, yTrain)
#yPredlr = clflr.predict_proba(xTestOHE)[:,1]
#if holdFlag:
#    yHoldlr = clflr.predict_proba(xHoldOHE)[:,1]
#if cvFlag:
#    printCV(clflr)
##%%
#############RF CV
### n_estimator=300, max_leaf_node=700 best till now
### doing with xTrianQ is not a good idea for now
#print 'inside RF..'
#if cvFlag:
#    clfrf = GridSearchCV(RandomForestClassifier(), cv=5, 
#                         param_grid={ 'max_leaf_nodes':[700],'n_estimators':[300]}, scoring='roc_auc')
#else:
#    clfrf = RandomForestClassifier(max_leaf_nodes =700, n_estimators=300)
#clfrf.fit(xTrain, yTrain)
#yPredrf = clfrf.predict_proba(xTest)[:,1] 
#if holdFlag:
#    yHoldrf = clfrf.predict_proba(xHold)[:,1]
#if cvFlag:
#    printCV(clfrf)
##%%    
#############XGB CV
print ('inside XGB..')
clfxgb = xgb.XGBClassifier(max_depth =5, colsample_bytree =0.3,n_estimators = 574,subsample =0.683,learning_rate =0.0203, seed=1234)
    
#%%
clfxgb.fit(xTrain, yTrain, eval_metric='auc')
yPredxgb = clfxgb.predict_proba(xTestClip)[:,1]
#%%
###############ADA CV

#%%
###############ETC CV
#print 'inside ETC..'
#if cvFlag:
#    clfetc =  GridSearchCV( ExtraTreesClassifier(), cv=5, param_grid={'max_depth': [20], 'n_estimators':[ 500]}, scoring='roc_auc')
#else:
#    clfetc = ExtraTreesClassifier(max_depth=20, n_estimators=500)
#clfetc.fit(xTrain, yTrain)
#yPredetc = clfetc.predict_proba(xTest)[:,1] 
#if holdFlag:
#    yHoldetc = clfetc.predict_proba(xHold)[:,1] 
#if cvFlag:
#    printCV(clfetc)
##%%
################### Linreg CV
#print 'inside linear...'
#if cvFlag:
#    clflin = GridSearchCV( LinearRegression(), cv=5, param_grid={},scoring='roc_auc')
#else:
#    clflin=LinearRegression()
#clflin.fit(xTrainOHE, yTrain)
#yPredlinr = clflin.predict(xTestOHE)
#yPredlinr[yPredlinr>1] = 1
#yPredlinr[yPredlinr<0] = 0
#if holdFlag:
#    yHoldlinr = clflin.predict(xHoldOHE)
#    yHoldlinr[yHoldlinr>1] = 1
#    yHoldlinr[yHoldlinr<0] = 0
#if cvFlag:
#    printCV(clflin)
##%%
############XGB Drop CV
#print 'inside XGB Drop..'
#if cvFlag:
#    clfdrop = GridSearchCV(xgb.XGBClassifier( min_child_weight = 1,  colsample_bytree = 0.3) , cv=5, param_grid={'learning_rate': [0.25], 'max_depth': [5], 'n_estimators':[30, 50]}, scoring='roc_auc')
#else:
#    clfdrop = xgb.XGBClassifier( min_child_weight = 1,  colsample_bytree = 0.3, learning_rate=0.1, n_estimators=100, max_depth=5)
#clfdrop.fit(xTrainDrop, yTrain)
#yPreddrop = clfdrop.predict_proba(xTestDrop)[:,1]
#if holdFlag:
#    yHolddrop = clfdrop.predict_proba(xHoldDrop)[:,1]
#if cvFlag:
#    printCV(clfdrop)
#%%
########Hold Out cell test-------------------
#if holdFlag:
#    ########## Prediction from the classifiers
#    yHoldPred = np.zeros((len(xHold),7))
#    yHoldPred[:,0] = yHoldrf
#    yHoldPred[:,1] = yHoldxgb
#    yHoldPred[:,2] = yHoldlr
#    yHoldPred[:,3] = yHoldetc
#    yHoldPred[:,4] = yHoldada
#    yHoldPred[:,5] = yHoldlinr
#    yHoldPred[:,6] = yHolddrop
#    ###----------------------------
#    wght1=np.array([3,15,1,1,2,1,3])/26.0
#    yHoldENS1 = np.dot(yHoldPred, wght1)
#    print '1:', roc_auc_score(yHold, yHoldENS1)
#    wght2=np.array([3,12,1,1,1,1,5])/24.0
#    yHoldENS2 = np.dot(yHoldPred, wght2)
#    print '2:', roc_auc_score(yHold, yHoldENS2)
#    ####-----------------
#    N=100
#    SEED=random.randint(0, 2000)
#    print 'SEED is:', SEED
#    bestreg=1e0
#    bestauc=0
#    for reg in [1e3, 1e4, 1e5]:
#        model=NNLSClassifier(alpha=reg)
#        A=[]
#        for i in xrange(N): 
#            P_train, P_cv, y_train, y_cv = train_test_split( yHoldPred, yHold, test_size=.8, random_state=SEED*i)
#            model.fit(P_train, y_train)
#            preds = model.predict(P_cv)
#            A += [roc_auc_score(y_cv, preds)]
#        print np.mean(A)
#        if (bestauc< np.mean(A)):
#            bestauc=np.mean(A)
#            bestreg=reg
#            modelFinal=model
#    wght3=modelFinal.weights_
#    yHoldENS3 = np.dot(yHoldPred, wght3)
#    print '3:', roc_auc_score(yHold, yHoldENS3)
#    ### Some more ensemble
#    A=[]
#    Pref=[1,0,4,5,3,6,2]
#    for i in range(1000):
#        wghts=np.array(sorted(5*np.exp(np.random.rand(7)), reverse=True))
#        wghts=wghts/sum(wghts)
#        wghts=wghts[Pref]
#        A += [[wghts, roc_auc_score(yHold, np.dot(yHoldPred,wghts))]]
#    num=30
#    A=np.array(A)
#    R=np.array([x[1] for x in A])
#    I=sorted(heapq.nlargest(num, range(len(R)), R.take))
#    wght4=np.mean(A[I,0], axis=0) 
#    yHoldENS4 = np.dot(yHoldPred, wght4)
#    print '4:', roc_auc_score(yHold, yHoldENS4)
#    # Final
#    yHoldENS=np.array([yHoldENS1,yHoldENS2,yHoldENS3,yHoldENS4]).T
#    A=[]
#    Pref=[3,0,1,2]
#    for i in range(1000):
#        wghts=np.array(sorted(10*np.exp(np.random.rand(4)), reverse=True))
#        wghts=wghts/sum(wghts)
#        wghts=wghts[Pref]
#        A += [[wghts, roc_auc_score(yHold, np.dot(yHoldENS,wghts))]]
#    num=30
#    A=np.array(A)
#    R=np.array([x[1] for x in A])
#    I=sorted(heapq.nlargest(num, range(len(R)), R.take))
#    wghtFinal=np.mean(A[I,0], axis=0)  
#    yFinal = np.dot(yHoldENS, wghtFinal) 
#    print 'Final:', roc_auc_score(yHold, yFinal)                
###------scores----------------
### Optimized by holding set
#wght1=np.array([3,25,1,1,2,1,3])/36.0
#wght2=np.array([5,22,1,1,1,1,5])/36.0
#wght3=modelFinal.weights_
#wght4=[0.1184577,   0.58915003,  0.05853682,  0.0291931,   0.08072655,  0.01613224, 0.10780357]
#wghtFinal= [0.1380594,   0.33408457,  0.28406904,  0.243787]
####Prediction---------------
########### Prediction from the classifiers
#yTestPred = np.zeros((len(xTest),7))
#yTestPred[:,0] = yPredrf
#yTestPred[:,1] = yPredxgb
#yTestPred[:,2] = yPredlr
#yTestPred[:,3] = yPredetc
#yTestPred[:,4] = yPredada
#yTestPred[:,5] = yPredlinr
#yTestPred[:,6] = yPreddrop
#yENS1 = np.dot(yTestPred, wght1)
#yENS2 = np.dot(yTestPred, wght2)
#yENS3 = np.dot(yTestPred, wght3)
#yENS4 = np.dot(yTestPred, wght4) 
### Final 
#yENS = np.array([yENS1,yENS2,yENS3,yENS4]).T
#%%
yFinal=yPredxgb#np.dot(yENS, wghtFinal)
yFinal=hardCode(yFinal,H)
#####-------------------
df1 = pd.DataFrame({'ID':xID, 'TARGET':yFinal})
df1.to_csv("santander_XGBhardcoded_fullZerocountClipped_txtxgb1_.csv",index=False)
#
