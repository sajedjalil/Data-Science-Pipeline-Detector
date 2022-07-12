import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import time
from sklearn import preprocessing, ensemble
from sklearn import feature_extraction
from sklearn import pipeline, metrics, grid_search
from sklearn.utils import shuffle
from scipy.sparse import hstack
from sklearn.cross_validation import KFold, train_test_split,StratifiedKFold
from scipy.stats import rankdata


def load_data():
    print ("Loading data ......")
    start = time.time()
    DV = feature_extraction.DictVectorizer(sparse=False)
    ENC = preprocessing.OneHotEncoder(sparse = False)
    POLY = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    LBL = preprocessing.LabelEncoder()

    labelCol =['Hazard']
    idCol =['Id']
    catCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']    
    numCols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14',  'T2_V1','T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14','T2_V15']
    oneWayCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12' , 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']  
    twoWayCols = [['T1_V17','T2_V12'],['T1_V5','T1_V9'],['T2_V13','T2_V11']] #T1_V5 T1_V9 , T1_V5 T1_V16, T2_V13 T2_V11    
    oneWayCatAggrCols = []
    oneWayNumAggrCols = []
    tweWayAggrCols = []

    #Load data
    trainData = pd.read_csv('../input/train.csv')
    testData = pd.read_csv('../input/test.csv')
    fullData=pd.concat([trainData,testData])
    trainSize = trainData.shape[0]
    testSize = testData.shape[0]
    print ("Train: %d rows loaded" % (trainSize))
    print ("Test: %d rows loaded" % (testSize))

    #Label Encoder
    for col in catCols:
        fullData[col]=LBL.fit_transform(fullData[col])
        print (col) 


    for col in catCols:
        # mean label and num cols
        aggr = fullData.groupby(col)[labelCol].agg(np.mean)
        aggr.rename(columns=lambda c: ('MEAN_'+c+'_BY_'+col).upper(), inplace = True) 
        fullData = fullData.join(aggr,how='left', on=col)        # mean numberic columns
        print (col)        

    #Encode columns
    fullEncCat = pd.get_dummies(fullData[catCols],columns=catCols)
    fullEncNum = pd.get_dummies(fullData[numCols],columns=numCols)
    dummyCatCols = list(fullEncCat.columns)
    dummyNumCols = list(fullEncNum.columns)
    
    fullData = pd.concat([fullData.reset_index(), fullEncCat.reset_index(), fullEncNum.reset_index()],axis=1)

    print ("Loading finished in %0.3fs" % (time.time() - start))    
    return fullData[:trainSize].drop(labelCol+idCol,axis=1).values, fullData[:trainSize][labelCol].values, fullData[trainSize:].drop(labelCol+idCol,axis=1).values, fullData[trainSize:][idCol].values


def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini  

def evalgini(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'Gini', -normalized_gini(labels, preds)    


def xgboost_pred(train,labels,test,test_labels, params):

    plst = list(params.items())

    # offset = 4000
    offset = int(train.shape[0]*0.10)
    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xtrain = xgb.DMatrix(train, labels)
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(params=plst, dtrain=xgtrain, num_boost_round=num_rounds, evals=watchlist,  early_stopping_rounds=80)
    print ("Best score:", model.best_score)
    print ("Best iteration:", model.best_iteration)
    #model = xgb.train(plst, xtrain, model.best_iteration)
    preds1 = model.predict(xgtest)

    print ("score1: %f" % (normalized_gini(test_labels,preds1)))


    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
    train = train[::-1,:]
    # labels = np.log(labels[::-1])
    labels = labels[::-1]
    xtrain = xgb.DMatrix(train, labels)

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(params=plst, dtrain=xgtrain, num_boost_round=num_rounds, evals=watchlist,  early_stopping_rounds=80)
    print ("Best score:", model.best_score)
    print ("Best iteration:", model.best_iteration)
    model = xgb.train(plst, xtrain, model.best_iteration)
    preds2 = model.predict(xgtest)
    print ("score2: %f" % (normalized_gini(test_labels,preds2)))
    

    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    preds = preds1*2.5 + preds2*7.5
    return preds,preds1,preds2

if __name__ == '__main__':
    #load train and test 
    # trainX, trainY, testX, testY, testId = load_data()
    trainX, trainY, testX, testId = load_data()
    #Using 10000 samples
    # idx = np.random.choice(trainX.shape[0],10000)
    
    trainY = rankdata(trainY.reshape(trainY.shape[0]),method = 'dense')
    testId = testId.reshape(testId.shape[0])
    
    # trainX=trainX[30000:40000]
    # trainY=trainY[30000:40000]

    start = time.time()
    params = {}
    params["objective"] = "rank:pairwise"
	#params["objective"] = "rank:pairwise"
    #params["eval_metric"] = "rmse"
    params["eta"] = 0.01
    params["min_child_weight"] = 100
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 9    
    # params["eval_metric"] = 'evalgini'   

    skf = StratifiedKFold(trainY, n_folds=4, random_state  = 42)
    # kf = KFold(trainX.shape[0], n_folds=4,shuffle=True, random_state  = 42 )
    gini_scores = []
    for train_index, test_index in skf:
        #Splict train set into k folds
        X_train_fold, X_test_fold = trainX[train_index], trainX[test_index]
        y_train_fold, y_test_fold = trainY[train_index], trainY[test_index]    
        y_pred_fold,y_pred_fold1,y_pred_fold2 = xgboost_pred(X_train_fold,y_train_fold,X_test_fold,y_test_fold, params)
    
    gini_scores=np.array(gini_scores)   
    mean_ginis = []
    for i in gini_scores.T:
        print (np.mean(i))
        mean_ginis.append(np.mean(i))
    mean_ginis =  np.array(mean_ginis)
    print (mean_ginis)
    print ("Maximum score:", np.max(mean_ginis))
    print ("Finished in %0.3fs" % (time.time() - start))   
