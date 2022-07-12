import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time

plt.rcParams['figure.figsize'] = (10, 10)

#%% load data and remove constant and duplicate columns  (taken from a kaggle script)

trainDataFrame = pd.read_csv('../input/train.csv')

# remove constant columns
colsToRemove = []
for col in trainDataFrame.columns:
    if trainDataFrame[col].std() == 0:
        colsToRemove.append(col)

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = trainDataFrame.columns
for i in range(len(columns)-1):
    v = trainDataFrame[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,trainDataFrame[columns[j]].values):
            colsToRemove.append(columns[j])

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

trainLabels = trainDataFrame['TARGET']
trainFeatures = trainDataFrame.drop(['ID','TARGET'], axis=1)


#%% look at single feature performance

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, 
                                                                       test_size=0.5, random_state=1)
verySimpleLearner = ensemble.GradientBoostingClassifier(n_estimators=20, max_features=1, max_depth=3, 
                                                        min_samples_leaf=100, learning_rate=0.1, 
                                                        subsample=0.65, loss='deviance', random_state=1)

startTime = time.time()
singleFeatureTable = pd.DataFrame(index=range(len(X_train.columns)), columns=['feature','AUC'])
for k,feature in enumerate(X_train.columns):
    trainInputFeature = X_train[feature].values.reshape(-1,1)
    validInputFeature = X_valid[feature].values.reshape(-1,1)
    verySimpleLearner.fit(trainInputFeature, y_train)
    
    validAUC = auc(y_valid, verySimpleLearner.predict_proba(validInputFeature)[:,1])
    singleFeatureTable.ix[k,'feature'] = feature
    singleFeatureTable.ix[k,'AUC'] = validAUC
        
print("finished evaluating single features. took %.2f minutes" %((time.time()-startTime)/60))

singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

singleFeatureTable.ix[:100,:]