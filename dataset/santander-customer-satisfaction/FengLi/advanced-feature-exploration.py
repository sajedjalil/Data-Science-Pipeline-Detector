# # Advanced Feature Exploration
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
# ### Show single feature AUC performace
# this is the same as in "Basic Feature Exploration" script, to be used later
#%% sort according to AUC and present the table
singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

singleFeatureTable.ix[:15,:]
# ### Generate 400 five-wise random feature combinations and calculate their AUC
#%% find interesting fivewise combinations

numFeaturesInCombination = 3
numCombinations = 400
numBestSingleFeaturesToSelectFrom = 25

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, 
                                                                       test_size=0.5, random_state=1)
weakLearner = ensemble.GradientBoostingClassifier(n_estimators=30, max_features=2, max_depth=3, 
                                                  min_samples_leaf=100,learning_rate=0.1, 
                                                  subsample=0.65, loss='deviance', random_state=1)

featuresToUse = singleFeatureTable.ix[0:numBestSingleFeaturesToSelectFrom-1,'feature']
featureColumnNames = ['feature'+str(x+1) for x in range(numFeaturesInCombination)]
featureCombinationsTable = pd.DataFrame(index=range(numCombinations), columns=featureColumnNames + ['combinedAUC'])

# for numCombinations iterations 
startTime = time.time()
for combination in range(numCombinations):
    # generate random feature combination
    randomSelectionOfFeatures = sorted(np.random.choice(len(featuresToUse), numFeaturesInCombination, replace=False))

    # store the feature names
    combinationFeatureNames = [featuresToUse[x] for x in randomSelectionOfFeatures]
    for i in range(len(randomSelectionOfFeatures)):
        featureCombinationsTable.ix[combination,featureColumnNames[i]] = combinationFeatureNames[i]

    # build features matrix to get the combination AUC
    trainInputFeatures = X_train.ix[:,combinationFeatureNames]
    validInputFeatures = X_valid.ix[:,combinationFeatureNames]
    # train learner
    weakLearner.fit(trainInputFeatures, y_train)
    # store AUC results
    validAUC = auc(y_valid, weakLearner.predict_proba(validInputFeatures)[:,1])        
    featureCombinationsTable.ix[combination,'combinedAUC'] = validAUC

validAUC = np.array(featureCombinationsTable.ix[:,'combinedAUC'])
print("(min,max) AUC = (%.4f,%.4f). took %.1f minutes" % (validAUC.min(),validAUC.max(), (time.time()-startTime)/60))

# show the histogram of the feature combinations performance 
plt.figure(); plt.hist(validAUC, 100, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('feature combination AUC histogram'); plt.show()
#%% sort according to combination AUC and look at the table

featureCombinationsTable = featureCombinationsTable.sort_values(by='combinedAUC', axis=0, ascending=False).reset_index(drop=True)
featureCombinationsTable.ix[:20,:]
# it's easy to see that this table contains a lot of feature overlap
# ### Visualize this by building a Pairwise Overlap Matrix
#%% visualize the overlap by building a pairwise overlap matrix

combinationOverlapMatrix = np.zeros((numCombinations,numCombinations))
for comb_i in range(numCombinations):
    for comb_j in range(comb_i+1,numCombinations):
        # get the features list for each combination        
        featuresComb_i = [featureCombinationsTable.ix[comb_i,featureColumnNames[x]] for x in range(numFeaturesInCombination)]
        featuresComb_j = [featureCombinationsTable.ix[comb_j,featureColumnNames[x]] for x in range(numFeaturesInCombination)]
        # store the number of overlapping features
        combinationOverlapMatrix[comb_i,comb_j] = 2*numFeaturesInCombination-len(set(featuresComb_i+featuresComb_j))
        combinationOverlapMatrix[comb_j,comb_i] = combinationOverlapMatrix[comb_i,comb_j]

plt.figure(); plt.imshow(combinationOverlapMatrix,cmap='autumn'); plt.title('combination overlap'); plt.colorbar()
# #### We would like to remove some of this redundancy
# ### Perform k-means on the overlap patterns and reorder the matrix
#%% we would like to get the top performing but most different feature combinations

numFeaturesToSelect = 25

cluserer = cluster.KMeans(n_clusters=numFeaturesToSelect)
clusterInds = cluserer.fit_predict(combinationOverlapMatrix)

#%% reorder features according to their new clusters

# group the rows into clusters
clusteredRows = {}
clusterMaxAUC = {}
clusterMaxInd = {}
for clusterInd in np.unique(clusterInds):
    clusteredRows[clusterInd] = combinationOverlapMatrix[clusterInds == clusterInd,:]
    clusterMaxAUC[clusterInd] = featureCombinationsTable.ix[clusterInds == clusterInd,'combinedAUC'].max(axis=0)
    clusterMaxInd[clusterInd] = featureCombinationsTable.ix[clusterInds == clusterInd,'combinedAUC'].idxmax(axis=0)    
    
import operator    
sortedClustersByMaxAUCTuple = sorted(clusterMaxAUC.items(), key=operator.itemgetter(1),reverse=True)

# calculate the reordering vector
finalFeaturesToKeep = []
reorderedVector = None
for k,item in enumerate(sortedClustersByMaxAUCTuple):
    if k == 0:
        reorderedVector = np.array((clusterInds == item[0]).nonzero())
    else:
        reorderedVector = np.hstack((reorderedVector,np.array((clusterInds == item[0]).nonzero())))
    finalFeaturesToKeep.append(clusterMaxInd[item[0]])
reorderedVector = reorderedVector.flatten()

# reorder the matrix by rows and columns
reorderedMatrix = combinationOverlapMatrix[reorderedVector,:]
reorderedMatrix = reorderedMatrix[:,reorderedVector]

# show the matrix
plt.figure(); plt.imshow(reorderedMatrix,cmap='autumn'); plt.title('reordered combination overlap'); plt.colorbar()

# # End Result
# ### The 15 Best least redundent five-wise feature combinations
#%% show the final combinations

featureCombinationsTable.ix[finalFeaturesToKeep,:]