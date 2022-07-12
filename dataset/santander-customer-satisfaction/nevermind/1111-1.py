import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time

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

verySimpleLearner = ensemble.GradientBoostingClassifier(n_estimators=10, max_features=1, max_depth=3, min_samples_leaf=100,
                                                        learning_rate=0.3, subsample=0.65, loss='deviance', random_state=1)

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, test_size=0.5, random_state=1)
        
startTime = time.time()
singleFeatureAUC_list = []
singleFeatureAUC_dict = {}
for feature in X_train.columns:
    trainInputFeature = X_train[feature].values.reshape(-1,1)
    validInputFeature = X_valid[feature].values.reshape(-1,1)
    verySimpleLearner.fit(trainInputFeature, y_train)
    
    trainAUC = auc(y_train, verySimpleLearner.predict_proba(trainInputFeature)[:,1])
    validAUC = auc(y_valid, verySimpleLearner.predict_proba(validInputFeature)[:,1])
        
    singleFeatureAUC_list.append(validAUC)
    singleFeatureAUC_dict[feature] = validAUC
        
validAUC = np.array(singleFeatureAUC_list)
timeToTrain = (time.time()-startTime)/60
print("(min,mean,max) AUC = (%.3f,%.3f,%.3f). took %.2f minutes" %(validAUC.min(),validAUC.mean(),validAUC.max(), timeToTrain))

# show the scatter plot of the individual feature performance 
plt.figure(); plt.hist(validAUC, 50, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('single feature AUC histogram'); plt.show()

# create a table with features sorted according to AUC
singleFeatureTable = pd.DataFrame(index=range(len(singleFeatureAUC_dict.keys())), columns=['feature','AUC'])
for k,key in enumerate(singleFeatureAUC_dict):
    singleFeatureTable.ix[k,'feature'] = key
    singleFeatureTable.ix[k,'AUC'] = singleFeatureAUC_dict[key]
singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

singleFeatureTable.ix[:15,:]
numSubPlotRows = 1
numSubPlotCols = 2
for plotInd in range(8):
    plt.figure()
    for k in range(numSubPlotRows*numSubPlotCols):
        tableRow = numSubPlotRows*numSubPlotCols*plotInd+k
        x = X_train[singleFeatureTable.ix[tableRow,'feature']].values.reshape(-1,1)[:,0]
        
        # use a huristic to find out if the variable is categorical, and if so add some random noise to it
        if np.unique(len(np.unique(x))) < 20:
            diffVec = abs(x[1:]-x[:-1])
            minDistBetweenCategories = min(diffVec[diffVec > 0])
            x = x + 0.12*minDistBetweenCategories*np.random.randn(np.shape(x)[0])
            
        y = y_train + 0.12*np.random.randn(np.shape(y_train)[0])
        # take only 3000 samples to be presented due to plotting issues
        randPermutation = np.random.choice(len(x), 3000, replace=False)
        plt.subplot(numSubPlotRows,numSubPlotCols,k+1)
        plt.scatter(x[randPermutation], y[randPermutation], c=y_train[randPermutation], cmap='jet', alpha=0.25)
        plt.xlabel(singleFeatureTable.ix[tableRow,'feature']); plt.ylabel('y GT')
        plt.title('AUC = %.4f' %(singleFeatureTable.ix[tableRow,'AUC']))            
        plt.ylim(-0.5,1.5); plt.tight_layout()
    plt.show()
    
verySimpleLearner2 = ensemble.GradientBoostingClassifier(n_estimators=10, max_features=2, max_depth=5, min_samples_leaf=100,
                                                        learning_rate=0.3322, subsample=0.65, loss='deviance', random_state=1)

# limit run time (on all feature combinations should take a few hours)
numFeaturesToUse = 20
featuresToUse = singleFeatureTable.ix[0:numFeaturesToUse-1,'feature']

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, test_size=0.5, random_state=1)
    
startTime = time.time()
featurePairAUC_list = []
featurePairAUC_dict = {}

for feature1Ind in range(len(featuresToUse)-1):
    featureName1 = featuresToUse[feature1Ind]
    trainInputFeature1 = X_train[featureName1].values.reshape(-1,1)
    validInputFeature1 = X_valid[featureName1].values.reshape(-1,1)

    for feature2Ind in range(feature1Ind+1,len(featuresToUse)-1):
        featureName2 = featuresToUse[feature2Ind]
        trainInputFeature2 = X_train[featureName2].values.reshape(-1,1)
        validInputFeature2 = X_valid[featureName2].values.reshape(-1,1)
        
        for feature3Ind in range(feature2Ind+1,len(featuresToUse)-2):
            featureName3 = featuresToUse[feature3Ind]
            trainInputFeature3 = X_train[featureName3].values.reshape(-1,1)
            validInputFeature3 = X_valid[featureName3].values.reshape(-1,1)
            for feature4Ind in range(feature3Ind+1,len(featuresToUse)-3):
                featureName4 = featuresToUse[feature4Ind]
                trainInputFeature4 = X_train[featureName4].values.reshape(-1,1)
                validInputFeature4 = X_valid[featureName4].values.reshape(-1,1)

        trainInputFeatures = np.hstack((trainInputFeature1,trainInputFeature2,trainInputFeature3,trainInputFeature4))
        validInputFeatures = np.hstack((validInputFeature1,validInputFeature2,validInputFeature3,validInputFeature4))
        
        verySimpleLearner2.fit(trainInputFeatures, y_train)
        
        trainAUC = auc(y_train, verySimpleLearner2.predict_proba(trainInputFeatures)[:,1])
        validAUC = auc(y_valid, verySimpleLearner2.predict_proba(validInputFeatures)[:,1])
            
        featurePairAUC_list.append(validAUC)
        featurePairAUC_dict[(featureName1,featureName2,featureName3,featureName4)] = validAUC
        
validAUC = np.array(featurePairAUC_list)
timeToTrain = (time.time()-startTime)/60
print("(min,mean,max) AUC = (%.3f,%.3f,%.3f). took %.1f minutes" % (validAUC.min(),validAUC.mean(),validAUC.max(), timeToTrain))
