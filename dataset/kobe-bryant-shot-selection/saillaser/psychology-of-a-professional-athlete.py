# Psychology of a Professional Athlete
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
import time

#%% load training data

allData = pd.read_csv('../input/data.csv')
data = allData[allData['shot_made_flag'].notnull()].reset_index()
#%% add some temporal columns to the data

data['game_date_DT'] = pd.to_datetime(data['game_date'])
data['dayOfWeek'] = data['game_date_DT'].dt.dayofweek
data['dayOfYear'] = data['game_date_DT'].dt.dayofyear

data['secondsFromPeriodEnd'] = 60*data['minutes_remaining']+data['seconds_remaining']
data['secondsFromPeriodStart'] = 60*(11-data['minutes_remaining'])+(60-data['seconds_remaining'])
data['secondsFromGameStart'] = (data['period'] <= 4).astype(int)*(data['period']-1)*12*60 + (data['period'] > 4).astype(int)*((data['period']-4)*5*60 + 3*12*60) + data['secondsFromPeriodStart']

# look at first couple of rows and verify that everything is good
data.ix[:20,['period','minutes_remaining','seconds_remaining','secondsFromGameStart']]
# ### Kobe is entrusted to take the last shot of every period
# it also looks like he's usually on the bench at the start of 2nd and 4th periods
#%% plot the shot attempts as a function of time (from start of game) with several different binnings
plt.rcParams['figure.figsize'] = (16, 10)

binsSizes = [24,12,6]

plt.figure();
for k, binSizeInSeconds in enumerate(binsSizes):
    timeBins = np.arange(0,60*(4*12+3*5),binSizeInSeconds)+0.01
    attemptsAsFunctionOfTime, b = np.histogram(data['secondsFromGameStart'], bins=timeBins)     
    
    maxHeight = max(attemptsAsFunctionOfTime) + 30
    barWidth = 0.999*(timeBins[1]-timeBins[0])
    plt.subplot(len(binsSizes),1,k+1); 
    plt.bar(timeBins[:-1],attemptsAsFunctionOfTime, align='edge', width=barWidth); plt.title(str(binSizeInSeconds) + ' second time bins')
    plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')
    plt.xlim((-20,3200)); plt.ylim((0,maxHeight)); plt.ylabel('attempts')
plt.xlabel('time [seconds from start of game]')

# ### Note that the accuracy of these shots is consisently lower than usuall
# this is probably due to the fact that a large amonut of these shots are from very far away 
#%% plot the accuracy as a function of time
plt.rcParams['figure.figsize'] = (15, 10)

binSizeInSeconds = 20
timeBins = np.arange(0,60*(4*12+3*5),binSizeInSeconds)+0.01
attemptsAsFunctionOfTime, b = np.histogram(data['secondsFromGameStart'], bins=timeBins)     
madeAttemptsAsFunctionOfTime, b = np.histogram(data.ix[data['shot_made_flag']==1,'secondsFromGameStart'], bins=timeBins)     
accuracyAsFunctionOfTime = madeAttemptsAsFunctionOfTime.astype(float)/attemptsAsFunctionOfTime
accuracyAsFunctionOfTime[attemptsAsFunctionOfTime <= 50] = 0 # zero accuracy in bins that don't have enough samples

maxHeight = max(attemptsAsFunctionOfTime) + 30
barWidth = 0.999*(timeBins[1]-timeBins[0])
 
plt.figure();
plt.subplot(2,1,1); plt.bar(timeBins[:-1],attemptsAsFunctionOfTime, align='edge', width=barWidth); 
plt.xlim((-20,3200)); plt.ylim((0,maxHeight)); plt.ylabel('attempts'); plt.title(str(binSizeInSeconds) + ' second time bins')
plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')
plt.subplot(2,1,2); plt.bar(timeBins[:-1],accuracyAsFunctionOfTime, align='edge', width=barWidth); 
plt.xlim((-20,3200)); plt.ylabel('accuracy'); plt.xlabel('time [seconds from start of game]')
plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0.0,ymax=0.7, colors='r')

# ### For later analysis, we'll want to assess shot difficulty based on shot properties
# (such as shot type and shot distance)
#%% create a new table for shot difficulty model

def FactorizeCategoricalVariable(inputDB,categoricalVarName):
    opponentCategories = inputDB[categoricalVarName].value_counts().index.tolist()
    
    outputDB = pd.DataFrame()
    for category in opponentCategories:
        featureName = categoricalVarName + ': ' + str(category)
        outputDB[featureName] = (inputDB[categoricalVarName] == category).astype(int)

    return outputDB

featuresDB = pd.DataFrame()
featuresDB['homeGame'] = data['matchup'].apply(lambda x: 1 if (x.find('@') < 0) else 0)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'opponent')],axis=1)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'action_type')],axis=1)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_type')],axis=1)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'combined_shot_type')],axis=1)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_zone_basic')],axis=1)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_zone_area')],axis=1)
featuresDB = pd.concat([featuresDB,FactorizeCategoricalVariable(data,'shot_zone_range')],axis=1)

featuresDB['playoffGame'] = data['playoffs']
featuresDB['locX'] = data['loc_x']
featuresDB['locY'] = data['loc_y']
featuresDB['distanceFromBasket'] = data['shot_distance']
featuresDB['secondsFromPeriodEnd'] = data['secondsFromPeriodEnd']

featuresDB['dayOfWeek_cycX'] = np.sin(2*np.pi*(data['dayOfWeek']/7))
featuresDB['dayOfWeek_cycY'] = np.cos(2*np.pi*(data['dayOfWeek']/7))
featuresDB['timeOfYear_cycX'] = np.sin(2*np.pi*(data['dayOfYear']/365))
featuresDB['timeOfYear_cycY'] = np.cos(2*np.pi*(data['dayOfYear']/365))

labelsDB = data['shot_made_flag']

# ## Build a model based on featuresDB table, and make sure it doesn't overfit 
# (i.e. the training error and the test error are the same)
# #### Use an ExtraTreesClassifier for that
#%% build a simple model and make sure it doesnt overfit

randomSeed = 1
numFolds = 4

mainLearner = ensemble.ExtraTreesClassifier(n_estimators=500, max_depth=5, 
                                            min_samples_leaf=100, max_features=100, 
                                            criterion='entropy', bootstrap=False, 
                                            n_jobs=2, random_state=randomSeed)
                        
crossValidationIterator = cross_validation.StratifiedKFold(labelsDB, n_folds=numFolds, 
                                                           shuffle=True, random_state=randomSeed)

startTime = time.time()
trainAccuracy = []; validAccuracy = [];
trainLogLosses = []; validLogLosses = []
for trainInds, validInds in crossValidationIterator:
    # split to train and valid sets
    X_train_CV = featuresDB.ix[trainInds,:]
    y_train_CV = labelsDB.iloc[trainInds]
    X_valid_CV = featuresDB.ix[validInds,:]
    y_valid_CV = labelsDB.iloc[validInds]
    
    # train learner
    mainLearner.fit(X_train_CV, y_train_CV)
    
    # make predictions
    y_train_hat_mainLearner = mainLearner.predict_proba(X_train_CV)[:,1]
    y_valid_hat_mainLearner = mainLearner.predict_proba(X_valid_CV)[:,1]

    # store results
    trainAccuracy.append(accuracy(y_train_CV, y_train_hat_mainLearner > 0.5))
    validAccuracy.append(accuracy(y_valid_CV, y_valid_hat_mainLearner > 0.5))
    trainLogLosses.append(log_loss(y_train_CV, y_train_hat_mainLearner))
    validLogLosses.append(log_loss(y_valid_CV, y_valid_hat_mainLearner))

print("-----------------------------------------------------")
print("total (train,valid) Accuracy = (%.5f,%.5f). took %.2f minutes" % (np.mean(trainAccuracy),np.mean(validAccuracy), (time.time()-startTime)/60))
print("total (train,valid) Log Loss = (%.5f,%.5f). took %.2f minutes" % (np.mean(trainLogLosses),np.mean(validLogLosses), (time.time()-startTime)/60))
print("-----------------------------------------------------")

# ### Use the model to add a "shotDifficulty" field to every original shot entry
# (which is actually the predicted probability of making the shot. meaning, the name is a bit confusing right now)
# ### Also, to get a feel for the important features, let's look at the feature importances according to ET Classifier
mainLearner.fit(featuresDB, labelsDB)
data['shotDifficulty'] = mainLearner.predict_proba(featuresDB)[:,1]

# just to get a feel for what determins shot difficulty, look at feature importances
featureInds = mainLearner.feature_importances_.argsort()[::-1]
featureImportance = pd.DataFrame(np.concatenate((featuresDB.columns[featureInds,None], mainLearner.feature_importances_[featureInds,None]), axis=1),
                                  columns=['featureName', 'importanceET'])

featureImportance.ix[:30,:]
# ## We would like to asses some aspects of the decision making process of Kobe Bryant
# ### For that we will collect two distinct groups of shots and analyse the differences between them:
# 
# (1) the shots that came right after a sucessful shot attempt
# 
# (2) the shots that came right after a miss
#%% collect data given that kobe made or missed last shot

timeBetweenShotsDict = {}
timeBetweenShotsDict['madeLast'] = []
timeBetweenShotsDict['missedLast'] = []

changeInDistFromBasketDict = {}
changeInDistFromBasketDict['madeLast'] = []
changeInDistFromBasketDict['missedLast'] = []

changeInShotDifficultyDict = {}
changeInShotDifficultyDict['madeLast'] = []
changeInShotDifficultyDict['missedLast'] = []

totalMadeAfterMade = 0
totalAttemptsAfterMade = 0
shotChancesListAfterMade = []

totalMadeAfterMissed = 0
totalAttemptsAfterMissed = 0
shotChancesListAfterMissed = []

for shot in range(1,data.shape[0]):

    # make sure the current shot and last shot were all in the same period of the same game
    sameGame   = data.ix[shot,'game_date'] == data.ix[shot-1,'game_date']
    samePeriod = data.ix[shot,'period']    == data.ix[shot-1,'period']

    if samePeriod and sameGame:
        madeLastShot       = data.ix[shot-1,'shot_made_flag'] == 1
        missedLastShot     = data.ix[shot-1,'shot_made_flag'] == 0
        
        timeDifferenceFromLastShot = data.ix[shot,'secondsFromGameStart']     - data.ix[shot-1,'secondsFromGameStart']
        distDifferenceFromLastShot = data.ix[shot,'shot_distance']            - data.ix[shot-1,'shot_distance']
        shotDifficultyDifferenceFromLastShot = data.ix[shot,'shotDifficulty'] - data.ix[shot-1,'shotDifficulty']

        # check for currupt data points (assuming all samples should have been chronologically ordered)
        if timeDifferenceFromLastShot < 0:
            continue
        
        if madeLastShot:
            timeBetweenShotsDict['madeLast'].append(timeDifferenceFromLastShot)
            changeInDistFromBasketDict['madeLast'].append(distDifferenceFromLastShot)
            changeInShotDifficultyDict['madeLast'].append(shotDifficultyDifferenceFromLastShot)
            
            # store also statistics about weather the shots actually went in
            shotChancesListAfterMade.append(data.ix[shot,'shotDifficulty'])
            if data.ix[shot,'shot_made_flag'] == 1:
                totalMadeAfterMade += 1
            totalAttemptsAfterMade += 1
            
        if missedLastShot:
            timeBetweenShotsDict['missedLast'].append(timeDifferenceFromLastShot)
            changeInDistFromBasketDict['missedLast'].append(distDifferenceFromLastShot)
            changeInShotDifficultyDict['missedLast'].append(shotDifficultyDifferenceFromLastShot)
            
            # store also statistics about weather the shots actually went in
            shotChancesListAfterMissed.append(data.ix[shot,'shotDifficulty'])
            if data.ix[shot,'shot_made_flag'] == 1:
                totalMadeAfterMissed += 1
            totalAttemptsAfterMissed += 1

# ### Plot histogram of "Time Since Last Shot Attempt" for the two groups
# It looks like after making a shot, kobe is a little bit more eager to throw the next shot
#%% after making a shot, kobe wants more
plt.rcParams['figure.figsize'] = (12, 8)

jointHist, timeBins = np.histogram(timeBetweenShotsDict['madeLast']+timeBetweenShotsDict['missedLast'],bins=200)
barWidth = 0.999*(timeBins[1]-timeBins[0])

timeDiffHist_GivenMadeLastShot, b = np.histogram(timeBetweenShotsDict['madeLast'],bins=timeBins)
timeDiffHist_GivenMissedLastShot, b = np.histogram(timeBetweenShotsDict['missedLast'],bins=timeBins)
maxHeight = max(max(timeDiffHist_GivenMadeLastShot),max(timeDiffHist_GivenMissedLastShot)) + 30

plt.figure();
plt.subplot(2,1,1); plt.bar(timeBins[:-1], timeDiffHist_GivenMadeLastShot, width=barWidth); plt.xlim((0,500)); plt.ylim((0,maxHeight))
plt.title('made last shot'); plt.ylabel('counts')
plt.subplot(2,1,2); plt.bar(timeBins[:-1], timeDiffHist_GivenMissedLastShot, width=barWidth); plt.xlim((0,500)); plt.ylim((0,maxHeight))
plt.title('missed last shot'); plt.xlabel('time since last shot'); plt.ylabel('counts')

# To better visualize this difference, let's look at cumulative histograms
#%% to make the difference clearer, show the cumulative histogram
plt.rcParams['figure.figsize'] = (12, 8)

timeDiffCumHist_GivenMadeLastShot = np.cumsum(timeDiffHist_GivenMadeLastShot).astype(float)
timeDiffCumHist_GivenMadeLastShot = timeDiffCumHist_GivenMadeLastShot/max(timeDiffCumHist_GivenMadeLastShot)
timeDiffCumHist_GivenMissedLastShot = np.cumsum(timeDiffHist_GivenMissedLastShot).astype(float)
timeDiffCumHist_GivenMissedLastShot = timeDiffCumHist_GivenMissedLastShot/max(timeDiffCumHist_GivenMissedLastShot)

maxHeight = max(timeDiffCumHist_GivenMadeLastShot[-1],timeDiffCumHist_GivenMissedLastShot[-1])

plt.figure();
madePrev = plt.plot(timeBins[:-1], timeDiffCumHist_GivenMadeLastShot, label='made Prev'); plt.xlim((0,500))
missedPrev = plt.plot(timeBins[:-1], timeDiffCumHist_GivenMissedLastShot, label='missed Prev'); plt.xlim((0,500)); plt.ylim((0,1))
plt.title('cumulative density function - CDF'); plt.xlabel('time since last shot'); plt.legend(loc='lower right')

# ### Plot histogram of "Current Shot Distance - Previous Shot Distance" for the two groups
# Note that if Kobe throws from close by, and then from far away, this will result in positive values of "curr shot distance - prev shot distance"
# and vise versa. If Kobe throws from far away and then from close by, this will result in negative values.
#%% after making a shot, kobe is a more confident and throws from further away
plt.rcParams['figure.figsize'] = (12, 8)

jointHist, distDiffBins = np.histogram(changeInDistFromBasketDict['madeLast']+changeInDistFromBasketDict['missedLast'],bins=100,density=False)
barWidth = 0.999*(distDiffBins[1]-distDiffBins[0])

distDiffHist_GivenMadeLastShot, b = np.histogram(changeInDistFromBasketDict['madeLast'],bins=distDiffBins)
distDiffHist_GivenMissedLastShot, b = np.histogram(changeInDistFromBasketDict['missedLast'],bins=distDiffBins)
maxHeight = max(max(distDiffHist_GivenMadeLastShot),max(distDiffHist_GivenMissedLastShot)) + 30

plt.figure();
plt.subplot(2,1,1); plt.bar(distDiffBins[:-1], distDiffHist_GivenMadeLastShot, width=barWidth); plt.xlim((-40,40)); plt.ylim((0,maxHeight))
plt.title('made last shot'); plt.ylabel('counts')
plt.subplot(2,1,2); plt.bar(distDiffBins[:-1], distDiffHist_GivenMissedLastShot, width=barWidth); plt.xlim((-40,40)); plt.ylim((0,maxHeight))
plt.title('missed last shot'); plt.xlabel('curr shot distance - prev shot distance'); plt.ylabel('counts')


# We can clearly see that the made group of shots is more leaning to the right
# ### It therefore looks like Kobe is more confident after making a shot, and because of it, he takes a larger risk and throws from further away
# This is even more evident than the previous plot, but let's plot the cumulative histograms again to make it clearer
#%% to make the difference clearer, show the cumulative histogram
plt.rcParams['figure.figsize'] = (12, 8)

distDiffCumHist_GivenMadeLastShot = np.cumsum(distDiffHist_GivenMadeLastShot).astype(float)
distDiffCumHist_GivenMadeLastShot = distDiffCumHist_GivenMadeLastShot/max(distDiffCumHist_GivenMadeLastShot)
distDiffCumHist_GivenMissedLastShot = np.cumsum(distDiffHist_GivenMissedLastShot).astype(float)
distDiffCumHist_GivenMissedLastShot = distDiffCumHist_GivenMissedLastShot/max(distDiffCumHist_GivenMissedLastShot)

maxHeight = max(distDiffCumHist_GivenMadeLastShot[-1],distDiffCumHist_GivenMissedLastShot[-1])

plt.figure();
madePrev = plt.plot(distDiffBins[:-1], distDiffCumHist_GivenMadeLastShot, label='made Prev'); plt.xlim((-40,40))
missedPrev = plt.plot(distDiffBins[:-1], distDiffCumHist_GivenMissedLastShot, label='missed Prev'); plt.xlim((-40,40)); plt.ylim((0,1))
plt.title('cumulative density function - CDF'); plt.xlabel('curr shot distance - prev shot distance'); plt.legend(loc='lower right')
# ## Lastly, Let's plot the "Shot Difficulty" change for the two groups
# here negative values indicate that kobe took a larger risk, and positive values indicate that kobe made a safer subsequent shot
#%% after making a shot, kobe is a more confident and makes much more difficult shots generally
plt.rcParams['figure.figsize'] = (12, 8)

jointHist, difficultyDiffBins = np.histogram(changeInShotDifficultyDict['madeLast']+changeInShotDifficultyDict['missedLast'],bins=100)
barWidth = 0.999*(difficultyDiffBins[1]-difficultyDiffBins[0])

shotDifficultyDiffHist_GivenMadeLastShot, b = np.histogram(changeInShotDifficultyDict['madeLast'],bins=difficultyDiffBins)
shotDifficultyDiffHist_GivenMissedLastShot, b = np.histogram(changeInShotDifficultyDict['missedLast'],bins=difficultyDiffBins)
maxHeight = max(max(shotDifficultyDiffHist_GivenMadeLastShot),max(shotDifficultyDiffHist_GivenMissedLastShot)) + 30

plt.figure();
plt.subplot(2,1,1); plt.bar(difficultyDiffBins[:-1], shotDifficultyDiffHist_GivenMadeLastShot, width=barWidth); plt.xlim((-1,1)); plt.ylim((0,maxHeight))
plt.title('made last shot'); plt.ylabel('counts')
plt.subplot(2,1,2); plt.bar(difficultyDiffBins[:-1], shotDifficultyDiffHist_GivenMissedLastShot, width=barWidth); plt.xlim((-1,1)); plt.ylim((0,maxHeight))
plt.title('missed last shot'); plt.xlabel('chance to make curr shot - chance to make prev shot'); plt.ylabel('counts')

# ### We can see that the plot is heavier on the left side
# ### It is therefore even more evident now that kobe feels he's "In The Zone" after making a shot 
# and therefore he allows himself to attempt more difficult shots
# # But, is he right? 
# Maybe Kobe really is "in the zone" and therefore it's "OK" for him to take on more difficult shots?
#%% but wait, maybe kobe is making more difficult shots because he's "in the zone"

predictedShotPercentAfterMade = np.array(shotChancesListAfterMade).mean()
predictedStadardDev = np.sqrt(predictedShotPercentAfterMade*(1-predictedShotPercentAfterMade))
stadardError = predictedStadardDev/np.sqrt(len(shotChancesListAfterMade))
predPlusErr  = predictedShotPercentAfterMade + 2*stadardError
predMinusErr = predictedShotPercentAfterMade - 2*stadardError
actualShotPercentAfterMade = float(totalMadeAfterMade)/totalAttemptsAfterMade

print("-----------------------------------------------------")
print('provided that kobe made the previous shot:')
print('flat prediction 95% confidence interval is ['+ str(predMinusErr)+', '+str(predPlusErr)+']')
print('and kobe actually made ' + str(actualShotPercentAfterMade) + ', which is within confidence interval')
print("-----------------------------------------------------")

predictedShotPercentAfterMissed = np.array(shotChancesListAfterMissed).mean()
predictedStadardDev = np.sqrt(predictedShotPercentAfterMissed*(1-predictedShotPercentAfterMissed))
stadardError = predictedStadardDev/np.sqrt(len(shotChancesListAfterMissed))
predPlusErr  = predictedShotPercentAfterMissed + 2*stadardError
predMinusErr = predictedShotPercentAfterMissed - 2*stadardError
actualShotPercentAfterMissed = float(totalMadeAfterMissed)/totalAttemptsAfterMissed

print("-----------------------------------------------------")
print('provided that kobe missed the previous shot')
print('flat prediction 95% confidence interval is ['+ str(predMinusErr)+', '+str(predPlusErr)+']')
print('and kobe actually made ' + str(actualShotPercentAfterMissed) + ', which is within confidence interval')
print("-----------------------------------------------------")

# ### Well, maybe he is right, but it's Not Supported by the Data...