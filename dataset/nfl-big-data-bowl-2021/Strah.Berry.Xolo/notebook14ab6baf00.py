# %% [code]
import pandas as pd
import numpy as np
import os
#import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches
pd.set_option('max_columns', 100)


# %% [code]
plays = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')

# %% [code]
plays["playType"].value_counts()

# %% [code] {"scrolled":true}
## Get all the pass plays
passPlays = plays.query('playType == "play_type_pass"')
passPlays
passPlaysTrim = pd.DataFrame({'gameId' : passPlays["gameId"],'playId' : passPlays["playId"]})
passPlaysTrim

# %% [code]
games = pd.read_csv('../input/nfl-big-data-bowl-2021/games.csv', index_col=0)
games.head()

# %% [code]
week = pd.read_csv('../input/nfl-big-data-bowl-2021/week11.csv')

# %% [code]
## Reading all of the week data

week1 = pd.read_csv('../input/nfl-big-data-bowl-2021/week1.csv')
week2 = pd.read_csv('../input/nfl-big-data-bowl-2021/week2.csv')
week3 = pd.read_csv('../input/nfl-big-data-bowl-2021/week3.csv')
week4 = pd.read_csv('../input/nfl-big-data-bowl-2021/week4.csv')
week5 = pd.read_csv('../input/nfl-big-data-bowl-2021/week5.csv')
week6 = pd.read_csv('../input/nfl-big-data-bowl-2021/week6.csv')
week7 = pd.read_csv('../input/nfl-big-data-bowl-2021/week7.csv')
week8 = pd.read_csv('../input/nfl-big-data-bowl-2021/week8.csv')
week9 = pd.read_csv('../input/nfl-big-data-bowl-2021/week9.csv')
week10 = pd.read_csv('../input/nfl-big-data-bowl-2021/week10.csv')
week11 = pd.read_csv('../input/nfl-big-data-bowl-2021/week11.csv')
week12 = pd.read_csv('../input/nfl-big-data-bowl-2021/week12.csv')
week13 = pd.read_csv('../input/nfl-big-data-bowl-2021/week13.csv')
week14 = pd.read_csv('../input/nfl-big-data-bowl-2021/week14.csv')
week15 = pd.read_csv('../input/nfl-big-data-bowl-2021/week15.csv')
week16 = pd.read_csv('../input/nfl-big-data-bowl-2021/week16.csv')
week17 = pd.read_csv('../input/nfl-big-data-bowl-2021/week17.csv')



# %% [code]
## Read in all of the player names

players = pd.read_csv('../input/nfl-big-data-bowl-2021/players.csv')

## Replace values to be in inches and not feet
playersAdj = players.replace("5-6", "66")
playersAdj = playersAdj.replace("5-7", "67")
playersAdj = playersAdj.replace("5-8", "68")
playersAdj = playersAdj.replace("5-9", "69")
playersAdj = playersAdj.replace("5-10", "70")
playersAdj = playersAdj.replace("5-11", "71")
playersAdj = playersAdj.replace("6-0", "72")
playersAdj = playersAdj.replace("6-1", "73")
playersAdj = playersAdj.replace("6-2", "74")
playersAdj = playersAdj.replace("6-3", "75")
playersAdj = playersAdj.replace("6-4", "76")
playersAdj = playersAdj.replace("6-5", "77")
playersAdj = playersAdj.replace("6-6", "78")
playersAdj = playersAdj.replace("6-7", "79")
playersAdj.height.value_counts()

# %% [code]
## Moment in play when pass is forwarded
week_PF = week.query('event == "pass_forward"')
## Moment in play when pass arrives
week_PA = week.query('event == "pass_arrived"')

# %% [code]
pass_arrived = week.query('gameId == 2018111500 and playId == 344 and frameId == 30')
pass_arrived

# %% [markdown]
# 

# %% [code]
fball_pa = pass_arrived.query('displayName == "Football"')
fball_pa_x = fball_pa['x'].values
fball_pa_y = fball_pa['y'].values
fball_pa_x
#fball_x = pd.DataFrame({'x' : fball_x})
#fball_y = pd.DataFrame({'y' : fball_y})
#fball_y

# %% [code]
n_fball_pa = pass_arrived.query('displayName != "Football"')
n_fball_pa_x = n_fball_pa['x']
n_fball_pa_y = n_fball_pa['y']
n_fball_nflId_pa = n_fball_pa['nflId']
player_locx_pa = pd.DataFrame({'nflId' : n_fball_nflId_pa,'x' : n_fball_pa_x})
player_locy_pa = pd.DataFrame({'nflId' : n_fball_nflId_pa, 'y' : n_fball_pa_y})


# %% [code]
## Find the distance from the ball
distFromBall_pa = ((player_locx_pa['x'] - fball_pa_x) ** 2) + ((player_locy_pa['y'] - fball_pa_y) ** 2) ** (1/2)
distFromBall_pa = pd.DataFrame({'nflId' : n_fball_nflId_pa,'distFromBall' : distFromBall_pa})

# %% [code]
tackle = week.query('gameId == 2018111500 and playId == 344 and event == "tackle"')
tackle

# %% [code]
fball_tkl = tackle.query('displayName == "Football"')
fball_tkl_x = fball_tkl['x'].values
fball_tkl_y = fball_tkl['y'].values
n_fball_tkl = tackle.query('displayName != "Football"')
n_fball_tkl_x = n_fball_tkl['x']
n_fball_tkl_y = n_fball_tkl['y']
n_fball_nflId_tkl = n_fball_tkl['nflId']
player_locx_tkl = pd.DataFrame({'nflId' : n_fball_nflId_tkl,'x' : n_fball_tkl_x})
player_locy_tkl = pd.DataFrame({'nflId' : n_fball_nflId_tkl, 'y' : n_fball_tkl_y})

## Find the distance from the ball
distFromBall_tkl = ((player_locx_tkl['x'] - fball_tkl_x) ** 2) + ((player_locy_tkl['y'] - fball_tkl_y) ** 2) ** (1/2)
distFromBall_tkl = pd.DataFrame({'nflId' : n_fball_nflId_tkl,'distFromBall' : distFromBall_tkl})


# %% [code]
def distanceFromBall (playFrame) : 
    
    fball = playFrame.query('displayName == "Football"')
    
    ## TODO later
    #numOfFB = len(fball.index)
    #totalNumOfRows = len(playFrame) 
    ## Some plays have multiple tackle play frames so have to iterate multiple times
    #numOfElementsPerSect = (totalNumOfRows / numOfFB) - 1
    
    fball_x = fball['x'].values
    fball_y = fball['y'].values
    n_fball = playFrame.query('displayName != "Football"')
    n_fball_x = n_fball['x']
    n_fball_y = n_fball['y']
    n_fball_s = n_fball['s']
    n_fball_a = n_fball['a']
    n_fball_nflId = n_fball['nflId']
    n_fball_gameId = n_fball['gameId']
    n_fball_playId = n_fball['playId']
    n_fball_position = n_fball['position']
    player_locx = pd.DataFrame({'nflId' : n_fball_nflId,'x' : n_fball_x})
    player_locy = pd.DataFrame({'nflId' : n_fball_nflId, 'y' : n_fball_y})

    ## Find the distance from the ball
    distFromBall = (((player_locx['x'] - fball_x) ** 2) + ((player_locy['y'] - fball_y) ** 2)) ** (1/2)
    distFromBall = pd.DataFrame({'gameId' : n_fball_gameId,
                                 'playId' : n_fball_playId,
                                 'nflId' : n_fball_nflId,
                                 'position' : n_fball_position,
                                 'x' : n_fball_x,
                                 'y' : n_fball_y,
                                 's' : n_fball_s,
                                 'a' : n_fball_a,
                                 'distFromBall' : distFromBall})
    
    return distFromBall

# %% [code] {"scrolled":true}
def gatherTackleData (weekTrack) : 
    
    ## Using the pass play data to get tracking data using the play and game id

    passGamesInWeek = pd.DataFrame() 

    #column_names = ["gameId", "playId", "nflId", "pfwd_x", "pfwd_y", "tckl_x", "tckl_y", "totalPlayerDisp"]

    #totalDisp = [pd.DataFrame(columns = column_names)]
    tackleDist = []

    # Does certain player make the tackle?
    doesPlayerTackle = []


    ## Iterate through each pass play to get the gameId, playId
    for playIndex, playRow in passPlaysTrim.iterrows():

        ## Gather the games in week which have pass plays
        if not weekTrack.loc[(weekTrack['gameId'] == playRow['gameId']) & (weekTrack['playId'] == playRow['playId'])].empty:
            #print(len(week.loc[(week['gameId'] == playRow['gameId']) & (week['playId'] == playRow['playId'])]))
            passGamesInWeek = weekTrack.loc[(weekTrack['gameId'] == playRow['gameId']) & (weekTrack['playId'] == playRow['playId'])]

            ## Get plays which only result in tackles finishing the play
            tacklePassPlays = passGamesInWeek.query('event == "tackle"')

            ## Don't use plays with multiple tackle frames, for now :)
            if (len(tacklePassPlays.query('displayName == "Football"')) > 1) :
                continue
            else :

                ## Filter out offensive positions so only defensive positions remain

                allPlayerDistFromBallTckl = tacklePassPlays
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "WR")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "TE")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "QB")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "RB")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "HB")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "FB")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "P")]
                allPlayerDistFromBallTckl = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['position'] != "LS")]
                allPlayerDistFromBallTckl = distanceFromBall(allPlayerDistFromBallTckl)

                playersClosestToBall = allPlayerDistFromBallTckl.query('distFromBall < 1.0')

                ## Get the distances from football to players on the play frame when the event == pass_forward
                passFwd = distanceFromBall(passGamesInWeek.query('event == "pass_forward" or event == "pass_shovel"'))

                ## Make sure only pass_forward plays that end in tackles are gathered
                #if ((passFwd['gameId'] == tacklePassPlays['gameId']) 
                #    & (passFwd['playId'] == tacklePassPlays['playId'])):

                passFwd = passFwd.loc[(passFwd['gameId'] == playRow['gameId']) 
                                            & (passFwd['playId'] == playRow['playId'])]
                
                if not passFwd.empty:
                    ## Only do stuff when the df isn't empty               
                    
                    if not allPlayerDistFromBallTckl.empty:
                        #print(allPlayerDistFromBallTckl)

                        ## Iterate through 
                        for trackIndex, trackRow in allPlayerDistFromBallTckl.iterrows():
                            #print(trackRow['nflId'])
                            
                            #print(trackRow['gameId'],trackRow['playId'])
                            
                            ## Get pass_forward data using the nflId from players which tackled
                            distAtPassFwd = passFwd.loc[(passFwd['gameId']   == playRow['gameId']) 
                                                 & (passFwd['playId'] == playRow['playId']) 
                                                 & (passFwd['nflId']  == trackRow['nflId'])]

                            #print(distAtPassFwd, playRow['gameId'], playRow['playId'], trackRow['nflId'])

                            ## Get x and y values for play frame when tackle happens
                            tckl_x = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['nflId'] == trackRow['nflId'])]['x'].values
                            tckl_y = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['nflId'] == trackRow['nflId'])]['y'].values
                            tckl_dst = allPlayerDistFromBallTckl.loc[(allPlayerDistFromBallTckl['nflId'] == trackRow['nflId'])]['distFromBall'].values

                            
                            ## Get x and y values for play frame when pass_forward happens
                            pfwd_x = distAtPassFwd['x'].values
                            pfwd_y = distAtPassFwd['y'].values
                            pfwd_s = distAtPassFwd['s'].values
                            pfwd_a = distAtPassFwd['a'].values
                            
                            ## Get player physical data
                            playerPhysData = playersAdj.loc[(playersAdj['nflId'] == trackRow['nflId'])]

                            ## Get the total displacement for a player between the pass_forward and tackle frame
                            totalPlayerDisp = (((tckl_x - pfwd_x) ** 2) + ((tckl_y - pfwd_y) ** 2)) ** (1/2)

                            ## Create dataframe
                            totalDisp = pd.DataFrame({'gameId' : playRow['gameId'],
                                                      'playId' : playRow['playId'],
                                                      'nflId' : trackRow['nflId'],
                                                      'pfwd_x' : pfwd_x,
                                                      'pfwd_y' : pfwd_y,
                                                      'tckl_x' : tckl_x,
                                                      'tckl_y' : tckl_y,
                                                      'speedPF' : pfwd_s,
                                                      'accelPF' : pfwd_a,
                                                      'height' : playerPhysData['height'],
                                                      'weight' : playerPhysData['weight'],
                                                      'distFrmBllWhenTackle' : tckl_dst,
                                                      'totalPlayerDisp' : totalPlayerDisp})

                            ## Check to see if player is within bounds to determine if they made the tackle
                            currPlayerTackle = playersClosestToBall.loc[(playersClosestToBall['gameId'] == playRow['gameId'])
                                                         & (playersClosestToBall['playId'] == playRow['playId'])
                                                         & (playersClosestToBall['nflId'] == trackRow['nflId'])]

                            if not currPlayerTackle.empty :
                                doesPlayerTackle.append("1")
                            else :
                                doesPlayerTackle.append("0")

                            tackleDist.append(totalDisp)

    finalTackleDist = pd.concat(tackleDist, ignore_index=True)
    
    return finalTackleDist, doesPlayerTackle



# %% [code]
## [jangeles]: Print versions of TensorFlow and Keras

import tensorflow as tf
from tensorflow import keras
print("Tensorflow Version:", tf.__version__)
print("Keras Version:", keras.__version__)

import time 
import datetime

# %% [code]
## Concatinating Weeks to make a bigger train and validation sets
trainWeekNames = [week1, week3, week5]
valWeekNames =  [week2, week4]

concatWeeksTrain = []
concatWeeksVal = []

for f in trainWeekNames:
    concatWeeksTrain.append(f)

for f in valWeekNames:
    concatWeeksVal.append(f)
    
concatWeeksTrain = pd.concat(concatWeeksTrain, ignore_index=True)
concatWeeksVal = pd.concat(concatWeeksVal, ignore_index=True)

print(len(concatWeeksTrain),len(concatWeeksVal))

# %% [code]
# ts stores the time in seconds 
startTime = datetime.datetime.now() 

# print the current timestamp 
print(startTime) 

#check1, check2 = gatherTackleData(week)
#trainData, trainTackle = gatherTackleData(week2)
trainData, trainTackle = gatherTackleData(concatWeeksTrain)

endTime = datetime.datetime.now() 
print(endTime) 


# %% [code]
print(datetime.datetime.now()) 

#valData, valTackle = gatherTackleData(week3)
valData, valTackle = gatherTackleData(concatWeeksVal)

print(datetime.datetime.now()) 

# %% [code]
## [jangeles]: 

## READ "Implementing MLPs with Keras" on Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow 
#(pages 295-308)
from tensorflow.keras.optimizers import SGD

model = keras. models.Sequential()
#model.add(keras.layers.Dense(300, activation ="sigmoid"))
#model.add(keras.layers.Dense(300, activation ="sigmoid"))
model.add(keras.layers.Dense(300, activation ="relu"))
#model.add(keras.layers.Dense(10, activation = "softmax"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))


sgd = SGD(learning_rate=0.005)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

#X_train = X_train.astype('float32')/255.0
#y_train = y_train.astype('uint')
#X_val = X_val.astype('float32')/255.0
#y_val = y_val.astype('uint')

trainDataNP = pd.DataFrame({'nflId' : trainData.nflId,
                            'speedPF' : trainData.speedPF,
                            'accelPF' : trainData.accelPF,
                            'height' : trainData.height,
                            'weight' : trainData.weight,
                            'distFrmBllWhenTackle' : trainData.distFrmBllWhenTackle,
                            'totalPlayerDisp' : trainData.totalPlayerDisp})
trainDataNP = trainDataNP.values
trainDataNP = trainDataNP.astype('float32')
trainTackleNP = np.array(trainTackle)
trainTackleNP = trainTackleNP.astype('float32')

valDataNP = pd.DataFrame({'nflId' : valData.nflId,
                          'speedPF' : valData.speedPF,
                          'accelPF' : valData.accelPF,
                          'height' : valData.height,
                          'weight' : valData.weight,
                          'distFrmBllWhenTackle' : valData.distFrmBllWhenTackle,
                          'totalPlayerDisp' : valData.totalPlayerDisp})
valDataNP = valDataNP.values
valDataNP = valDataNP.astype('float32')
valTackleNP = np.array(valTackle)
valTackleNP = valTackleNP.astype('float32')


# %% [code]
#model.fit(X_train, y_train, batch_size=50, epochs=10, validation_data =(X_val,y_val))
history = model.fit(trainDataNP, trainTackleNP, batch_size=50, epochs=50, validation_data = (valDataNP, valTackleNP))

# %% [code]
print("Learning Curve")
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# %% [code]
#testWeekNames = [week11]
#concatWeeksTest = []
#for f in testWeekNames:
#    concatWeeksTest.append(f)
#concatWeeksTest = pd.concat(concatWeeksTest, ignore_index=True)

## Creating a test set
print(datetime.datetime.now()) 

testData, testTackle = gatherTackleData(week11)

print(datetime.datetime.now()) 

testDataNP = pd.DataFrame({'nflId' : testData.nflId,
                           'speedPF' : testData.speedPF,
                           'accelPF' : testData.accelPF,
                           'height' : testData.height,
                           'weight' : testData.weight,
                           'distFrmBllWhenTackle' : testData.distFrmBllWhenTackle,
                           'totalPlayerDisp' : testData.totalPlayerDisp})
testDataNP = testDataNP.values
testDataNP = testDataNP.astype('float32')
testTackleNP = np.array(testTackle)
testTackleNP = testTackleNP.astype('float32')

# %% [code]
## [jangeles]: Print the accuracy of the algorithm

loss, acc = model.evaluate(testDataNP, testTackleNP, verbose=0)
print('Loss: %.3f' % loss)
print('Accuracy: %.3f' % acc)

# %% [code] {"scrolled":true}
#predWeekNames = [week11]
#concatWeeksPred = []
#for f in predWeekNames:
#    concatWeeksPred.append(f)
#concatWeeksPred = pd.concat(concatWeeksPred, ignore_index=True)

## Create a prediction set
## currently the same as the test set

print(datetime.datetime.now()) 

predData, predTackle = gatherTackleData(week11)

print(datetime.datetime.now()) 

predDataNP = pd.DataFrame({'nflId' : predData.nflId,
                           'speedPF' : predData.speedPF,
                           'accelPF' : predData.accelPF,
                           'height' : predData.height,
                           'weight' : predData.weight,
                           'distFrmBllWhenTackle' : predData.distFrmBllWhenTackle, 
                           'totalPlayerDisp' : predData.totalPlayerDisp})
predDataNP = predDataNP.values
predDataNP = predDataNP.astype('float32')
predTackleNP = np.array(predTackle)
predTackleNP = predTackleNP.astype('float32')

#predDataNP = testDataNP
#predTackleNP = testTackleNP


# %% [code]
from sklearn.metrics import classification_report, confusion_matrix

## Predict
y_pred = model.predict_classes(predDataNP)

predictedTackle = np.array(y_pred)
predictedTackle = np.concatenate(predictedTackle)

# %% [code]
## Create a new dataframe which shows the prediction value next to the expected value to verify against.

predictResults = pd.DataFrame({'nflId' : predData['nflId'], 
                               'distFrmBllWhenTackle' : predData['distFrmBllWhenTackle'],
                               'totalPlayerDisp' : predData['totalPlayerDisp'],
                               'predictedTackle' : predictedTackle,
                               'actualTackle' : predTackle})

predictResults.loc[(predictResults.predictedTackle == 0)]

# %% [code]
