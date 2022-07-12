import numpy as np
import pandas as pd
import time

#%% load data
print('loading data...')
INPUT_PATH = '../input/'     
                   
childPref = pd.read_csv(INPUT_PATH + 'child_wishlist.csv', header=None).as_matrix()[:, 1:]
santaPref = pd.read_csv(INPUT_PATH + 'gift_goodkids.csv' , header=None).as_matrix()[:, 1:]

numChildren = childPref.shape[0]
numGifts    = santaPref.shape[0]
numGiftsPerChild = numChildren / numGifts
numTwins = 4000

#%% create lookup matrix
print('creating child vs. gift value matrix...')
childPreferenceMatrix = -1*np.ones((numChildren,numGifts),np.float32)
for childID in range(numChildren):
    for giftOrder, giftID in enumerate(childPref[childID,:]):
        childPreferenceMatrix[childID,giftID] = 2*(10 - giftOrder) 

santaPreferenceMatrix = -1*np.ones((numChildren,numGifts),np.float32)
for giftID in range(numGifts):
    for childOrder, childID in enumerate(santaPref[giftID,:]):
        santaPreferenceMatrix[childID,giftID] = 2*(1000 - childOrder) 

child_vs_gift_matrix =  childPreferenceMatrix/(20.0*numChildren)
child_vs_gift_matrix += santaPreferenceMatrix/(2000000.0*numGifts)
child_vs_gift_matrix = child_vs_gift_matrix.astype(np.float32)

del childPreferenceMatrix
del santaPreferenceMatrix

# the preferences of the twins should be combined
child_vs_gift_matrix[0:numTwins:2] = child_vs_gift_matrix[0:numTwins:2] + child_vs_gift_matrix[1:numTwins:2]
child_vs_gift_matrix[1:numTwins:2] = child_vs_gift_matrix[0:numTwins:2]

#%% define some helper functions below:

# scoring function
def calculateTotalHapiness(pred):
    # twins
    totalHapiness = 0
    for i in range(0,numTwins,2):
        child_id = i
        gift_id = pred[i]
        totalHapiness += child_vs_gift_matrix[child_id,gift_id]
    
    # rest of the children
    for i in range(numTwins, numChildren):
        child_id = i
        gift_id = pred[i]
        totalHapiness += child_vs_gift_matrix[child_id,gift_id]
        
    return totalHapiness


# this function takes a numChildren X numGifts matrix and converts it greedily to a prediction vector
def AssignGifts_GreedyChildren_Adaptive(child_vs_gift_selection_matrix=child_vs_gift_matrix, numPasses=15):
    child_vs_gift_selection_matrix = child_vs_gift_selection_matrix.copy()
    
    giftAssignment = -np.ones((numChildren), dtype=np.int32)
    giftCount      = np.zeros((numGifts),    dtype=np.int32)
    
    print('-'*40)
    print('assigning gifts to twins')
    startTime  = time.time()
    
    # sort the twins
    sortedTwins = 2*(child_vs_gift_selection_matrix[0:numTwins:2].max(axis=1).argsort())
    
    for childInd in sortedTwins:
        selectedGift = child_vs_gift_selection_matrix[childInd,:].argmax()
        if giftCount[selectedGift] < numGiftsPerChild and giftAssignment[childInd] == -1:
            giftAssignment[childInd] = selectedGift
            giftAssignment[childInd+1] = selectedGift
            giftCount[selectedGift] += 2
            
    print('starting adaptive pass over the rest of the children')
    childrenPerPass = int(1+(numChildren-numTwins) / (numPasses+1.0))
    
    # at each pass we lower a threshold and assign gift to childern that are above the threshold
    for k in range(numPasses+1):
        # sort the children accroding to the maximum possible matrix value of each child
        maxValuePerChild = child_vs_gift_selection_matrix.max(axis=1)
        sortedChildren   = (numTwins + maxValuePerChild[numTwins:].argsort())[::-1]
        
        thresholdChildInd   = min(numChildren-numTwins-1,int(numTwins + (k+1)*childrenPerPass))        
        assignmentThreshold = maxValuePerChild[thresholdChildInd]

        numAssignedSoFar = (giftAssignment > -1).sum()        
        if (numAssignedSoFar > (0.99*numChildren)) or (k >= (numPasses)):
            # make this last iteration
            assignmentThreshold = child_vs_gift_selection_matrix.min() - 1.0
            thresholdChildInd = len(sortedChildren)
            
        if numAssignedSoFar >= numChildren:
            break
        
        for childInd in sortedChildren[:thresholdChildInd]:
            # don't assign gifts unless high on priority list ( larger than 'assignmentThreshold' )
            if giftAssignment[childInd] == -1 and child_vs_gift_selection_matrix[childInd,:].max() >= assignmentThreshold:
                selectedGift = child_vs_gift_selection_matrix[childInd,:].argmax()
        
                giftAssignment[childInd] = selectedGift
                giftCount[selectedGift] += 1
                
                if giftCount[selectedGift] >= numGiftsPerChild:
                    child_vs_gift_selection_matrix[:,selectedGift] = -1.0
                    
        print('pass %d: total assigned so far = %d' %(k+1,(giftAssignment > -1).sum()))

    print('finished %d adaptive passes. took %.3f seconds' %(k+1, time.time()-startTime))
    assignmentScore = calculateTotalHapiness(giftAssignment)
    print('-'*40)

    return giftAssignment, assignmentScore
    

#%% normalize rows and columns of the assignment value matrix
print('normalizing rows and columns of the value matrix')
startingMatrix = child_vs_gift_matrix.copy()
startingMatrix -= startingMatrix.min()
startingMatrix /= startingMatrix.max()

# divide by gift variability (boosts unpopular gifts)
print('heuristic #1: boost unpopular gifts before greedy assignment')
giftVariability  = startingMatrix.std(axis=0)
for giftInd, giftVar in enumerate(giftVariability):
    startingMatrix[:,giftInd] /= giftVar

# divide by child favoribility
print('heuristic #2: penalize good kids before greedy assignment')
childFavorability = startingMatrix.mean(axis=1)
for childInd, childFavor in enumerate(childFavorability):
    startingMatrix[childInd,:] /= childFavor

# create an assignment vector from the child vs. gift value matrix
print('assign gifts to children in a child centered greedy fashion')
pred, score = AssignGifts_GreedyChildren_Adaptive(startingMatrix, numPasses=16)
print('predicted score = %.8f' %(score))

#%% create a submission
out = open('heuristicSub.csv', 'w')
out.write('ChildId,GiftId\n')
for i in range(len(pred)):
    out.write(str(i) + ',' + str(pred[i]) + '\n')
out.close()