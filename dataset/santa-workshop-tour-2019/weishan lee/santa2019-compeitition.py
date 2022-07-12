# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randrange, random
import sys
import time
import math
# Set your own project id here
PROJECT_ID = 'santa2019'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

nFamilies = 5000
nDays = 100
fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'
dfFamily = pd.read_csv(fpath, index_col='family_id')

#%% calculate penalty matrix Pnt
Pnt = np.empty([nDays,nFamilies])
for j in range(nDays):
    for i in range(nFamilies):
        if (j+1==dfFamily['choice_0'][i]):
            Pnt[j,i]=0
        elif (j+1==dfFamily['choice_1'][i]):
            Pnt[j,i]=50
        elif (j+1==dfFamily['choice_2'][i]):
            Pnt[j,i]=50 + 9*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_3'][i]):
            Pnt[j,i]=100 + 9*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_4'][i]):
            Pnt[j,i]=200 + 9*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_5'][i]):
            Pnt[j,i]=200 + 18*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_6'][i]):
            Pnt[j,i]=300 + 18*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_7'][i]):
            Pnt[j,i]=300 + 36*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_8'][i]):
            Pnt[j,i]=400 + 36*dfFamily['n_people'][i]
        elif (j+1==dfFamily['choice_9'][i]):
            Pnt[j,i]=500 + (36+199)*dfFamily['n_people'][i]
        else:
            Pnt[j,i]=500 + (36+398)*dfFamily['n_people'][i]

#%% functions definitions

def genFD():
    """
    Generate a set of binary numbers FD_ij for family i to visit the shop
    on day j. i in [0,5000), j in [0,100)
    """
    jC = [] # collection of j values
    FD = np.zeros([nFamilies,nDays],int)
    for i in range(nFamilies):
        j = randrange(0, nDays) # generate day number between [0,nDays)
        jC += [j]
        FD[i,j] = 1
    return (FD,jC)

def calNj(FD):
    """
    Calculate the total number of people on day j
    """
    Nj = np.empty(nDays+1)
    for j in range(nDays):
        NjTemp= 0
        for i in range(nFamilies):
            NjTemp += FD[i,j] * dfFamily['n_people'][i]
        Nj[j] = NjTemp
    Nj[nDays] = Nj[nDays-1]
    return (Nj)

def checkNjBound(Nj):
    """
    Check if Nj is between [125,300]
    """
    for j in range(nDays):
        if (Nj[j]>300) or (Nj[j]<125):
            return (False)
    return (True)

def onePCost(i,j,FD,Pnt):
    return (FD[i,j] * Pnt[j,i])

def oneAPnt(j,Nj):
    #ans = (Nj[j]-125.0)/400.0 * Nj[j]**(0.5+math.fabs(Nj[j]-Nj[j+1])/50.0)
    ans = (Nj[j]-125.0)/400.0 * math.pow( Nj[j], 0.5 + math.fabs(Nj[j]-Nj[j+1])/50.0 )
    return ans

def calPCost(FD,Pnt):
    '''
    Calculate pCost = preference cost
    '''
    pCost = 0
    for i in range(nFamilies):
        for j in range(nDays):
            pCost += onePCost(i,j,FD,Pnt)
    return (pCost)
    """
    pCostList = []
    for i in range(nFamilies):
        for j in range(nDays):
            pCostList += [onePCost(i,j,FD,Pnt)]
    ans = math.fsum(pCostList)
    return ans
    """
def calAPnt(Nj):
    '''
    # this function is used because deltaAPntFunc causes accuracy issue.
    Calculate aPnt = account penalty
    '''
    """
    aPnt = 0.0
    for j in range(nDays):
        aPnt += oneAPnt(j,Nj)
    return (aPnt)
    """
    oneAPntList = []
    for j in range(nDays):
        oneAPntList += [oneAPnt(j,Nj)]
    ans = math.fsum(oneAPntList)
    return ans

"""
def deltaAPnt(j,k,Nj): 
    # This function causes accuracy issues.
    # Note: j and k are different!
    if j == 0:
        ans = oneAPnt(j,Nj) + oneAPnt(k-1,Nj) + oneAPnt(k,Nj)
    elif k == 0:
        ans = oneAPnt(k,Nj) + oneAPnt(j-1,Nj) + oneAPnt(j,Nj)
    else:
        ans = oneAPnt(j-1,Nj) + oneAPnt(j,Nj) + oneAPnt(k-1,Nj) + oneAPnt(k,Nj)
    return ans
"""   
   
def calScore(FD,Pnt,Nj):
    # this function is used only to double check if two methods calculating 
    # the score have the same results.
    '''
    Calculate the score = preference cost + account penalty
    '''
    pCost = calPCost(FD,Pnt)
    #aPnt = calAPnt(Nj)
    ans = math.fsum([pCost,calAPnt(Nj)])
    return ans

data = {'family_id': [i for i in range(nFamilies)]}

def outPutCSV(jC,score):
    #outPutCSV(FDUpdate,score):
    dfCSV = pd.DataFrame(data)
    assigned_day = []
    for i in range(nFamilies): assigned_day += [jC[i]+1]
    dfCSV['assigned_day'] = assigned_day 
    dfCSV_file = open("./submission_{}.csv".format(int(score)),'w+',newline='') 
    dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
    dfCSV_file.close()
    
softWareVersion = "V_3_errorCheck "    
errorLog = open("errogLog.txt","a+") 
#def checkError(i,j,k,cdt,FD,Nj,pCost,aPnt,score,pst):
def checkError(i,j,k,cdt,FD,Nj,pCost,score,pst):
    if cdt == "swapBack":
        if FD[i,j]!=1 or FD[i,k]!=0:
            msg = " FD Error! FD[{},{}] = {}, FD[{},{}] = {}, Line {}\n".format(i,j,FD[i,j],i,k,FD[i,k],pst)
            strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
            errorLog.write(strW)
            sys.exit(msg)
    if cdt == "swapForward":
        if FD[i,j]!=0 or FD[i,k]!=1:
            msg = " FD Error! FD[{},{}] = {}, FD[{},{}] = {}, Line {}\n".format(i,j,FD[i,j],i,k,FD[i,k],pst)
            strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
            errorLog.write(strW)
            sys.exit(msg)
    if ( checkNjBound(Nj) == False ):
        msg = " Nj Bound Error! Nj[{}] = {}, Nj[{}] = {}, Line {}\n".format(j,Nj[j],k,Nj[k],pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)
    if ( ( pCost - calPCost(FD,Pnt) ) > 1e-4 ): 
        msg = " pCost Error! pCost = {}, calPCost = {}, Line {}\n".format(pCost,calPCost(FD,Pnt),pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)
    """
    if ( ( aPnt - calAPnt(Nj) ) > 1e-4 ) :
        msg = " aPnt Error! aPnt = {}, calAPnt = {}, Line = {}\n".format(aPnt,calAPnt(Nj),pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)
    """
    if ( ( score - calScore(FD,Pnt,Nj) ) > 1e-4 ):
        msg = " score Error! score = {}, calScore = {}, Line = {}\n".format(score,calScore(FD,Pnt,Nj),pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)
    if (pCost<0.0):
        msg = " pCost negative! pCost = {}, Line = {}\n".format(pCost,pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)
    """
    if (aPnt<0.0):
        msg = " aPnt negative! aPnt = {}, Line = {}\n".format(aPnt,pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)
    """
    if score<0.0:
        aPnt = calAPnt(Nj)
        msg = " score negative! score = {}, pCost = {}, aPnt = {}, Line = {}\n".format(score,pCost,aPnt,pst)
        strW = softWareVersion + time.asctime( time.localtime(time.time()) ) + msg
        errorLog.write(strW)
        sys.exit(msg)

#%% calculate the score by using simulated annealing
#Calculate the initial score

#Use genFD()
FD,jC = genFD()
Nj = calNj(FD)
while ( checkNjBound(Nj) == False ):
    FD,jC = genFD()
    Nj = calNj(FD)

pCost = calPCost(FD,Pnt)
#score = pCost + calAPnt(Nj)
score  = math.fsum([pCost,calAPnt(Nj)])
print("Initial score = {:.4f}\n".format(score))
outPutCSV(FD,score)

# Main loop

Tmax = 100.0   #score
Tmin = 1e-3 #Score target # 40000.0
tau = 1e6
targetScore = 250000.0
minScore = score

"""
tRecord = []
scoreRecord = []
#deltaScoreRecord = []

t0=0 # setting up the beginning of the time "lump"
tRecord += [0]
scoreRecord += [score]
#deltaScoreRecord += [0]
"""

while (score>targetScore):
    T = Tmax
    t = 0 # time step 

    while T>Tmin:
        # Cooling
        t += 1
        T = Tmax*np.exp(-t/tau)
        
        oldPCost = pCost 
        oldScore = score
        # Could be either good or bad state after the second loop.
    
        # Choose another new state to swap.
        i = randrange(0, nFamilies) # Choose a specific family to swap
        j = jC[i]                   # The day for family i to visit the shop
        k = randrange(0, nDays)     # Choose a day to swap with the original j
        while (k == j): k = randrange(0, nDays) # Make sure the new state and the 
                                            # old state are distinct                     
        pCost = pCost - onePCost(i,j,FD,Pnt)
        FD[i,j] , FD[i,k] = FD[i,k] , FD[i,j]
        Nj[j] = Nj[j] - dfFamily['n_people'][i]
        Nj[k] = Nj[k] + dfFamily['n_people'][i]
        if ( k == (nDays-1) ):  Nj[k+1] = Nj[k]
        jC[i] = k
        pCost = pCost + onePCost(i,k,FD,Pnt)
        #score = pCost + calAPnt(Nj)
        score  = math.fsum([pCost,calAPnt(Nj)])
    
        # Make sure Nj of the new state is between [125,300]
    
        while( checkNjBound(Nj) == False ):
            #Swap back to the previous state (could be either good or bad state)       
            pCost = pCost - onePCost(i,k,FD,Pnt)
            FD[i,j] , FD[i,k] = FD[i,k] , FD[i,j]   # Swap back FD
            Nj[j] = Nj[j] + dfFamily['n_people'][i] # Swap back Nj[j]
            Nj[k] = Nj[k] - dfFamily['n_people'][i] # Swap back Nj[k]
            if (k == nDays-1):  Nj[k+1] = Nj[k]
            jC[i] = j   # Swap back jC[i]
            # End swapping back
            pCost = oldPCost
            score = oldScore
        
            # Choose another new state to swap.
            oldPCost = pCost
            oldScore = score
            i = randrange(0, nFamilies) # Choose a specific family to swap
            j = jC[i]                   # The day for family i to visit the shop
            k = randrange(0, nDays)     # Choose a day to swap with the original j
            while (k == j): k = randrange(0, nDays) # Make sure the new state and the 
                                                # old state are distinct
            pCost = pCost - onePCost(i,j,FD,Pnt)
            FD[i,j] , FD[i,k] = FD[i,k] , FD[i,j]
            Nj[j] = Nj[j] - dfFamily['n_people'][i]
            Nj[k] = Nj[k] + dfFamily['n_people'][i]
            if ( k == (nDays-1) ):  Nj[k+1] = Nj[k]
            jC[i] = k
            pCost = pCost + onePCost(i,k,FD,Pnt)
            #score = pCost + calAPnt(Nj)
            score  = math.fsum([pCost,calAPnt(Nj)])
        
        # After succefully choosing a good state to swap, calculate the score of the
        # new good state.    
        #score = pCost + calAPnt(Nj) 
        #checkError(i,j,k,"swapForward",FD,Nj,pCost,score,"320")
        # Calculate the change in score 
        deltaScore = score - oldScore
        
        if score < minScore: 
            minScore = score
            outPutCSV(jC,score)
        #if score < oldScore: outPutCSV(jC,score)
        print("Delta score = {:.4f}".format(deltaScore))
        print("New score = {:.4f}\n".format(score))
    
        try:
            ans = math.exp(-deltaScore/T)
        except OverflowError:
            if -deltaScore/T > 0:
                ans = float('inf')
            else:
                ans = 0.0
    
        # If the move is rejected, swap them back again
        if random() > ans:
            # Start swapping back to the old state (could be either good or bad state)
            pCost = pCost - onePCost(i,k,FD,Pnt)
            FD[i,j] , FD[i,k] = FD[i,k] , FD[i,j]   # Swap back FD
            Nj[j] = Nj[j] + dfFamily['n_people'][i] # Swap back Nj[j]
            Nj[k] = Nj[k] - dfFamily['n_people'][i] # Swap back Nj[k]
            if (k == nDays-1):  Nj[k+1] = Nj[k]
            jC[i] = j   # Swap back jC[i]
            # End swapping back
            pCost = oldPCost
            score = oldScore
            #pCost = pCost + onePCost(i,j,FD,Pnt)
            #score = pCost + calAPnt(Nj)
            #score  = math.fsum([pCost,calAPnt(Nj)])

        """
        tRecord += [t0+t]
        scoreRecord += [score]
        #deltaScoreRecord += [deltaScore] 
        """
    #t0 = t0 + t # go to next time "lump"
"""
data = {'timeStep': tRecord,'score':scoreRecord} # ,'deltaScore': deltaScoreRecord}
dfCSV = pd.DataFrame(data)
dfCSV_file = open("./enVsTime.csv".format(int(score)),'w',newline='') 
dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
dfCSV_file.close()
"""