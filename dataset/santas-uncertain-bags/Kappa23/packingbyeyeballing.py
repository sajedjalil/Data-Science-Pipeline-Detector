# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# loads data from the Input

def loadInput( fileName ):
    giftList = pd.read_csv( fileName, dtype=None, header=0 )
    return giftList

# dispatcher algorithm to find gift weight

def glovesWeight ( n ):
    x = np.random.rand( n )
    x[x<0.3] += 3
    return x
    
#
# Gift types and number of gift types have been hardcoded here.
#

def findGiftWeight( GiftType, n ):
    dispatcher = {'horse' : (np.maximum(0, np.random.normal(5,2,n))),
    'ball' : (np.maximum(0, 1 + np.random.normal(1,0.3,n))),
    'bike' : (np.maximum(0, np.random.normal(20,10,n))),
    'train' : (np.maximum(0, np.random.normal(10,5,n))),
    'coal' : (47 * np.random.beta(0.5,0.5,n)),
    'book' : (np.random.chisquare(2,n)),
    'doll' : (np.random.gamma(5,1,n)),
    'blocks' : (np.random.triangular(5,10,20,n)),
    'gloves' : (glovesWeight(n))}
    
    return dispatcher[GiftType]
    
# process Input and get statistical means

def processInput( fileName ):
    
    giftList = loadInput( fileName )
    nGifts = len(giftList)
    giftList['GiftType'] = giftList.GiftId.apply(lambda x: x.split('_')[0])
    giftList['GiftWeight'] = np.zeros(nGifts)
    giftList['GiftWeight'] = giftList.GiftType.apply(lambda x: findGiftWeight(x,1)[0])
        
    giftListSummary = pd.DataFrame()
    giftListSummary['GiftType'] = giftList['GiftType'].unique()
    nGiftTypes = len(giftListSummary['GiftType'])
    
    giftListSummary['nGifts'] = giftListSummary.GiftType.apply(lambda x : len(giftList[giftList['GiftType']==x]))
    giftListSummary['nGiftsPacked'] = 0
    giftListSummary['nGiftsNotPacked'] = giftListSummary.GiftType.apply(lambda x : len(giftList[giftList['GiftType']==x]))
    giftListSummary['weight_average'] = np.zeros(nGiftTypes)
    giftListSummary['weight_STD'] = np.zeros(nGiftTypes)
    
    n = 100000 #an arbitrarily large number for statistical analysis
    
    for i in np.arange(nGiftTypes):
        x = findGiftWeight(giftListSummary['GiftType'][i], n)
        giftListSummary['weight_average'][i] = np.average(x)
        giftListSummary['weight_STD'][i] = np.std(x)

    return giftList, giftListSummary
    
# write output - no fancy algorithm here - just eyeballing it and putting in names by inspection

def processOutput( ):
    
    fileName = '../input/gifts.csv'
    
    giftList, giftListSummary = processInput( fileName )
    
    packedBags = []
    
    for i in np.arange(1000):
        print(i)
        currentBag = []        
        itemCount = np.array([1,1,0,1,0,1,1,1,0])

        for i in np.arange(len(itemCount)):
            if (itemCount[i] <= giftListSummary['nGiftsNotPacked'][i]):
                for j in np.arange(itemCount[i]):
                    giftName = giftListSummary['GiftType'][i]
                    currGiftID = giftListSummary['nGiftsPacked'][i]
                    currentBag.append(giftName+'_'+str(currGiftID))
                    giftListSummary['nGiftsPacked'][i] += 1
                    giftListSummary['nGiftsNotPacked'][i] -= 1
        packedBags.append(currentBag)
        
    # Write to File 'submission.csv'
    
    subFile = open('submission.csv','w')
    subFile.write('Gifts\n')
    
    for currentBag in packedBags:
        subFile.write(currentBag[0])
        for currentItem in currentBag[1:]:
            subFile.write(' ')
            subFile.write(currentItem)
        subFile.write('\n')
    subFile.close()
    return packedBags

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.