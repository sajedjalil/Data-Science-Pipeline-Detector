import numpy as np
import csv
import pandas as pd
import string

#Load the data
gifts = pd.read_csv("../input/gifts.csv")



def scoreBags(bags):
    score=0.
    for b in bags:
        thisBagWeight = bagWeight(b)
        if thisBagWeight<=50. and len(b)>=3:
            score+=thisBagWeight
    return score


def probScore(bags):
    allScores = []
    for x in range(10):
        allScores.append(scoreBags(bags))
    allScores=np.array(allScores)
    return allScores.min(),allScores.max(),allScores.mean()



def gift_weight(gift):
    if "horse" in gift:
        return max(0, np.random.normal(5,2,1)[0])
    elif "ball" in gift:
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    elif "bike" in gift:
        return max(0, np.random.normal(20,10,1)[0])
    elif "train" in gift:
        return max(0, np.random.normal(10,5,1)[0])
    elif "coal" in gift:
        return 47 * np.random.beta(0.5,0.5,1)[0]
    elif "book" in gift:
        return np.random.chisquare(2,1)[0]
    elif "doll" in gift:
        return np.random.gamma(5,1,1)[0]
    elif "block" in gift:
        return np.random.triangular(5,10,20,1)[0]
    elif "gloves" in gift:
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    else:
        return -999999999



def probGiftWeight(gift):
    allW = []
    for i in range(3):
        allW.append(gift_weight(gift))
    return np.array(allW).mean()

def bagWeight(bag):
    return np.array([gift_weight(x) for x in bag]).sum()


def probBagWeight(bag):
    allWeights = []
    for x in range(10):
        allWeights.append(bagWeight(bag))
    allWeights=np.array(allWeights)
    return allWeights.mean()



#Apply an initial weight to the input gifts, and sort them ascending
gifts['InitialWeight']=gifts['GiftId'].apply(probGiftWeight)
gifts=gifts.sort_values('InitialWeight',ascending=True)



# Create 1000 empty bags
bags = []
for x in range(1000):
    bags.append([])


bagThresh = 36.515555


leftover = []

startj=0
for i in range(len(gifts)):
    thisGift = gifts.iloc[i][0]
    j=startj
    while j<1000 and thisGift is not None:
        if gift_weight(thisGift)<(bagThresh-probBagWeight(bags[j])):
            bags[j].append(thisGift)
            thisGift=None
            if bagWeight(bags[j])>=bagThresh:
                startj=j
            j=9999999999
        j+=1
    if thisGift is not None:
        leftover.append([thisGift,gift_weight((thisGift))])




#empty bags without enough items:


for i in range(1000):
    b=bags[i]
    if len(b)<3:
        for gift in b:
            leftover.append([gift,gift_weight(gift)])
        bags[i]=[]



leftover.sort(key=lambda x: x[1],reverse=False)

leftover2 = []
flipper=True
#verbose=True


emptyindexes =[]
for i in range(1000):
    if len(bags[i])<3:
        emptyindexes.append(i)



startj=0
for i in range(len(leftover)):
    leftover.sort(key=lambda x: x[1],reverse=flipper)
    thisGift = leftover[i][0]
    j=0
    while j<len(emptyindexes) and thisGift is not None:
        if gift_weight(thisGift)<(bagThresh-probBagWeight(bags[emptyindexes[j]])):
            bags[emptyindexes[j]].append(thisGift)
            thisGift=None
            if bagWeight(bags[emptyindexes[j]])>=bagThresh:
                startj=j
            if flipper:
                flipper=False
            else:
                flipper=True
        j+=1
    if thisGift is not None:
        leftover2.append([thisGift,gift_weight((thisGift))])


# Save the output

with open('santasubmit.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['Gifts'])
    for b in bags:
        if len(b)>=3:
            spamwriter.writerow([' '.join(b)])
            
            

