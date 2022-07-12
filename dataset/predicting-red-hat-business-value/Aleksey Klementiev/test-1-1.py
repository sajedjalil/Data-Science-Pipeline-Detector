# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
peoples = pd.read_csv('../input/people.csv', parse_dates=['date'])
act_train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])
subm = pd.read_csv('../input/sample_submission.csv')

#peoples_act_train = pd.merge(act_train, peoples, how='left', on='people_id')

peoples_act_train = peoples.combine_first(act_train)

def calculateEntropy(mass):
    countElements = mass.size
    countElementsType1 = mass[(mass>0)].size
    countElementsType2 = mass[(mass==0)].size
    if countElements==0 or countElementsType1==0 or countElementsType2==0:
        return 1
    
    pType1 =  countElementsType1/countElements
    pType2 =  countElementsType2/countElements
    return -((pType1*math.log(pType1, 2))+(pType2*math.log(pType2, 2)))

def getCommonEntropies(ent1, ent2):
    return (ent1+ent2)/2
    
def splitMassByParameterWithTwoValues(mass, param, paramValueSeparate):
    mass1 = mass[mass[param]!=paramValueSeparate]
    mass2 = mass[mass[param]==paramValueSeparate]
    return mass1,mass2

#####

#act_train_cut = act_train[:10]
#peoples_cut = peoples[:10]
peoples_act_train_cut = peoples_act_train[:10000]
#peoples_act_train_cut = peoples_act_train

beforeEntropy = calculateEntropy(peoples_act_train_cut.outcome)
print(beforeEntropy)

###########################

import collections
import copy

def isFirstMassWithLessEntropy(mass1,mass2):
    if len(mass2) == 0:
        return False
    if (calculateEntropy(mass2)<calculateEntropy(mass1)):
        return False
    else:
        return True

def buildTreeBySequence(bestRes, seq, inputMass, resMass=[], usedAttr=[]):
    if len(seq) == 0:
        #print('END: >>>')
        #print('BEST:')
        if (len(bestRes['valsMass'])==0):
            bestRes['valsMass'] = resMass
        elif (isFirstMassWithLessEntropy(resMass, bestRes['valsMass'])):
            bestRes['valsMass'] = resMass
            bestRes['seq'] = usedAttr
        #print(bestRes['seq'])
        #print(collections.Counter(bestRes['valsMass']))
        #print('END: <<<')
        return

    item = seq[0]
    if len(seq) == 1:
        seq = []
    else:
        seq = seq[1:]
        
                    
    # split to two mass by attribute
    trueSeq, falseSeq = splitMassByParameterWithTwoValues(inputMass, item, True);
    
    # calculate entropy
    entropyTrueSeq = calculateEntropy(trueSeq.outcome)
    entropyFalseSeq = calculateEntropy(falseSeq.outcome)
    #print(entropyTrueSeq)
    #print(entropyFalseSeq)
    
    usedAttrUpdate = copy.deepcopy(usedAttr) 
    
    
    # compare entropy
    if entropyTrueSeq<beforeEntropy:
        #print(str(item)+': TRUE')
        resMass = trueSeq.outcome;
        usedAttrUpdate.append({'atr':item,'cond':'TRUE'})
        buildTreeBySequence(bestRes, seq, trueSeq, resMass, usedAttrUpdate)
    else:
        if (len(bestRes['valsMass'])==0):
            bestRes['valsMass'] = resMass
        elif (isFirstMassWithLessEntropy(resMass, bestRes['valsMass'])):
            bestRes['valsMass'] = resMass
            usedAttrUpdate.append({'atr':item,'cond':'STOP'})
            bestRes['seq'] = usedAttrUpdate

    if entropyFalseSeq<beforeEntropy:
        #print(str(item)+': FALSE')
        resMass = falseSeq.outcome;
        usedAttrUpdate.append({'atr':item,'cond':'FALSE'})
        buildTreeBySequence(bestRes, seq, falseSeq, resMass, usedAttrUpdate)
    else:
        if (len(bestRes['valsMass'])==0):
            bestRes['valsMass'] = resMass
        elif (isFirstMassWithLessEntropy(resMass, bestRes['valsMass'])):
            bestRes['valsMass'] = resMass
            usedAttrUpdate.append({'atr':item,'cond':'STOP'})
            bestRes['seq'] = usedAttrUpdate
        

    
    

atributesMass = ['char_10', 'char_11', 'char_12', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37'];
random.shuffle(atributesMass)
#atributesMass = ['char_10', 'char_11', 'char_12', 'char_14', 'char_15'];
bestRes = {'seq':[],'valsMass':[]}
#buildTreeBySequence(bestRes, atributesMass, peoples_act_train_cut, [], [])

#print(bestRes['seq'])
#print(collections.Counter(bestRes['valsMass']))

test_d = peoples_act_train_cut

#print(test_d)

# категориальный - char_1, char_4, char_5, char_6, char_7

data_cut = test_d[['char_1','char_4']]
print(test_d.corr())



"""
for x in peoples_act_train_cut.columns:
    print(x+': '+str(peoples_act_train_cut[x][0]))
"""
    
############################

""""
atrMass = ['char_10', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_37'];
for atrName in atrMass:
    twoPart = splitMassByParameterWithTwoValues(peoples_act_train_cut, atrName, True);
    entropyPartOne = calculateEntropy(twoPart[0].outcome)
    entropyPartSecond = calculateEntropy(twoPart[1].outcome)
    afterEntropy = getCommonEntropies(entropyPartOne, entropyPartSecond);
    print(atrName+': '+str(afterEntropy))
    

0.9909175980298324
char_10: 0.9684671584417361
char_11: 0.9685124159813121
char_12: 0.9680226042811657
char_13: 0.9685681407687741
char_14: 0.9683493810412944
char_15: 0.9683925423124553
char_16: 0.9691255229513481
char_37: 0.9686888534081011
"""

"""
uniqueChar1 = peoples_act_train.char_5.unique()

for atrValue in uniqueChar1:
    twoPart = splitMassByParameterWithTwoValues(peoples_act_train_cut, 'char_5', atrValue);
    if (twoPart[0].outcome).size == 0 or (twoPart[1].outcome).size==0:
        continue
    
    entropyPartOne = calculateEntropy(twoPart[0].outcome)
    entropyPartSecond = calculateEntropy(twoPart[1].outcome)
    afterEntropy = getCommonEntropies(entropyPartOne, entropyPartSecond);
    print(str(atrValue)+': '+str(afterEntropy))


0.9909175980298324
type 5: 0.9685361257409598
type 9: 0.9681289094933268
type 4: 0.9662593466582552
type 8: 0.9702305814756215
type 7: 0.9661982337935341
type 6: 0.9883250718497275
type 1: 0.9891350301199551
type 2: 0.9505214372688606
type 3: 0.9810303716680138
"""



#print('INFO:')
#print(peoples_act_train_cut) # have 'outcome'


# print(peoples) # char_1='type 1'/'type 2', char_38

# find all boolean char's params

