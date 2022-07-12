"""I was looking at Alexandre's work and was wondering how much you really need the specialized script. It turns out a lot! 
I wrote a script that allows you to plot all the non-events on top of each other. For instance, HandStart is 0 from 0 to 
2113 and 5110 to 7925 etc. You can visually compare how similar each electrode is for all the events in one trial on one 
subject. There are a lot of places where there is a crazy amount of offset that seems unrelated to the events at all. This 
is the type of thing that is sensitive to experimental limits which is probably dealt with in the MNE script (aka why 
the values can be predicted so well once you're using a specialized script). But other statistical methods will have a long
uphill climb if they're trying to ignore all the limits that the neuro people already know about."""

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

df_events = pd.read_csv('../input/train/subj1_series8_events.csv',header=0)
df_data = pd.read_csv('../input/train/subj1_series8_data.csv',header=0)
df_events = df_events.drop(['id'],axis=1)
result = pd.concat([df_events,df_data],axis=1)

def removeStr(string,start):
    string = string[start:]
    return int(string)

def getStartAndEndTimes(array,column,store):
    store.append(0)
    for i in range(0,117332):
        handStart_currentValue = array[i,column]
        handStart_nextValue = array[i+1,column]
        if handStart_currentValue == 0 and handStart_nextValue == 1:
            store.append(array[i,6])

def plotSimilar(array,sensor):
    i=0
    while i<(len(array)-2):
        result.loc[array[i]:array[i+1],:].plot(x=['id'],y=[sensor],figsize=(10,10))
        i=i+2
        

result_arr = result.values

for i in range(0,117333):
    result_arr[i,6]=removeStr(result_arr[i,6],14)

handStart_range=[]
firstDigitTouch_range=[]
bothStartLoadPhase_range=[]
liftOff_range=[]
replace_range=[]
bothRelease_range=[]

getStartAndEndTimes(result_arr,0,handStart_range)
getStartAndEndTimes(result_arr,1,firstDigitTouch_range)
getStartAndEndTimes(result_arr,2,bothStartLoadPhase_range)
getStartAndEndTimes(result_arr,3,liftOff_range)
getStartAndEndTimes(result_arr,4,replace_range)
getStartAndEndTimes(result_arr,5,bothRelease_range)

sensors = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC2','FC1','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10']

