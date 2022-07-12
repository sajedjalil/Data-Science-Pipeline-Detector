## [subcode] declarations

import matplotlib.pyplot as plt
import pandas as pd
import math as math
import numpy as np
import itertools as itr
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing

##first of all I noticed that same day images have ~ the same zoom and rotation.
##day1 and 2 have 1.3 zoom, while others about 1.
##days 1,2,3 have <10 degrees absolute rotation, while 4,5 have more then 10 degrees
##so I manually entered 2 points for each picture to convert it to csv data so I can work with it.

##this file works with csv file created by my C# programm I coded.
##This programm just shows you images, you can navigate previous/next.
##the goal of it is to set 2 points on image which are presented on each image in set.
##so train.csv and test.csv have similiar format:
##file_name,set_number,image_number,p1_x,p1_y,p2_x,p2_y

##now, based on these points we are calculating:
##zoom (I called that dm) (distance in current image / mean distance in set)
##rotation (I called that da) (current angle / mean angle in set)
##shift (dd) (distance between current point1(x/y) and mean point in set)

##calcultated values are used to teach SVM
##also we need to build all permutations to build so SVM can be ready for another order of images

##I tried all 3 da,dm,dd to predict. It showed 100% result on test set but very poor result on leaderboard ~0.14 so it was overfitted
##when I tried da and dm only I got 0.3
##then I tried dm only and got 0.7, so dm was the strongest feature
##this is the latest version I had, I am strongly classifying rotations and zooms for SVM, it gave me 0.72 on public and 0.77 on private leaderboard

##dm is delta multiply (zoom)
features_dm=['dm1','dm2','dm3','dm4','dm5']
##da is delta angle (rotation)
features_da=['da1','da2','da3','da4','da5']
##dd is delta distance (shift)
features_dd=['dd1','dd2','dd3','dd4','dd5']

features=features_dm+features_da

dfcols=['setId','day',\
        'i1','i2','i3','i4','i5',\
        'da1','da2','da3','da4','da5',\
        'dm1','dm2','dm3','dm4','dm5',\
        'dd1','dd2','dd3','dd4','dd5'
        ]

def getangle(x1,y1,x2,y2):
    a=-1*math.degrees(math.atan2(y2-y1,x2-x1))
    return a

def getangler(x):
    return getangle(x['p1_x'], x['p1_y'], x['p2_x'], x['p2_y'])

def getdistance(x1,y1,x2,y2):
    return math.hypot(x2 - x1, y2 - y1)

def getdistancer(x):
    return getdistance(x['p1_x'], x['p1_y'], x['p2_x'], x['p2_y'])

##main func - it converts points to da,dm,dd for train and test set, the same way
def calcfeatures(df,tv):
    print('da')

    df['angle'] = df.apply(getangler,axis=1)
    mean_angler=df.groupby(['set_number'])['angle'].mean()
    df['da']=df.apply(lambda r:abs(r.angle-mean_angler[r.set_number]),axis=1)

    df.dah=df.da
    datv=tv[0]
    df.loc[df.da<datv[0],'da']=1
    df.loc[(df.da>datv[0]) & (df.da<datv[1]) ,'da']=2
    df.loc[df.da>datv[1],'da']=3

    print('dm')
    
    df['distance']=df.apply(getdistancer,axis=1)
    mean_distancer=df.groupby(['set_number'])['distance'].mean()
    df['dm']=df.apply(lambda r:r.distance/mean_distancer[r.set_number],axis=1)

    df.dmh=df.dm
    dmtv=tv[1]
    df.loc[df.dm<dmtv,'dm']=1
    df.loc[df.dm>dmtv,'dm']=2
    
    print('dd')
    med_pos=df.groupby(['set_number'])['p1_x','p1_y'].mean()
    df['dd']=df.apply(lambda r:getdistance(\
        r.p1_x,r.p1_y,med_pos.loc[r.set_number].p1_x,med_pos.loc[r.set_number].p1_y)\
                                ,axis=1)
    
    print('normalize dd')
    mean_dd=df.groupby(['set_number'])['dd'].mean()    
    df['dd']=df.apply(lambda r:r.dd/mean_dd[r.set_number],axis=1)

    return df

## [subcode] load train


train_url = "../input/train.csv"
train = pd.read_csv(train_url)

##features treshold values to classify: da<10 degrees = 1, 10<da<14=2,da>14=3
##dm<1=1, dm>1=2
traintv=[[10,14],1]
train=calcfeatures(train,traintv)


print(train)
print(train.describe())

## [subcode] transform train



##here we transform train from format line per image to
##120 lines per set (all permutations to teach SVM

trainsets=train.set_number.unique()
trainsets=trainsets.astype(np.int32,copy=False)
trainsets.sort()

n=len(trainsets)*120


dftrain=pd.DataFrame(index=np.arange(0,n),columns=dfcols)
i=0
for set_n in trainsets:
    for perm in itr.permutations([1,2,3,4,5]):
        
        df=dftrain.loc[i]
        df.setId=set_n
        df.day='{} {} {} {} {}'.format(perm[0],perm[1],perm[2],perm[3],perm[4])

        df.i1=perm[0]
        df.i2=perm[1]
        df.i3=perm[2]
        df.i4=perm[3]
        df.i5=perm[4]

        df.dm1=train.loc[(train.set_number==set_n) & (train.image_number==perm[0]),'dm'].values[0]
        df.dm2=train.loc[(train.set_number==set_n) & (train.image_number==perm[1]),'dm'].values[0]
        df.dm3=train.loc[(train.set_number==set_n) & (train.image_number==perm[2]),'dm'].values[0]
        df.dm4=train.loc[(train.set_number==set_n) & (train.image_number==perm[3]),'dm'].values[0]
        df.dm5=train.loc[(train.set_number==set_n) & (train.image_number==perm[4]),'dm'].values[0]

        df.da1=train.loc[(train.set_number==set_n) & (train.image_number==perm[0]),'da'].values[0]
        df.da2=train.loc[(train.set_number==set_n) & (train.image_number==perm[1]),'da'].values[0]
        df.da3=train.loc[(train.set_number==set_n) & (train.image_number==perm[2]),'da'].values[0]
        df.da4=train.loc[(train.set_number==set_n) & (train.image_number==perm[3]),'da'].values[0]
        df.da5=train.loc[(train.set_number==set_n) & (train.image_number==perm[4]),'da'].values[0]

        df.dd1=train.loc[(train.set_number==set_n) & (train.image_number==perm[0]),'dd'].values[0]
        df.dd2=train.loc[(train.set_number==set_n) & (train.image_number==perm[1]),'dd'].values[0]
        df.dd3=train.loc[(train.set_number==set_n) & (train.image_number==perm[2]),'dd'].values[0]
        df.dd4=train.loc[(train.set_number==set_n) & (train.image_number==perm[3]),'dd'].values[0]
        df.dd5=train.loc[(train.set_number==set_n) & (train.image_number==perm[4]),'dm'].values[0]

        i=i+1

## [subcode] teach SVM



clf=svm.SVC()

X=dftrain.loc[:,features]
Y=dftrain.loc[:,'day']
clf.fit(X,Y)

dftrain.loc[:,'predicted']=clf.predict(X)


success=dftrain[dftrain.day==dftrain.predicted]
print('dftrain selector')

suc_cnt=len(success)
all_cnt=len(X)
print(suc_cnt)
print(all_cnt)
print(suc_cnt/all_cnt)


## [subcode] load  test


test_url = "../input/test.csv"
test = pd.read_csv(test_url)

##treshold values to classify are different for test set:
##da<6.764=1,6.764<da<8.903=2,da>8.903=3
##(tresholds were taken based on counts in each class)
##dm<1.00334=1,dm>1.00334=2
testtv=[[6.764,8.903],1.00334]
test=calcfeatures(test,testtv)

## [subcode] transform test

##transform to line per set format to predict days permutation

testsets=test.set_number.unique()
testsets=testsets.astype(np.int32,copy=False)
testsets.sort()

ntest=len(testsets)
dftest=pd.DataFrame(index=np.arange(0,ntest),columns=dfcols)

i=0
for set_n in testsets:
        df=dftest.loc[i]
        df.setId=set_n

        df.dm1=test.loc[(test.set_number==set_n) & (test.image_number==1),'dm'].values[0]
        df.dm2=test.loc[(test.set_number==set_n) & (test.image_number==2),'dm'].values[0]
        df.dm3=test.loc[(test.set_number==set_n) & (test.image_number==3),'dm'].values[0]
        df.dm4=test.loc[(test.set_number==set_n) & (test.image_number==4),'dm'].values[0]
        df.dm5=test.loc[(test.set_number==set_n) & (test.image_number==5),'dm'].values[0]

        df.da1=test.loc[(test.set_number==set_n) & (test.image_number==1),'da'].values[0]
        df.da2=test.loc[(test.set_number==set_n) & (test.image_number==2),'da'].values[0]
        df.da3=test.loc[(test.set_number==set_n) & (test.image_number==3),'da'].values[0]
        df.da4=test.loc[(test.set_number==set_n) & (test.image_number==4),'da'].values[0]
        df.da5=test.loc[(test.set_number==set_n) & (test.image_number==5),'da'].values[0]        

        df.dd1=test.loc[(test.set_number==set_n) & (test.image_number==1),'dd'].values[0]
        df.dd2=test.loc[(test.set_number==set_n) & (test.image_number==2),'dd'].values[0]
        df.dd3=test.loc[(test.set_number==set_n) & (test.image_number==3),'dd'].values[0]
        df.dd4=test.loc[(test.set_number==set_n) & (test.image_number==4),'dd'].values[0]
        df.dd5=test.loc[(test.set_number==set_n) & (test.image_number==5),'dd'].values[0]

        i=i+1

dftest.to_csv('dftest.csv',index=False)

## [subcode] load dftest

dftest=pd.read_csv('dftest.csv')

## [subcode] predict test

Xtest=dftest.loc[:,features]

dftest.loc[:,'day']=clf.predict(Xtest)

dfout=dftest[['setId','day']]
dfout.to_csv('SVM-dm-da-0_7.csv',index=False)
