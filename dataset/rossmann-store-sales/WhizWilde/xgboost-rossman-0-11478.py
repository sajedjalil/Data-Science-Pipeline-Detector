# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:46:22 2015
based on codes and ideas from FoxTrot, Dune Dweller, Paso, Cast42, Ananya Mishra, Tushar Tilwankar, ML, and Chenlong Chen, as well as Scirpus and Neil Slater
@author: Whizwilde
Disclaimer: I am concious it is not clean, but it will edit a cleaner version later. I wanted to show the "raw" version to show where I used controls... Settings are what gave me my best single submission. I applied the 0.984 factor by hand.
Seems that monthdelta doesn't work here...
"""
import sys
import pandas as pd
import numpy as np
import datetime
import monthdelta
from isoweek import Week
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import operator
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt

def value_conversion(data):
    print('value_conversion')
    print(data.columns)

    print(datetime.datetime.now())
    
#Else, done at store level    
#    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
#    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
#    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
#    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
#    data['StoreType'] = data['StoreType'].astype(int)
#Else, done at store level
#    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
#    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
#    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
#    data['Assortment'] = data['Assortment'].astype(int)    
    
    data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = 1
    data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = 2
    data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = 3
    data['StateHoliday'] = data['StateHoliday'].astype(int)
    
    print('Finished value_conversion')
   #ints instead, or should i convert all to floats??
    print(datetime.datetime.now())    
    return data
    
    
    


def competition_setting(data,storecompetition,CurrentDateSeries):
    #Note:storecompetition=store['Store','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
    #TRAIN: Store    DayOfWeek    Date*    Sales    Customers    Open    Promo    StateHoliday    SchoolHoliday

    #data2 =data.drop['StoreType','Assortment']
    print('competition_setting') 
    print(datetime.datetime.now())
    data2 = data.merge(storecompetition, how='outer', on = 'Store', copy = False)
    del storecompetition
    print('data2 after merge')
    print(data2.columns)
    #print(data2.head(1))
    print(datetime.datetime.now())
    data2=value_conversion(data2)
    #I think this is more efficient to delete each df once not necessary anymore. I don't think merge destroy the original copies
    #truthmask=pd.DataFrame[[data2['year']>data2['CompetitionOpenSinceYear']]or [data2['year']==data2['CompetitionOpenSinceYear'] & data2['monthOfYear']>=data2['CompetitionOpenSinceMonth']]]
    print(datetime.datetime.now())

    #series of numpy bool    
    #correct the NA filling with corresponding dates (calculation was not possible before, but maybe i should feel this after merging)
    #data2['CompetitionOpenSinceYear']=data2['CompetitionOpenSinceYear'].astype(int)
    #data2['CompetitionOpenSinceMonth']=data2['CompetitionOpenSinceMonth'].astype(int)
    data2['CompetitionOpenSinceYear']=[int(data2['CompetitionOpenSinceYear'].iloc[i]) if not data2['CompetitionOpenSinceYear'].iloc[i]==-1 else data2['year'].iloc[i]+1 for  i in range (len(data2['year']))]
    data2['CompetitionOpenSinceMonth']=[int(data2['CompetitionOpenSinceMonth'].iloc[i]) if not data2['CompetitionOpenSinceMonth'].iloc[i]==-1 else data2['monthOfYear'].iloc[i] for  i in range (len(data2['year']))   ]
    #else data2['monthOfYear'].iloc[i] since +1 would potentially go >12. 

    #data2['CompetitionOpenSinceMonth']=data2['CompetitionOpenSinceMonth'].astype(int)
    #else later float error    
    #print ("type(data2['CompetitionOpenSinceMonth'])")    
    #print (type(data2['CompetitionOpenSinceMonth'][0]))#np.int64
    #print ((data2['CompetitionOpenSinceMonth'].dtype))
    
    competitionyearsupbool=np.array((data2['year'])>(data2['CompetitionOpenSinceYear']))
    data2['CompetitionOpenSinceMonth']=data2['CompetitionOpenSinceMonth'].astype(np.int32)
    competitionyeareqbool=np.array((data2['year'])==(data2['CompetitionOpenSinceYear']))  
    competitionmonthbool=np.array((data2['monthOfYear'])>=(data2['CompetitionOpenSinceMonth']))   #TODO or change for for i in range(len(data2['monthOfYear']))])
    competitionyeareq= np.array((competitionyeareqbool) & (competitionmonthbool) )    
    truthmask=np.array((competitionyearsupbool) |(competitionyeareq) )   #should be a numpy array)
    
    #segmented since original was not working:"""
    #truthmask=pd.DataFrame[data2['year']>data2['CompetitionOpenSinceYear'],data2['year']==data2['CompetitionOpenSinceYear'] & data2['monthOfYear']>=data2['CompetitionOpenSinceMonth']]    
    #truthmask=truthmask.any(1)
    #truthmask=pd.DataFrame[series for series in truthmask.iterrows()]
    #why would this not be working? """
    
    
    data2['ActiveCompetition']=truthmask
    
    #Replace the following:
    #competition distance set as -1 when competition not here/not here yet
    data2['CompetitionDistance']=[int(data2['CompetitionDistance'].iloc[i]) if element else -1 for (i,element) in enumerate(truthmask)    ]
    
    #Rossman9
    '''
    data2['CompetitionDistanceCategory']=[-1 if value==-1 else 1 if value<=20 else 3 if value<=100 else 5 if value<=500 else 10 if value<=5000 else 25 if value<=25000 else 50  for value in data2['CompetitionDistance']    ]
    print("data2['CompetitionDistanceCategory'].isnull().sum()")
    print(data2['CompetitionDistanceCategory'].isnull().sum())
    data2['CompetitionDistanceCategory'].fillna(-1, inplace=True)#Rossman 10. May cause the problem
    print(type(data2['CompetitionDistance']))
    print(type(data2['CompetitionDistance'][0]))
    #I guess it should be more efficient to use a df than a simple array? Still -1 for missing value too
    '''
    
       
#NOW CALCULATING THE DATES   CompetitionOpenSinceXmonthsWhereX:
       
    #CurrentDateSeries=pd.DataFrame(datetime.date(data2['year'][i],data2['monthOfYear'][i],1) for i in range (len(data2['year'])))
    ##Now calculated in preprocessing function. Avoid repeating

    CurrentDateSeries=np.array([datetime.date(data['year'][i],data['monthOfYear'][i],1) for i in range (len(data['year']))])
    print('CurrentDateSeries')    
    print(type(CurrentDateSeries))
    print(CurrentDateSeries.shape)    
    #print(CurrentDateSeries.dtypes)    
    print(CurrentDateSeries[0])
    print(type(CurrentDateSeries[0]))
    
    CompetitionOpenSinceDateSeries=np.array([datetime.date(data2['CompetitionOpenSinceYear'][i],int(data2['CompetitionOpenSinceMonth'][i]),1) if not data2['CompetitionOpenSinceYear'][i]==-1 else datetime.date(data2['year'][i],data2['monthOfYear'][i],1)  for i in range (len(data2['year']))])
    #CompetitionOpenSinceDateSeries=pd.DataFrame([datetime.date(data2['CompetitionOpenSinceYear'][i],int(data2['CompetitionOpenSinceMonth'][i]),1) if not data2['CompetitionOpenSinceYear'][i]==-1 else datetime.date(data2['year'][i],data2['monthOfYear'][i],1)  for i in range (len(data2['year']))])
    #for some reason, that doesn't work with monthdelta. Maybe because of datetime64 objects instead of datetime.date   
    print('CompetitionOpenSinceDateSeries.shape')  
    print(CompetitionOpenSinceDateSeries.shape)    
    #print(CompetitionOpenSinceDateSeries.dtypes)    
    #CompetitionOpenSinceDateSeries=pd.DataFrame([datetime.date(data2['year'][i],data2['monthOfYear'][i],1)  for i in range (len(data2['year']))], dtype=datetime.date)#test
    
    #Could be refined with something converting this tupple

    #    CompetitionOpenSinceDateSeries=pd.DataFrame(datetime.date(data2['CompetitionOpenSinceYear'][i],data2['CompetitionOpenSinceMonth'][i],1) for i in range (len(data2['year'])))
    #np.timedelta64(a, 'M')
    #simpler with predate correction
    #this will pass if     
    #data2['CompetitionOpenSinceXmonthsWhereX']=[datetime.timedelta(CurrentDateSeries[i]- CompetitionOpenSinceDateSeries[i])for i in range (len(data2['year']))]#little voluntary mistake since there are days past in the current month
    print('CurrentDateSeries[i]') 
    print(CurrentDateSeries[0])
    print(type(CurrentDateSeries[0]))
    #print(CurrentDateSeries[0].dtype)
    print(type(CurrentDateSeries))
#    print(CurrentDateSeries.dtypes)
    #print(len(CurrentDateSeries[0]))
    #print(len(CurrentDateSeries)) 
    print('CompetitionOpenSinceDateSeries[i]')     
    print(CompetitionOpenSinceDateSeries[0]) 
   # print(type(CompetitionOpenSinceDateSeries[0]) )
    #print(CompetitionOpenSinceDateSeries[0].dtype) 

    print(type(CompetitionOpenSinceDateSeries) )
    #print(CompetitionOpenSinceDateSeries.dtypes)    
    #print(len(CompetitionOpenSinceDateSeries[0]))
    #print( len(CompetitionOpenSinceDateSeries)) 
    print("len(data2['year']")    
    print(len(data2['year']))
    data2['CompetitionOpenSinceXmonthsWhereX']=[monthdelta.monthmod(CurrentDateSeries[i], CompetitionOpenSinceDateSeries[i])for i in range (len(data2['year']))]#little voluntary mistake since there are days past in the current month
    #data2['CompetitionOpenSinceXmonthsWhereX']  =[x[0].months if truthmask[i] else 0 for i,x in enumerate(data2['CompetitionOpenSinceXmonthsWhereX']) ] #tuples monthdelta, timedelta
    data2['CompetitionOpenSinceXmonthsWhereX']  =[x[0].months for i,x in enumerate(data2['CompetitionOpenSinceXmonthsWhereX']) ] #tuples monthdelta, timedelta
#    data2['CompetitionOpenSinceXmonthsWhereX'][data2['CompetitionOpenSinceXmonthsWhereX']<0]=0
    data2['CompetitionOpenSinceXmonthsWhereX'].loc[data2['CompetitionOpenSinceXmonthsWhereX']<0]=0
    print("type(data2['CompetitionOpenSinceXmonthsWhereX'][100])")    

    print(type(data2['CompetitionOpenSinceXmonthsWhereX'][100]))  
    print(data2['CompetitionOpenSinceXmonthsWhereX'][100])  

    #data2['CompetitionOpenSinceXmonthsWhereX']  =data2['CompetitionOpenSinceXmonthsWhereX'].iloc[data2['CompetitionOpenSinceXmonthsWhereX']<(0,0)]=(0,0)
    #data2['CompetitionOpenSinceXmonthsWhereX']  =data2['CompetitionOpenSinceXmonthsWhereX'][data2['CompetitionOpenSinceXmonthsWhereX']<0]=0  
    
    #doesn't work> astuples are immutables. So I will consider only days for now.    
    #data2['CompetitionOpenSinceXmonthsWhereX']  =[x if truthmask[i] else (0,0) for i,x in enumerate(data2['CompetitionOpenSinceXmonthsWhereX']) ] #tuples monthdelta, timedelta
    
    #data2['CompetitionOpenSinceXmonthsWhereX'][data2['CompetitionOpenSinceXmonthsWhereX'][0].months<0]=0
    
    #data2['CompetitionOpenSinceXmonthsWhereX'][data2['CompetitionOpenSinceXmonthsWhereX'][1].days<0]=0
    #month/time-delta tuples
    
    data2.drop(['ActiveCompetition','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1, inplace = True)    
    #Dropped 'ActiveCompetition' ROSSMAN10
    del truthmask    
    print ('data2_comp columns:'+data2.columns)
    print('Finished competition_setting') 
    print(datetime.datetime.now())
    return data2


    #data2['CompetitionOpenSince']=[storecompetition['CompetitionOpenSinceYear'][k]-data['year'][k]-storecompetition['CompetitionDistance'] if element else "0" for (k, element) in truthmask.enumerate()]
    
"""data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + \
    (data.month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1, 
             inplace = True)
"""
"""
    # Promo open time in months
    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + \
    (data.woy - data.Promo2SinceWeek) / float(4)
    data['PromoOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis = 1, inplace = True)
"""


def promo_setting(data,storepromo,CurrentDateSeries):
  ##Note:storepromo=store['Store','Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval']
    print('promo_setting')     
    print(datetime.datetime.now())
    data2 = data.merge(storepromo, how='outer', on = 'Store', copy = False)
    del storepromo
    #print('data2 after merge')
    #print(data2.columns)    
    #print(data2.head(1))

    data2['Promo2SinceYear']=[data2['Promo2SinceYear'].iloc[i] if not data2['Promo2SinceYear'].iloc[i]==-1 else data2['year'].iloc[i]+1 for  i in range (len(data2['year']))]
    
    
    print("prepromo data2['year'][i]")
    print(type(data2['year'][0]))
    print(data2['year'][0])
    print(type(data2['year']))
    #data2['Promo2SinceWeek']=[data2['Promo2SinceWeek'].iloc[i] if not data2['Promo2SinceWeek'].iloc[i]==-1 else Week(data2['year'][i],data2['weekOfYear'][i]).week for  i in range (len(data2['year']))   ]
    data2['Promo2SinceWeek']=[data2['Promo2SinceWeek'].iloc[i] if not data2['Promo2SinceWeek'].iloc[i]==-1 else data2['weekOfYear'][i] for  i in range (len(data2['year']))   ]#•TODO: ou data2['weekOfYear'].iloc[i]
    #This is voluntarily "inexact" as this column will be dropped and the current week will give a 0 in the next operation for promo2sincemonth
    data2['Promo2SinceWeek']=data2['Promo2SinceWeek'].astype(np.float64)#have to be float64 to be accepted by Week in>> Promo2SinceMonth=[int(Week(data2['Promo2SinceYear'][i],data2['Promo2SinceWeek'][i]).monday().month) for i in range (len(data2['year']))]
    print('Promo2SinceWeek')
    print(type(data2['Promo2SinceWeek']))
    print(type(data2['Promo2SinceWeek'][0]))    
    print ("test -1 in data2['Promo2SinceWeek']")
    print (-1 in data2['Promo2SinceWeek'])
    #else data2['monthOfYear'].iloc[i] since +1 would potentially go >12. 

    promoyearsupbool=np.array(data2['year']>data2['Promo2SinceYear'])
    promoyeareqbool=np.array(data2['year']==data2['Promo2SinceYear'])
    promomonthbool=np.array([data2['weekOfYear'][i]>=data2['Promo2SinceWeek'][i] for i in range(len(data2['weekOfYear']))])
# promomonthbool=[data2['weekOfYear'][i]>=Week(data2['Promo2SinceYear'][i],data2['Promo2SinceWeek'][i]).monday().year for i in range(len(data2['weekOfYear']))]
#   w eek    monday().year   
    promoyeareq= np.array((promoyeareqbool) & (promomonthbool))
    truthmask=np.array((promoyearsupbool)|(promoyeareq))#might be wrong
    data2['InPromo2']=truthmask    

    #I think this is more efficient to delete each df once not necessary anymore. I don't think merge destroy the original copies
    # truthmask=pd.DataFrame[data2['year']>data2['Promo2SinceYear'], data2['year']==data2['Promo2SinceYear'] & data2['dayOfYear']>=data2['Promo2SinceWeek']*7]
    #truthmask=truthmask.any(1)
    #truthmask=pd.DataFrame[series for series in truthmask.iterrows()]   
   
   
   #explicitMatch=np.array(data2['explicitMonth'][i] in data2['PromoInterval'][i] for i in range(len(data2['PromoInterval'])))    #si rétablit, drop 'PromoInterval'
   #print ("explicitMatch")    
   #print (explicitMatch.shape)    
   #print (data2['PromoInterval'].shape)
   #Not sure it works this way, maybe comparison is not element wise. Also problem with a lot of -1
#    data2['InPromo2']=truthmask    
    #data2['Promo2Now']=[explicitMatch and truthmask]
    #this may be wrong. We have some doubt about the meaning of PromoInterval
    #might have to built something to take the first month of promo2 each time then build over this
    #del explicitMatch
    print("data2['PromoInterval']")   
    print(type(data2['PromoInterval']))
    print(type(data2['PromoInterval'][0]))
    data2['Promo2Month']=np.array([(data2['monthOfYear'][i]-data2['PromoInterval'][i])%3 if data2['InPromo2'][i]==True else -1 for i in range(len(data2['PromoInterval']))])    #si rétablit, drop 'PromoInterval')   #Not sure it works this way, maybe comparison is not element wise. Also problem with a lot of -1)#should split promointerval etc)   
    #Rossman9 should check InPromO2 values but I believe this should be True/False and not -1 and else -1 and not 0
    print("data2['Promo2Month']")    
    
    data2['AnyPromoActive']=np.array([True if (data2['Promo2Month'][i]!=-1 & data2['Promo'][i]!=1)  else 0 for i in range(len(data2['PromoInterval']))])    
    #this define the number of the month is the trimester period of promo2. Will change the fillna -1 into 0
    #data2['Promo2Now']=[explicitMatch and truthmask]
    #this may be wrong. We have some doubt about the meaning of PromoInterval
    #might have to built something to take the first month of promo2 each time then build over this
    #del explicitMatch
    
    #.astype(int) #astype so this is binary #or with brackets
   # weekToMonth=lambda x: Week(2011, 40).monday()
    #can't figure out how to input the second parameter
    
    #CurrentDateSeries=pd.DataFrame(datetime.date(data2['year'][i],data2['monthOfYear'][i],1) for i in range (len(data2['year'])))
    ##Now calculated in preprocessing function. Avoid repeating


    ##Now calculated in preprocessing function. Avoid repeating
    Promo2SinceMonth=[int(Week(data2['Promo2SinceYear'][i],data2['Promo2SinceWeek'][i]).monday().month) for i in range (len(data2['year']))]#have to be int 
    Promo2SinceDateSeries=np.array([datetime.date(data2['Promo2SinceYear'][i],Promo2SinceMonth[i],1) for i in range (len(data2['year']))])
   # print('Promo2SinceWeek')
    #print(type(data2['Promo2SinceWeek']))
    #print(type(data2['Promo2SinceWeek'][0]))
    #print('Promo2SinceMonth')
    #print(type(Promo2SinceMonth))
    #print(type(Promo2SinceMonth[0]))
    #Promo2SinceDateSeries=pd.DataFrame([datetime.date(data2['Promo2SinceYear'][i],Week(data2['Promo2SinceYear'][i],data2['Promo2SinceWeek'][i]).monday(),1) for i in range (len(data2['year']))])

    del Promo2SinceMonth
#    data2.drop(['Promo2SinceYear', 'Promo2SinceWeek','explicitMonth','PromoInterval'], axis = 1, inplace = True)    #
    data2.drop(['Promo2','Promo2SinceYear', 'Promo2SinceWeek','PromoInterval'], axis = 1, inplace = True)    #

    print('Promo2SinceDateSeries')    
#    print(type(Promo2SinceDateSeries))
#    print(type(Promo2SinceDateSeries[0]))
#    print(Promo2SinceDateSeries)
#    print(Promo2SinceDateSeries[0])
#    print (-1 in Promo2SinceDateSeries)
#    print('CurrentDateSeries')    
#    print(type(CurrentDateSeries))
#    print(type(CurrentDateSeries[0]))
#    print(CurrentDateSeries)
#    print(CurrentDateSeries[0])    
#    print (-1 in CurrentDateSeries)
    #data2['Promo2SinceXmonthsWhereX']=[datetime.timedelta(CurrentDateSeries-Promo2SinceDateSeries)for i in range (len(data2['year']))]
#en jours utiliser timedelta
    data2['Promo2SinceXmonthsWhereX']=[monthdelta.monthmod(CurrentDateSeries[i], Promo2SinceDateSeries[i]) for i in range (len(data2['year']))]
#    data2['Promo2SinceXmonthsWhereX']  =[x[0].months if truthmask[i] else 0 for i,x in enumerate(data2['Promo2SinceXmonthsWhereX']) ] #tuples monthdelta, timedelta
#    data2['Promo2SinceXmonthsWhereX'].loc[data2['Promo2SinceXmonthsWhereX']<0]=-1
    data2['Promo2SinceXmonthsWhereX']  =[x[0].months if x[0].months>0 else -1 for i,x in enumerate(data2['Promo2SinceXmonthsWhereX']) ] #tuples monthdelta, timedelta
    #TODO: check if it works
#Maybe redundant. Have to check the algorithm to see whether or not the negative deltas are eliminated by the line with else 0

    print("type(data2['Promo2SinceXmonthsWhereX'][100])")    
    
    print(type(data2['Promo2SinceXmonthsWhereX'][100]))  
    print(data2['Promo2SinceXmonthsWhereX'][100])  

    #data2['Promo2SinceXmonthsWhereX']  =[int(x) if truthmask[i] else int(0) for i,x in enumerate(data2['Promo2SinceXmonthsWhereX']) ]
#'explicitMonth' not needed anymore

    #how to program something to calculate the time since the beginning of the promo2?
    del truthmask
    print ('data2_comp columns:'+data2.columns)
    print('Finished promo_setting')     
    print(datetime.datetime.now())
    return data2

def stats_setting3(data):
    
    print('Stats settings')
    print(data.columns)
    print(data.dtypes) 
    #sys.exit()
#To Test: 'StateHoliday', 'InPromo2'instead of the month 'AnyPromoActive''year'
#already tested: 'StoreType' 'Assortment''ActiveCompetition''Promo2Month'',,'monthOfYear', 'WeekEnd'#useless with day of week,'weekOfYear','Store'
#Seems to sparse: 'dayOfMonth', 'dayOfYear',
#Not to test: 'CompetitionOpenSinceXmonthsWhereX', 'Promo2SinceXmonthsWhereX', 'CompetitionDistance' as they are increasing or the sale for a given store 'CompetitionDistanceCategory'instead of 'ActiveCompetition'
#Dropped and used to define other variables:'Promo2''Promo2SinceWeek','Promo2SinceYear','PromoInterval''CompetitionOpenSinceMonth','CompetitionOpenSinceYear'

#should I set to -1 Std or bigger. Idem with fillna for missing values?
#Could I have avoided to make code copy conversion here (copy+changing names, functions, to which extent? first lists, methods?)

    print("Convert Sales to the right type")
   # print(type(data['Sales']))
    #print(type(data['Sales'][0]))
    #data['Sales']=np.log(data['Sales'].astype(float)+1)#WARNING: COULD BE A MISTAKE
    data['Sales']=np.log1p(data['Sales'].astype(float))
    data['Customers']=np.log1p(data['Customers'].astype(float))#Rossman9 think this has to be tested
    
    #Log will be used instead of     
    #maybe useful for not breaking means?
    #necessary since some value are not numerical and int will cause the code to break
   
     #rossman 9.5
#First Part: by StoreType and Assortment
   # type_assortment_DoW_select=['StoreType','Assortment','Promo','Promo2Month','DayOfWeek']
#Rossman10
#Second Part: by specific store
    store_DoW_select=['Store','Promo','DayOfWeek']

#Second Block:store_DoW
#wonder if can do a groupby of groupb
    print("store_DoW_sales_means")
    store_DoW_sales_means = data.groupby(store_DoW_select)['Sales'].mean()
    store_DoW_sales_means.name = 'store_DoW_sales_means'
    store_DoW_sales_means = store_DoW_sales_means.reset_index()
    data = pd.merge(data, store_DoW_sales_means, on = store_DoW_select, how='left')
        
    print("store_DoW_customers_means")
    store_DoW_customers_means = data.groupby(store_DoW_select)['Customers'].mean() #.Customers won't work. Why?
    store_DoW_customers_means.name = 'store_DoW_customers_means'
    store_DoW_customers_means=store_DoW_customers_means.reset_index()    
    data = pd.merge(data, store_DoW_customers_means, on = store_DoW_select, how='left')
    
   #data['Sales','Customers','SalesMean','CustomersMean','SalesMedian','CustomersMedian']=data['Sales','Customers','SalesMean','CustomersMean','SalesMedian','CustomersMedian'].astype(int)
#to spare memory?

    print("store_DoW_sales_std")
    store_DoW_sales_std = data.groupby(store_DoW_select)['Sales'].std()
    store_DoW_sales_std.name = 'store_DoW_sales_std'
    store_DoW_sales_std = store_DoW_sales_std.reset_index()
    data = pd.merge(data, store_DoW_sales_std, on = store_DoW_select, how='left')

    
    print("store_DoW_customers_std")
    store_DoW_customers_std = data.groupby(store_DoW_select)['Customers'].std()
    store_DoW_customers_std.name = 'store_DoW_customers_std'
    store_DoW_customers_std = store_DoW_customers_std.reset_index()
    data = pd.merge(data, store_DoW_customers_std, on = store_DoW_select, how='left')

    data['store_DoW_sales_means'].fillna(0, inplace=True)
    data['store_DoW_customers_means'].fillna(0, inplace=True)
    data['store_DoW_sales_std'].fillna(0, inplace=True)
    data['store_DoW_customers_std'].fillna(0, inplace=True)

#grouping for Month trend

    store_month_select=['Store','Promo','monthOfYear']

#Second Block:store_DoW
#wonder if can do a groupby of groupb
    print("store_month_sales_means")
    store_month_sales_means = data.groupby(store_month_select)['Sales'].mean()
    store_month_sales_means.name = 'store_month_sales_means'
    store_month_sales_means = store_month_sales_means.reset_index()
    data = pd.merge(data, store_month_sales_means, on = store_month_select, how='left')
        
    print("store_month_customers_means")
    store_month_customers_means = data.groupby(store_month_select)['Customers'].mean() #.Customers won't work. Why?
    store_month_customers_means.name = 'store_month_customers_means'
    store_month_customers_means=store_month_customers_means.reset_index()    
    data = pd.merge(data, store_month_customers_means, on = store_month_select, how='left')
    
   #data['Sales','Customers','SalesMean','CustomersMean','SalesMedian','CustomersMedian']=data['Sales','Customers','SalesMean','CustomersMean','SalesMedian','CustomersMedian'].astype(int)
#to spare memory?

    print("store_month_sales_std")
    store_month_sales_std = data.groupby(store_month_select)['Sales'].std()
    store_month_sales_std.name = 'store_month_sales_std'
    store_month_sales_std = store_month_sales_std.reset_index()
    data = pd.merge(data, store_month_sales_std, on = store_month_select, how='left')

    
    print("store_month_customers_std")
    store_month_customers_std = data.groupby(store_month_select)['Customers'].std()
    store_month_customers_std.name = 'store_month_customers_std'
    store_month_customers_std = store_month_customers_std.reset_index()
    data = pd.merge(data, store_month_customers_std, on = store_month_select, how='left')

    data['store_month_sales_means'].fillna(0, inplace=True)
    data['store_month_customers_means'].fillna(0, inplace=True)
    data['store_month_sales_std'].fillna(0, inplace=True)#à dropper. Kept only for observation
    data['store_month_customers_std'].fillna(0, inplace=True)#à dropper. Kept only for observation

#Rossman10 #This is tricky but sales means are transmitted to the whole set
    print('Special_month_sales')
    data['Special_month_sales']=np.round((data['store_month_sales_means']-data['store_DoW_sales_means'])/data['store_DoW_sales_std'],1)
    print('Special_month_customers')
    data['Special_month_customers']=np.round((data['store_month_customers_means']-data['store_DoW_customers_means'])/data['store_DoW_customers_std'],1)

    print('Special_day_sales')
    data['Special_day_sales']=np.round((data['store_DoW_sales_means']-data['store_month_sales_means'])/data['store_month_sales_std'],1)
    print('Special_day_customers')
    data['Special_day_customers']=np.round((data['store_DoW_customers_means']-data['store_month_customers_means'])/data['store_month_customers_std'],1)



#rossman9-5
#    data['specialweek_SalesMean_IP2']=np.array([((data['Sales'][i]-data['store_IP2_DoW_sales_means'][i])/data['store_IP2_DoW_sales_std'][i])  for i in range (len(data['year']))])  
#    data['specialweek_SalesMedian_IP2']=np.array([((data['Sales'][i]-data['store_IP2_DoW_sales_medians'][i])/data['store_IP2_DoW_sales_std'][i])  for i in range (len(data['year']))])  
#    data['specialweek_CustomersMean_IP2']=np.array([((data['Customers'][i]-data['store_IP2_DoW_customers_means'][i])/data['store_IP2_DoW_customers_std'][i]) if data['Customers'][i]!='NaN' for i in range (len(data['year']))])    
#    data['specialweek_CustomersMedian_IP2']=np.array([((data['Customers'][i]-data['store_IP2_DoW_customers_medians'][i])/data['store_IP2_DoW_customers_std'][i]) if data['Customers'][i]!='NaN' for i in range (len(data['year']))])    
#    data['specialweek_SalesMean_P2M']=np.array([((data['Sales'][i]-data['store_DoW_sales_means'][i])/data['store_IP2_DoW_sales_std'][i])  for i in range (len(data['year']))])  
#    data['specialweek_SalesMedian_P2M']=np.array([((data['Sales'][i]-data['store_DoW_sales_medians'][i])/data['store_IP2_DoW_sales_std'][i])  for i in range (len(data['year']))])  
#    data['specialweek_CustomersMean_P2M']=np.array([((data['Customers'][i]-data['store_DoW_customers_means'][i])/data['store_DoW_customers_std'][i]) if data['Customers'][i]!=None for i in range (len(data['year']))])    
#    data['specialweek_CustomersMedian_P2M']=np.array([((data['Customers'][i]-data['store_DoW_customers_medians'][i])/data['store_DoW_customers_std'][i]) if data['Customers'][i]!=None  for i in range (len(data['year']))])    

    #'''
#Should try different groupings for means and medians #Was not effective with InPromo2, Promo2Month, Storetype, Assortment
#also should eval progression per year on mean month 
#take inflation into account to lower prices in training first then add them again: using google trends but no time
#also for population variations
#also for migratory movements
#also for maps places/weather
#
#### #rossman7
    print('Finished Processing Means, Medians')
   #ints instead, or should i convert all to floats??
    print(datetime.datetime.now())    
    return data

def preprocessing(train,test,store,test_index):
    print('preprocessing')    
    #print ('top test value', train.index)
    #train['Id']=0    
    train['Id']=train.index+test_index+1 #train index start at 0 so it will double the final ID value of test, erroneously, unless we add one.
   #is it possible to do train.index=train.index+1
    print ('data selected on open')
    
       
    
    train=train[train['Open']!=0]# replace the #data=data[data['Open']!=0] line. Select only 
    
   #print('data selection done')

    #print(type(train))
    #Should be refined but will do for now.
    #print(type(test))
#""" Doesn't work, no idea why?
# booleantest=np.array([data['Id'][i]!=0 for i in range(len(data['Id']))] )
#    print('boolean test done')
#    booleantrain=np.array([[data['Id'][i]==0 & data['Open']!=0] for i in range(len(data['Id']))])#S6
#    print('boolean train done')    
#    boolean=booleantrain|booleantest
#    print(boolean)
#    print('boolean done')
#    print(data.info())
#    data=data[boolean]"""
    test['Sales']='NaN'#'?' will break the means
    test['Customers']='NaN'
    #test['Sales']=None /np.nan
    #test['Customers']=None/np.nan

    #☺test['Open'].fillna(1, inplace=True)#better predict them than not
#    store['CompetitionOpenSinceYear'].fillna(-1, inplace=True)    #Nan would cause unorderable problems
#    store['CompetitionOpenSinceMonth'].fillna(1, inplace=True)    #Nan would cause unorderable problems    
    testsave=test['Open']    
    testsave.fillna(1, inplace=True)# will be simpler on testsave than on the whole test
    
    lists=[train,test]
    data=pd.concat(lists, ignore_index=True)#tracing shown that I had to ignore_index
    #rossman10: suppress this 
    #rossman 9.5
#    print("store_open_days")
#    store_open_days = data.groupby(['Store'])['Open'].sum()
#    store_open_days.name = 'store_open_days'
#    store_open_days = store_open_days.reset_index()
#    data = pd.merge(data, store_open_days, on = ['Store'], how='left')
#    #could be better to 
    
    data.drop(['Open'], axis = 1,inplace = True)#for some reason, data=... will cause a none type?

    
    #print(type(data))
    #print(data.columns)
    #print(type(data))     
    #print(data.values)
    #print('original batch header') 
    #print(type(data)) 
    #print(data.columns)    
    #print(data.head(1))    
   
    #add date columns
       
    #print(type
    data['year'] = data['Date'].dt.year.astype(int)
    #print(type((data['year']).iloc[0]))#numpy.int64. Should I downsize this type? or use c_type and unsigned integers?
    #print((data['year']).iloc[0])
    data['monthOfYear'] = data['Date'].dt.month.astype(int)
    #print(type(data['monthOfYear'][1]))
    #print(type(data['monthOfYear'].iloc[1]))
    data['dayOfMonth'] = data['Date'].dt.day.astype(int)
    #print(type(data['dayOfMonth'][1]))
    #data['dayOfYear'] = data['Date'].timetuple().tm_yday
    
    #subset = [[data['year'].astype(int), data['monthOfYear'].astype(int),data['dayOfMonth'].astype(int)]]#yields a column of three different columns
    #subset = [data['year'].astype(int), data['monthOfYear'].astype(int),data['dayOfMonth'].astype(int)]#yields three different columns
    """#subset = [[ data['year'].iloc[i],data['monthOfYear'].iloc[i],data['dayOfMonth'].iloc[i] ] for i in range(len(data['year']))]
    #data['dayOfYear']=[datetime.date(date[0],date[1],date[2]).timetuple().tm_yday() for uniquedate in subset]
    #data['weekOfYear']=[datetime.date(uniquedate[0],uniquedate[1],uniquedate[2]).isocalendar()[1] for uniquedate in subset]
 """
#THIS WORKs and yield same results than other methods. But they will be preferred for the sake of uniformity of code and avoidance of useless calculations (subset)
#    print('calculating day/week of year /from datetime')
#    print(type(data['Date']))
#    print(type(data['Date'][0]))
#    print(type(data['Date'][110].dayofyear))
#    print(type(data['Date'][50].weekofyear))
#    subset = [  datetime.date(data['year'].iloc[i],data['monthOfYear'].iloc[i],data['dayOfMonth'].iloc[i])  for i in range(len(data['year']))]
#    data['dayOfYear']= [int(uniquedate.timetuple().tm_yday) for uniquedate in subset]
#    data['weekOfYear']=[int(uniquedate.isocalendar()[1]) for uniquedate in subset]
#    data['explicitMonth']=[i.strftime("%b") for i in data['Date']] #or would this be better to use apply with a lambda?
#    print(type(data['dayOfYear']))
#    print(type(data['dayOfYear'][110]))
#    print(data['dayOfYear'][110])
#    print(type(data['weekOfYear']))
#    print(type(data['weekOfYear'][50]))
#    print(data['weekOfYear'][50])    

    print('calculating day/week of year /from date/timestamps')
    data['dayOfYear']= [int(uniquedate.dayofyear) for uniquedate in data['Date']] #can also be achieved through use of .timetuple().tm_yday on a datetime.date object (not sure on a timestamp)
    data['weekOfYear']=[int(uniquedate.weekofyear) for uniquedate in data['Date']] #can also be achieved through use of .isocalendar()[1] on a datetime.date object (not sure on a timestamp)
#    data['explicitMonth']=[uniquedate.strftime("%b") for uniquedate in data['Date']] #or would this be better to use apply with a lambda?
#'explicitMonth is not needed anymore. Nice trick but better to process numbers here.

    print('calculating WeekEnds')
    data['explicitDay']=[ uniquedate.strftime("%a") for uniquedate in data['Date']] #or would this be better to use apply with a lambda?
    weekend=['Sat','Sun']
    data['WeekEnd']=[1 if day in weekend else 0 for day in data['explicitDay']]
    del weekend
    data.drop(['Date','explicitDay' ], axis = 1,inplace = True)
#    data.drop(['explicitDay' ], axis = 1,inplace = True) #Might need not  to drop 'Date' at this point

#TODO: how to put last week sales/problem with reset index/time index?
#    data_dateindexed= data.groupby([ 'Store', 'Date'])
#    print(data_dateindexed.index)
#    print(data_dateindexed.head(20))
#    data_dateindexed=data_dateindexed.reset_index()  
#    data_dateindexed=pd.to_datetime(data_dateindexed['Date'])
#    print('data_dateindexed[0:50] ') 
#    print('data_dateindexed[1117:1167] ') 
#    print(data_dateindexed[0:50] )   
#    print(data_dateindexed[1117:1167] ) 
#    
#    print(data.columns)
#    data = pd.merge(data, data_dateindexed, how='left')
#    data.drop(['Date'], axis = 1,inplace = True)
 



   #print('typesubset:')
   #print(type(subset))
   #print('subset[0]:')
   #print (subset[0])
   #print(type(subset[0]))
    #Week(data2['Year'][i],data2['SinceWeek'][i]).monday()>datetime
#        
    #v2 faster
    
    #print('typesubset:')
    #print(type(subset))
    #print (subset.index)
    #print('subset[0]:')
    #print(subset[0])
    
    #Or values.astype(int) but not sure it doesn't mess with the index?
    #tuples = [tuple(x) for x in subset.values]
    #data['dayOfYear']=pd.DataFrame.from_records(tuples).timetuple().tm_yday()
    #data['weekOfYear']=pd.DataFrame.from_records([tuple.timetuple().isocalendar()[1] for tuple in tuples])
    
    #data['dayOfYear']=[datetime.date(date[0],date[1],date[2]).timetuple().tm_yday() for uniquedate in subset]
       
    #data['explicitMonth']=data['Date'].dt.strftime("%b") #DatetimePropertie
#TODO This is useless and  I guess prone to error to use this for comparison with the string of promo2. It will be simpler to convert it.
    
   #print("sample data['explicitMonth']")
    #print(type(data['explicitMonth'][1]))    
    #print(data['explicitMonth'][1])
#TODO On the other hand, this one is acceptable as complexity depends a lot on the number of days.
    #print("sample data['explicitDay']")    
    #print(type(data['explicitDay'][1]))
    #print(data['explicitDay'][1])   
    #I wonder if I could have done this from 'Date' directly?
    print('Finished basic dates calculations')
    print(data.columns)   
    
    store.fillna(-1, inplace=True) #Nan would cause unorderable problems
    store.loc[ store['PromoInterval'] == 'Jan,Apr,Jul,Oct', 'PromoInterval'] = 0
    store.loc[ store['PromoInterval'] == 'Feb,May,Aug,Nov', 'PromoInterval'] = 1
    store.loc[ store['PromoInterval'] == 'Mar,Jun,Sept,Dec', 'PromoInterval'] = 2
    store['PromoInterval']=store['PromoInterval'].astype(int)    
    #could be done later but it will make this easier    

    storeconstants=store[['Store','StoreType','Assortment']]
    storeconstants.loc[storeconstants['StoreType'] == 'a', 'StoreType'] = '1'
    storeconstants.loc[storeconstants['StoreType'] == 'b', 'StoreType'] = '2'
    storeconstants.loc[storeconstants['StoreType'] == 'c', 'StoreType'] = '3'
    storeconstants.loc[storeconstants['StoreType'] == 'd', 'StoreType'] = '4'
    storeconstants['StoreType'] = storeconstants['StoreType'].astype(int)
    
    storeconstants.loc[storeconstants['Assortment'] == 'a', 'Assortment'] = '1'
    storeconstants.loc[storeconstants['Assortment'] == 'b', 'Assortment'] = '2'
    storeconstants.loc[storeconstants['Assortment'] == 'c', 'Assortment'] = '3'
    storeconstants['Assortment'] = storeconstants['Assortment'].astype(int)    
    data = data.merge(storeconstants, how='outer', on = 'Store', copy = False)
    print(type(data['Assortment'][0]))
    print(type(data['StoreType'][0]))
    
    
    storecompetition=store[['Store','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']]#
    storepromo=store[['Store','Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval']]#Promo2SinceWeek is int64 with is a problem later
    storepromo['Promo2SinceWeek']=storepromo['Promo2SinceWeek'].astype(int)
    storepromo['Promo2SinceYear']=storepromo['Promo2SinceYear'].astype(int)
    print("storetypes")

# Build Current Date Series to use further to define time gaps   
  
    #CurrentDateSeries=pd.DataFrame([datetime.date(data['year'][i],data['monthOfYear'][i],1) for i in range (len(data['year']))])
    print('preprocessing CurrentDateSeries[i]') 
        
    CurrentDateSeries=np.array([datetime.date(data['year'][i],data['monthOfYear'][i],1) for i in range (len(data['year']))])
    #TODO, check: Is it still useful?would it work with timestamps from 'date' in case I would not have to do this and not drop 'Date' earlier
    data = competition_setting(data,storecompetition,CurrentDateSeries)
    data = promo_setting(data,storepromo,CurrentDateSeries)
    data = stats_setting3(data)#rossman9-5
   
    print('Finished preprocessing')
    print(data.columns)
    #print(data.head(5))
    return data,testsave
#store['Store','Promo2','Promo2SinceWeek','Promo2SinceYear','PromoInterval']   


def rmspe_metric(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_metric2(y, yhat):
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return rmspe    

"""    
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return 'rmspe', rmspe_metric(y,yhat)
"""
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w
def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

"""
def custobj(preds, dtrain):
    labels = dtrain.get_label()
    grad = [(y - yh) / y ** 2 if y != 0 else 0 for (yh, y) in zip(preds, labels)]
    hess = [1. / y ** 2 if y != 0 else 0 for y in labels]
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'RMSPE', np.sqrt(np.average([(1 - yh/y) ** 2 for (yh, y) in zip(preds, labels) if y != 0]))
"""
##############################################################
print ('start!')
print(datetime.datetime.now())
   
##Block for normal processing
train_file = '../input/train.csv'
test_file = '../input/test.csv'
store_file = '../input/store.csv'
output_file = 'predictions.csv'

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
train = pd.read_csv( train_file,parse_dates=['Date'], dtype={'StateHoliday': object},date_parser=dateparse )
#engine=python? use?
test = pd.read_csv( test_file,parse_dates=['Date'],dtype={'StateHoliday': object},date_parser=dateparse )
test_index=len(test['Id'])
store = pd.read_csv( store_file )

data,testsave=preprocessing(train,test,store,test_index)#TODO: testsave is never used elsewhere use it or erase it
data.to_csv('./input/pdata.csv', index=False)
###END Block for normal processing

##Block for quick processing
#data_file = './input/pdata.csv'
#data = pd.read_csv( data_file )
#test_index=41088
##END Block for quick processing

#Patch fill NA:
#data.fillna(0, inplace=True)
#Now this block is valid for both quick and table processing
#TODO: mean normalization/scaling into preprocessing?

print("data.columns after preprocessing")
print(data.columns)
passed=['InPromo2','store_month_sales_std','store_month_customers_std','store_DoW_customers_std','store_DoW_sales_std','store_month_customers_means','store_month_customers_medians','store_DoW_customers_means','store_DoW_customers_medians','Special_month_customers','Special_day_customers','AnyPromoActive']
#
#No medians:        'type_assortment_DoW_sales_medians','store_DoW_sales_medians','Except_store_P2M_DoW_sales_medians',
#No IP2: 'type_assortment_IP2_DoW_sales_means','type_assortment_IP2_DoW_sales_medians','type_assortment_IP2_DoW_sales_std','store_IP2_DoW_sales_means','store_IP2_DoW_sales_medians','store_IP2_DoW_sales_std','Except_store_IP2_DoW_sales_means','Except_store_IP2_DoW_sales_medians'
#NO CUSTOMER based data: 'Customers','total_customers','total_year_customers','type_assortment_DoW_customers_means','type_assortment_DoW_customers_medians','type_assortment_DoW_customers_std','store_DoW_customers_means','store_DoW_customers_medians','store_DoW_customers_std','Except_store_P2M_DoW_customers_means','Except_store_P2M_DoW_customers_medians','type_assortment_IP2_DoW_customers_means','type_assortment_IP2_DoW_customers_medians','type_assortment_IP2_DoW_customers_std','store_IP2_DoW_customers_means','store_IP2_DoW_customers_medians','store_IP2_DoW_customers_std',Except_store_IP2_DoW_customers_means','Except_store_IP2_DoW_customers_medians'],
#'store_DoW_sales_std','type_assortment_IP2_DoW_sales_means','type_assortment_IP2_DoW_sales_std'

data.drop(passed, axis = 1, inplace = True)#EDIT to vary initial input
print("data.columns after preprocessing and data dropping")
print(data.columns)
train_final=data[(data['Id'])>test_index]
test_final=data[data['Id']<=test_index]



train_goal=train_final['Sales']


"""
#Used to check data 
train_final.to_csv('./input/ptrain2.csv', index=False)
test_final.to_csv('./input/ptest2.csv', index=False)
"""
#X_train, X_valid = cross_validation.train_test_split(train_final, test_size=0.01)
test_size=0.012
random_state=10
X_train, X_valid = cross_validation.train_test_split(train_final, test_size=test_size, random_state=random_state)
#X_train=train_final[train_final['year']<2015]
#X_valid=train_final[train_final['year']==2015]
X_train_goal=X_train['Sales']
X_valid_goal=X_valid['Sales']

#If script works, then it comes from the conditional

#X_train_goal = np.log1p(X_train.Sales)
#X_valid_goal = np.log1p(X_valid.Sales)
#X_train_goal = X_train.Sales
#X_valid_goal = X_valid.Sales
#X_train_goal = np.array(X_train.Sales)
#X_valid_goal = np.array(X_valid.Sales)

#print ("test null starts")
#print (X_train_goal.isnull().sum())
#print (X_valid_goal.isnull().sum())
#print (X_train.isnull().sum())
#print (X_valid.isnull().sum())
#print ("test null finished")

X_train.drop(['Id','Sales','Customers'], axis = 1, inplace = True)
X_valid.drop(['Id','Sales','Customers'], axis = 1, inplace = True)
train_final.drop(['Id','Sales','Customers'], axis = 1, inplace = True)
train_final.to_csv('./input/ptrain.csv', index=False)
test_index=test_final['Id']
test_final.drop(['Id','Sales','Customers'], axis = 1, inplace = True) 
test_final.to_csv('./input/ptest.csv', index=False)


#TRAIN: Store    DayOfWeek    Date    Sales    Customers    Open    Promo    StateHoliday    SchoolHoliday
#Test: Id    Store    DayOfWeek    Date    Open    Promo    StateHoliday    SchoolHoliday
#Store: Store    StoreType    Assortment    CompetitionDistance    CompetitionOpenSinceMonth    CompetitionOpenSinceYear    Promo2    Promo2SinceWeek    Promo2SinceYear    PromoInterval

params = {"objective": "reg:linear",#reg:logistic
          "booster" : "gbtree",#gblinear
          "base_score":0.5,# [ default=0.5 ]
          
          "eta": 0.021,
          "gamma":0,#default=0     minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be.    range: [0,∞]
          
          "max_depth": 23,
          "min_child_weight": 23, #Default=1 minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. 
          "max_delta_step":0, # [default=0]#    Maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update    range: [0,∞]

          "subsample": 0.77,#0.9# 0.85#0.5, #0.7 #Default=1
          "colsample_bytree": 1,#0.4#0.5,# 0.7,#Default=1
          "lambda":1,       #Default=1
          "alpha":0,       #Default=0
          "silent": 0,
          "thread": 4,
          "seed": 2501
          }
num_boost_round = 3000
early_stopping_rounds=421
'''
[1689]  train-rmspe:0.057056    eval-rmspe:0.094322

Validating
error0.09432
'''


print("Train a XGBoost model")
#X_train, X_valid = train.head(len(train) - val_size), train.tail(val_size)
dtrain = xgb.DMatrix(X_train, X_train_goal)
dvalid = xgb.DMatrix(X_valid, X_valid_goal)
dtest = xgb.DMatrix(test_final)
#watchlist = [(dvalid, 'eval'), (dtrain, 'train')]#IMPORTANT or reverse???
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]#original
#watchlist = [(dvalid, 'train'), (dvalid, 'eval')]#TEST>>inf/inf
#watchlist = [(dvalid, 'train'), (dtrain, 'eval')]#TEST>>inf/inf
#watchlist = [(dtrain, 'train'), (dtrain, 'eval')]#TEST
#xgb.cv(params, dtrain, num_round, nfold=3, seed = 0, obj = custobj, feval=evalerror) #Todo see whether it helps
gbm = xgb.train(params, dtrain, num_boost_round,  evals=watchlist, early_stopping_rounds=early_stopping_rounds, feval=rmspe_xg, verbose_eval=True)#not rmspe_xg

print("Validating")
#train_probs = gbm.predict(dtest)#Or X_valid
valid_probs = gbm.predict( xgb.DMatrix(X_valid))
indices = valid_probs < 0
valid_probs[indices] = 0
error = rmspe_metric2(X_valid_goal.values, valid_probs)#avec ou sans .values?#maybe 2 is not good
print('error{:.6f}'.format(error))

print("Make predictions on the test set")
Test_predict_xgb = gbm.predict(dtest)
indices = Test_predict_xgb < 0
Test_predict_xgb[indices] = 0
result_xgb = pd.DataFrame({'Id': test_index, 'Sales': np.expm1( Test_predict_xgb)}).set_index('Id')
result_xgb=result_xgb.sort_index()

time=datetime.datetime.now()
moment=time.isoformat('-')
moment=moment.replace(":","")
result_xgb.to_csv('submission_xgb_'+moment+'.csv')



create_feature_map(train_final.columns)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(12, 20))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb_'+moment+'.png', bbox_inches='tight', pad_inches=1)


print ('Starting Fitting')
rf = RandomForestRegressor(n_jobs = -1, n_estimators = 23)
rf.fit(train_final, train_goal)
train_predict=rf.predict(train_final)
print('fitting model')
print ("RPMSE_exp_norm", rmspe_metric(np.exp(train_predict)-1, np.exp(train_goal)-1))#goals are already in log version
print ("RPMSE_log", rmspe_metric(train_predict, train_goal))#goals are already in log version
# Load and process test data
# Make predictions
Test_predict_rf = rf.predict(test_final)

# Make Submission

result_rf = pd.DataFrame({'Id': test_index, 'Sales': np.exp( Test_predict_rf)-1}).set_index('Id')
result_rf = result_rf.sort_index()
result_rf.to_csv('submission_rf.csv')
print('submission 1 created')


gbr=GradientBoostingRegressor (loss='huber', max_leaf_nodes=None, subsample=0.5, n_estimators = 150)
gbr.fit(train_final, train_goal)
Test_predict_gbr = gbr.predict(test_final)
# Make Submission
result_gbr = pd.DataFrame({'Id': test_index, 'Sales': np.exp( Test_predict_gbr)-1}).set_index('Id')
result_gbr = result_gbr.sort_index()
result_gbr.to_csv('submission_gbr.csv')
print('submission 2 created')

result_mean=(result_rf['Sales']+result_gbr['Sales'])/2
result_mean= pd.DataFrame({'Id': test_index, 'Sales':result_mean['Sales']}).set_index('Id')
result_mean = result_mean.sort_index()
result_mean.to_csv('submission_mean.csv')
print('submission 3 created')

#I need to clean up this part, this did not work
#passed='\n'.join(passed)
#params='\n'.join(params)
#num_boost_round=str(num_boost_round)
#early_stopping_rounds=str(early_stopping_rounds)
#test_size=str(test_size)
#random_state=str(random_state)
#Histo_dat=moment+'passed\n'+passed+'params\n'+params+'num_boost_round\n'+num_boost_round+'early_stopping_rounds\n'+early_stopping_rounds+'test_size\n'+test_size+'random_state\n'+random_state+'error\n'+error
##string.to_csv('data_xgb'+moment+'.csv')
#
#with open('historical data.txt', 'a') as the_file:
#    the_file.write(Histo_dat+'\n')
#print('Historical Data  created')

'''OR
print(Histo_dat+'\n', file=f)

The alternative would be to use:

f = open(Histo_dat+'\n','w')
f.write(Histo_dat+'\n') 
f.close() '''

"""rmspe = sqrt(1/n_nonzero * s)
where, s = \sum_{i=1, y_i ! = 0}^{n} ((y_i - yhat_i)/y_i)^2 and n_nonzero is the number of nonzero entries."""

