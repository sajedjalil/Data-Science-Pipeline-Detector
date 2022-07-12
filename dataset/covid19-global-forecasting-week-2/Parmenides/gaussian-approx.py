# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error as mse

def gaussian(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2./(2.*sigma**2.))

def GetRegion(regional,f="Fatalities"):
    y=regional[f].values
    y=np.diff(y,prepend=[0])
    x=regional["Date"].values.astype(np.datetime64)
    x0=x[0].copy()
    dx=x-x0
    max_y=np.max(y)
    y=y/np.max(y)
    return dx,y,x0,max_y

def SetRegion(regional,x0):
    x=regional["Date"].values.astype(np.datetime64)
    f_id=regional["ForecastId"].values.astype(np.int)
    dx=x-x0
    return dx,f_id

def doFit_fat(region):
    dx,y,x0,max_y=GetRegion(region,f="Fatalities")
    n=dx.shape[0]
    mu=np.sum(dx.astype(np.float)*y)/n
    sigma=np.sum(y*np.square(dx.astype(np.float)-mu))/n
    popt,pcov=curve_fit(gaussian,dx,y,p0=[1.,mu,sigma],maxfev=1000000)
    return popt,pcov,x0,max_y

def doFit_CC(region):
    dx,y,x0,max_y=GetRegion(region,f="ConfirmedCases")
    n=dx.shape[0]
    mu=np.sum(dx.astype(np.float)*y)/n
    sigma=np.sum(y*np.square(dx.astype(np.float)-mu))/n
    popt,pcov=curve_fit(gaussian,dx,y,p0=[1.,mu,sigma],maxfev=1000000)
    return popt,pcov,x0,max_y


def ShiftDistr(region,distr,typef):
    dx,y,x0,max_y=GetRegion(region,f=typef)
    shift=dx.shape[0]
    min_mse=1000000.
    for i in distr:
        for mu in np.arange(-shift,shift):
            curve=gaussian(dx.astype(np.int),i[0],i[1]+mu,i[2])
            m=mse(y,curve)
            if m<min_mse:
                min_mse=m
                best_popt=i
                best_popt[1]=i[1]+mu
    
    return best_popt,x0,max_y

 

df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv").fillna(0)
countries=df["Country_Region"].unique()

list_par=[]
for i in countries: 

    tmp=df.loc[df["Country_Region"]==i]
    states=tmp["Province_State"].unique()

    if states.size<=1:
        if tmp["ConfirmedCases"].max()>10000:
            #print(i)
            popt,pcov,x0,_=doFit_CC(tmp)
            list_par.append(popt) 

    else:
        for k in states:
            tmpk=tmp.loc[tmp["Province_State"]==k]
            if tmpk["ConfirmedCases"].max()>10000:
                popt,pcov,x0,_=doFit_CC(tmpk)
                list_par.append(popt)

print(list_par)

list_fat=[]
for i in countries: 

    tmp=df.loc[df["Country_Region"]==i]
    states=tmp["Province_State"].unique()

    if states.size<=1:
        if tmp["Fatalities"].max()>1000:
            #print(i)
            popt,pcov,x0,_=doFit_fat(tmp)
            list_fat.append(popt) 

    else:
        for k in states:
            tmpk=tmp.loc[tmp["Province_State"]==k]
            if tmpk["Fatalities"].max()>1000:
                popt,pcov,x0,_=doFit_fat(tmpk)
                list_fat.append(popt)

print(list_fat)

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv").fillna(0)
fout=open("submission.csv","w")
print("ForecastId,ConfirmedCases,Fatalities",file=fout)

for i in countries: 

    tmp=df.loc[df["Country_Region"]==i]
    states=tmp["Province_State"].unique()

    if states.size<=1:
        print(i,end=' ')
        scc=tmp.loc[tmp["Date"].values.astype(np.datetime64)==datetime.date(2020,3,18)]["ConfirmedCases"].values
        sfat=tmp.loc[tmp["Date"].values.astype(np.datetime64)==datetime.date(2020,3,18)]["Fatalities"].values

        if tmp["ConfirmedCases"].max()>0:
            test_tmp=test.loc[test["Country_Region"]==i]            
            popt,x0,max_y=ShiftDistr(tmp,list_par,typef="ConfirmedCases")            
            test_dx,test_id=SetRegion(test_tmp,x0)
            curve1=max_y*gaussian(test_dx.astype(np.int),*popt)
            print(popt,end=' ')            
        else:
            test_dx,test_id=SetRegion(test_tmp,x0)
            curve1=np.zeros(test_dx.shape[0])
            print('[0,0,0]',end=' ') 

        if tmp["Fatalities"].max()>0:
            popt,x0,max_y=ShiftDistr(tmp,list_fat,typef="Fatalities")
            print(popt)
            curve2=max_y*gaussian(test_dx.astype(np.int),*popt)
        else:
            #test_dx,test_id=SetRegion(test_tmp,x0)
            curve2=np.zeros(test_dx.shape[0])
            print('[0,0,0]')
            
        for i_id,i_c1,i_c2 in zip(test_id,scc+np.cumsum(curve1),sfat+np.cumsum(curve2)):
            print("{:d},{:d},{:d}".format(i_id,np.round(i_c1).astype(np.int),np.round(i_c2).astype(np.int)),file=fout)

    else:
        for k in states:
            print(i,",",k,end=' ')
            tmpk=tmp.loc[tmp["Province_State"]==k]
            test_tmp=test.loc[test["Country_Region"]==i]
            test_tmpk=test_tmp.loc[test_tmp["Province_State"]==k]
            scc=tmpk.loc[tmpk["Date"].values.astype(np.datetime64)==datetime.date(2020,3,18)]["ConfirmedCases"].values
            sfat=tmpk.loc[tmpk["Date"].values.astype(np.datetime64)==datetime.date(2020,3,18)]["Fatalities"].values

            
            if tmpk["ConfirmedCases"].max()>0:                                
                popt,x0,max_y=ShiftDistr(tmpk,list_par,typef="ConfirmedCases")
                print(popt,end=' ')
                test_dx,test_id=SetRegion(test_tmpk,x0)
                curve1=max_y*gaussian(test_dx.astype(np.int),*popt)
            else:
                test_dx,test_id=SetRegion(test_tmpk,x0)
                curve1=np.zeros(test_dx.shape[0])
                print('[0,0,0]',end=' ')

            if tmpk["Fatalities"].max()>0:                
                popt,x0,max_y=ShiftDistr(tmpk,list_fat,typef="Fatalities")
                print(popt)
                curve2=max_y*gaussian(test_dx.astype(np.int),*popt)
            else:
                #test_dx,test_id=SetRegion(test_tmpk,x0)
                curve2=np.zeros(test_dx.shape[0])
                print('[0,0,0]')
                
            for i_id,i_c1,i_c2 in zip(test_id,scc+np.cumsum(curve1),sfat+np.cumsum(curve2)):
                print("{:d},{:d},{:d}".format(i_id,np.round(i_c1).astype(np.int),np.round(i_c2).astype(np.int)),file=fout)

fout.close()
