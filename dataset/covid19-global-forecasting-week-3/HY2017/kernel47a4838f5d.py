# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import numpy as np 
import pandas as pd
from sklearn import metrics
#import lightgbm as lgb
#from xgboost import XGBRegressor
import time

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
#submission = pd.read_csv('submission.csv')

train = train.fillna('NaN_')
train['con_id'] = train['Province_State']+train['Country_Region']

test = test.fillna('NaN_')
test['con_id'] = test['Province_State']+test['Country_Region']

train = train.drop(['Province_State','Country_Region'],axis=1)
test = test.drop(['Province_State','Country_Region'],axis=1)





def s_com_model(para_o,day):
    
    [N0,ra,af0,aft,sep,na0] = para_o
    rc = 0.25
    #ra=1/7
    #rc=0.5
    if ra>= 0.6:
        ra = 0.6
    rb=1-ra-rc
    #af0 = 0.35
    
    
    Na = [na0]
    Nb = [rb*N0]
    Nc = [rc*N0]
    
    #day = 43
    for i in range(0,day):
        if i >= sep:
            af = af0*np.exp(-(i-sep)/aft)
        else:
            af = af0
        Na1 = Na[i]+ra*Nb[i]*(1+af)+ra*af*Nc[i]
        Nb1 = Nb[i]*(1+af)*rb+rb*af*Nc[i]
        Nc1 = Nb[i]*(1+af)*rc+Nc[i]*6/7+rc*af*Nc[i]
        
        Na.append(Na1)
        Nb.append(Nb1)
        Nc.append(Nc1)
    
    Na = np.array(Na)
    Nb = np.array(Nb)
    Nc = np.array(Nc)
    
    total_NO = pd.DataFrame()
    
    total_NO['Na'] = Na
    total_NO['Nb'] = Nb
    total_NO['Nc'] = Nc
    
    s = 7
    case0 = total_NO[:-s]
    case7 = total_NO[s:]
    
    R7_model = (case7.values-case0)/case0.values
    return total_NO, R7_model


def flow_fit(df):
    
    CC  = df.ConfirmedCases.dropna()
    CC0 = CC[:-7]
    CC7 = CC[7:]
    R7 = (CC7.values-CC0)/CC0.values
    day = CC.shape[0]
    na0 = CC[0]
    para0 = [4*na0,0.2,0.8,10,10,na0]
    
    daily_m, R7_m = s_com_model(para0,day-1)
    dd0 = np.abs(CC.diff()[1:].values)
    ddm = daily_m.Na.diff()[1:].values
    error_check0 = np.square(np.log(daily_m.Na.values+1)-np.log(CC+1)).sum()+10*np.square(np.log(R7_m.Na.values+1)-np.log(R7+1)).sum()+10*np.abs((dd0-ddm+1)/(dd0+ddm+1)).sum()

    para_ok = np.array(para0)
    
    
    for ii in range(0,3):
        if ii==0:
            fiti = 1000
            para00 = para0
        else:
            fiti = 300
            para00 = para_ok
        
        for i in range(0,fiti):
            aa = np.random.rand(5)*2
            aa = np.append(aa,1)
            parai = para00*aa
            
            daily_m_fit, R7_m_fit = s_com_model(parai,day-1)
            ddi = daily_m_fit.Na.diff()[1:].values
            
            error_check = np.square(np.log(daily_m_fit.Na.values+1)-np.log(CC+1)).sum()+10*np.square(np.log(R7_m_fit.Na.values+1)-np.log(R7+1)).sum()+10*np.abs((dd0-ddi+1)/(dd0+ddi+1)).sum()
                
            if error_check < error_check0:
                error_check0 = error_check
                para_ok = parai
                #print(str(ii)+','+str(i))
    
    print('Error : '+str(error_check0))
    daily_m_final, R7_m_final = s_com_model(para_ok,df.shape[0]-1)
    
    df['ConfirmedCases'][day:] = daily_m_final.Na.values[day:]

    
    a0 = df['ConfirmedCases'][day-1]
    df[day:].loc[df[day:].ConfirmedCases <= a0 , 'ConfirmedCases'] = a0

    aa = df['Fatalities'][day-5:day]    
    bb = df['ConfirmedCases'][day-5-7:day-7]
    
    ss = (aa.values/bb.values)
    ss[ss == np.inf] =0
    ss[np.isnan(ss)]=0
    death_rate = ss.mean()
    df['Fatalities'][day:] = df['ConfirmedCases'][day-7:-7]*death_rate

    b0 = df['Fatalities'][day-1]
    df[day:].loc[df[day:].Fatalities <= b0, 'Fatalities'] = b0    
    
    return df

time1 =time.time()
n=0
test_sub = pd.DataFrame()
for conid in train.con_id.unique().tolist():
    print(conid)
    train_temp = train.loc[train.con_id == conid].drop(['Id','con_id'],axis=1)
    test_temp = test.loc[test.con_id == conid].drop(['con_id'],axis=1)
    test_temp = pd.merge(test_temp, train_temp, how='left', on = ['Date'])
    
    test_temp = flow_fit(test_temp)
    
    if n==0:
        test_sub = test_temp.copy()
    else:
        test_sub = pd.concat([test_sub,test_temp])    
    n=n+1
print(n)    
time2 = time.time()
print(time2-time1)   
test_sub = test_sub.fillna(0)
test_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv',index=False) 