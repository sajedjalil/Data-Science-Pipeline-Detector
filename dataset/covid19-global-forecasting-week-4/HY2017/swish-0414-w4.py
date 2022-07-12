# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import numpy as np 
import pandas as pd
from sklearn import metrics
#import lightgbm as lgb
#from xgboost import XGBRegressor
import time

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
#submission = pd.read_csv('submission.csv')

train = train.fillna('NaN_')
train['con_id'] = train['Province_State']+train['Country_Region']

test = test.fillna('NaN_')
test['con_id'] = test['Province_State']+test['Country_Region']

train = train.drop(['Province_State','Country_Region'],axis=1)
test = test.drop(['Province_State','Country_Region'],axis=1)





def s_com_model(para_o,day, aft_check):
    
    [N0,ra,af0,aft,sep,na0] = para_o
    rc = 0.25
    #ra=1/7
    #rc=0.5
    if ra>= 0.6:
        ra = 0.6
    if ra<=0.05:
        ra=0.05
    rb=1-ra-rc
    #af0 = 0.35
    
    if aft_check==1 and aft<10:
        aft=10
    if aft_check==2 and aft<15:
        aft=15

    
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
    
    R7_model = (case7.values-case0)/(case0.values+1)
    return total_NO, R7_model


def flow_fit(df):
    df0=df.copy()
    aft_check = 0
    if df.ConfirmedCases.diff().max() > 500:
        aft_check = 1
    if df.ConfirmedCases.diff().max() > 1000:
        aft_check = 2        
    CC  = df.ConfirmedCases.dropna()
    CC0 = CC[:-7]
    CC7 = CC[7:]
    R7 = (CC7.values-CC0)/(CC0.values+1)
    day = CC.shape[0]
    na0 = CC[0]
    para0 = [4*na0+1,0.2,0.8,20,20,na0]
    
    daily_m, R7_m = s_com_model(para0,day-1,aft_check)
    dd0 = np.abs(CC.diff()[1:].values)
    ddm = daily_m.Na.diff()[1:].values
    
    d03 = (CC[5:].values-CC[:-5].values)
    dm3 = (daily_m.Na[5:].values-daily_m.Na[:-5].values)
    
    dd03 = d03[1:]-d03[:-1]
    ddm3 = dm3[1:]-dm3[:-1]
    
    error_check0 = 10*np.square(np.log(daily_m.Na.values+1)-np.log(CC+1)).sum()+10*np.square(np.log(R7_m.Na.values+1)-np.log(R7+1)).sum()+0.5*np.abs((dd0-ddm+1)/(dd0+ddm+1)).sum()+5.0*np.abs((d03-dm3+1)/(d03+dm3+1)).sum()+1.0*np.abs((dd03-ddm3+1)/(dd03+ddm3+1)).sum()

    para_ok = np.array(para0)
    
    do_again = 0
    do_check = 0
    cs=1
    while (do_again==0):
        for ii in range(0,5):
            if ii==0 and cs==1:
                fiti = 5000
                para00 = para0
            else:
                fiti = 300
                para00 = para_ok
            
            for i in range(0,fiti):
                aa = np.random.rand(5)*2
                aa = np.append(aa,1)
                parai = para00*aa
                
                daily_m_fit, R7_m_fit = s_com_model(parai,day-1,aft_check)
                ddi = daily_m_fit.Na.diff()[1:].values
                
                di3 = daily_m_fit.Na[5:].values - daily_m_fit.Na[:-5].values
                ddi3 = di3[1:]-di3[:-1]
                
                error_check = 10*np.square(np.log(daily_m_fit.Na.values+1)-np.log(CC+1)).sum()+10*np.square(np.log(R7_m_fit.Na.values+1)-np.log(R7+1)).sum()+cs*0.5*np.abs((dd0-ddi+1)/(dd0+ddi+1)).sum()+5.0*np.abs((d03-di3+1)/(d03+di3+1)).sum()+cs*1.0*np.abs((dd03-ddi3+1)/(dd03+ddi3+1)).sum()
        
                if error_check < error_check0:
                    error_check0 = error_check
                    para_ok = parai
                    #print(str(ii)+','+str(i))
                    e0 = 10*np.square(np.log(daily_m_fit.Na.values+1)-np.log(CC+1)).sum()
                    e1 = 10*np.square(np.log(R7_m_fit.Na.values+1)-np.log(R7+1)).sum()
                    e2 = 0.5*np.abs((dd0-ddi+1)/(dd0+ddi+1)).sum()
                    e3 = 5.0*np.abs((d03-di3+1)/(d03+di3+1)).sum()
                    e4 = 1.0*np.abs((dd03-ddi3+1)/(dd03+ddi3+1)).sum()
        print('Errors : '+str(error_check0)+' e0= '+str(e0)+' e1= '+str(e1)+' e2= '+str(e2)+' e3= '+str(e3)+' e4= '+str(e4))
        
        if e0+e1+e3 > 0.5:
            do_again = do_again + do_check
            do_check = 1
            #para0 = para_ok
            cs=0
            error_check0 = e0+e1+e3
        else:
            do_again = 1
            
    #print('Errors : '+str(error_check0)+' e0= '+str(e0)+' e1= '+str(e1)+' e2= '+str(e2)+' e3= '+str(e3))
    daily_m_final, R7_m_final = s_com_model(para_ok,df.shape[0]-1,aft_check)
    
    df['ConfirmedCases'][day:] = daily_m_final.Na.values[day:]
    
    a0 = df['ConfirmedCases'][day-1]
    df[day:].loc[df[day:].ConfirmedCases <= a0 , 'ConfirmedCases'] = a0

    
    aa = df['Fatalities'][day-3:day]    
    bb = df['ConfirmedCases'][day-3-7:day-7]
    """
    aa = df['Fatalities'][day-5:day] 
    bb = df['ConfirmedCases'][day-5:day]
    """
    ss = (aa.values/bb.values)
    ss[ss == np.inf] =0
    ss[np.isnan(ss)]=0
    death_rate = ss.mean()
    if death_rate > 0.15:
        death_rate = 0.15
    df['Fatalities'][day:] = df['ConfirmedCases'][day-7:-7]*death_rate

    b0 = df['Fatalities'][day-1]
    df[day:].loc[df[day:].Fatalities <= b0, 'Fatalities'] = b0    
    if e0+e1 > 0.3:
        print('***Turn to logistic***')
        df = logistic(df0)    
        
    return df

def logistic_curve(para,nx):
    [A1, A2, x0, p, x00] = para
    if x0<10:
        x0=10
    if p<1.1:
        p=1.1
    x = np.arange(nx)+1
    y = A2+(A1-A2)/(1+((x+x00)/(x0+x00))**p)
    return y

def logistic(df):

    CC  = df.ConfirmedCases.dropna()
    day = CC.shape[0]
    
    n2 = CC.max()
    n1 = CC.min()
    para0 = np.array([n1/4,10*n2,day+10,3,20]) #[A1,A2,x0,p,x00]
    yy0 = logistic_curve(para0,day)
    error0 = np.square(np.log(yy0+1)-np.log(CC+1)).sum()
    
    
    for i in range(0,1000):
        aa = np.random.rand(5)*2
        parai = para0*aa
        yyi = logistic_curve(parai,day)
        errori = np.square(np.log(yyi+1)-np.log(CC+1)).sum()
        
        if errori < error0:
            para_ok = parai
            error0 = errori
            
        if i > 500:
            para0 = para_ok
    print(para_ok)
    print('log_Errors : '+str(error0)) 
    
    yy = logistic_curve(para_ok, df.shape[0])
    
    df['ConfirmedCases'][day:] = yy[day:]
    
    a0 = df['ConfirmedCases'][day-1]
    df[day:].loc[df[day:].ConfirmedCases <= a0 , 'ConfirmedCases'] = a0

    
    aa = df['Fatalities'][day-3:day]    
    bb = df['ConfirmedCases'][day-3-7:day-7]
    """
    aa = df['Fatalities'][day-5:day] 
    bb = df['ConfirmedCases'][day-5:day]
    """
    ss = (aa.values/bb.values)
    ss[ss == np.inf] =0
    ss[np.isnan(ss)]=0
    death_rate = ss.mean()
    if death_rate > 0.15:
        death_rate = 0.15
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
    
    if test_temp.ConfirmedCases.diff().max() > 100:
        test_temp = flow_fit(test_temp)
    else:
        test_temp = logistic(test_temp)
        
    if n==0:
        test_sub = test_temp.copy()
    else:
        test_sub = pd.concat([test_sub,test_temp])    
    n=n+1
print(n)    
time2 = time.time()
print(time2-time1)   
#test_sub = test_sub.fillna(0)
test_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv',index=False) 