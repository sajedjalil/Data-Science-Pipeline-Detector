# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import scipy.optimize as opt
import os
import datetime as dt

# In[95]:


print(os.listdir("./"))


# In[96]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
#train.head()

EMPTY_VAL = "EMPTY_VAL"

def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state

#train_ = train[train["ConfirmedCases"] > 0]
train_ = train

train_['Province_State'].fillna(EMPTY_VAL, inplace=True)
train_['Province_State'] = train_.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
test['Province_State'].fillna(EMPTY_VAL, inplace=True)
test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

def f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b


def logistic(xs, L, k, x_0):
    result = []
    for x in xs:
        xp = k*(x-x_0)
        if xp >= 0:
            result.append(L / ( 1. + np.exp(-xp) ) )
        else:
            result.append(L * np.exp(xp) / ( 1. + np.exp(xp) ) )
    return result


unique = pd.DataFrame(train_.groupby(['Country_Region', 'Province_State'],as_index=False).count())

def date_day_diff(d1, d2):
    delta = dt.datetime.strptime(d1, "%Y-%m-%d") - dt.datetime.strptime(d2, "%Y-%m-%d")
    return delta.days

log_regions = []

for index, region in unique.iterrows():
    st = region['Province_State']
    co = region['Country_Region']
    #print(co,st)
    rdata = train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]

    t = rdata['Date'].values
    t = [float(date_day_diff(d, t[0])) for d in t]
    y = rdata['ConfirmedCases'].values
    y_ = rdata['Fatalities'].values

    try:
        p0 = [max(y), 0.0, max(t)]
        p0_ = [max(y_), 0.0, max(t)]
        try:
            popt, pcov = opt.curve_fit(logistic, t, y, p0, maxfev=10000)
        except:
            popt, pcov = opt.curve_fit(f, t, y,method="trf", maxfev=10000)
        try:
            popt_, pcov_ = opt.curve_fit(f, t, y_,method="trf", maxfev=10000)
        except:
            popt_, pcov_ = opt.curve_fit(logistic, t, y_, p0_, maxfev=10000)
        log_regions.append((co,st,popt,popt_))
        #print("1st",co,st,popt,popt_)
    except:
        p0 = [max(y), 0.0, max(t)]
        p0_ = [max(y_), 0.0, max(t)]
        popt, pcov = opt.curve_fit(f, t, y,method="trf", maxfev=10000)
        popt_, pcov_ = opt.curve_fit(f, t, y_,method="trf", maxfev=10000)
        log_regions.append((co,st,popt,popt_))
        #print("2nd",co,st,popt,popt_)


print("All done!")

log_regions = pd.DataFrame(log_regions)
log_regions.head()
log_regions.columns = ['Country_Region','Province_State','ConfirmedCases','Fatalities']
#log_regions.head(10)

log_regions.loc[log_regions['ConfirmedCases'].str[1] > 0.42, 'ConfirmedCases'] = 0.42
log_regions['ConfirmedCases'].str[1].quantile([.1, .25, .5, .75, .95, .99])
sublist = []

for index, rt in log_regions.iterrows():
    st = rt['Province_State']
    co = rt['Country_Region']
    popt = list(rt['ConfirmedCases'])
    popt_ = list(rt['Fatalities'])
    #print(co,st,popt,popt_)
    rtest = test[(test['Province_State']==st) & (test['Country_Region']==co)]
    for index, rt in rtest.iterrows():
        try:
            tdate = rt['Date']
            ca = logistic([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt)
            try:
                fa = logistic([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt_)
            except:
                fa = f([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt_)
            sublist.append((rt['ForecastId'], int(ca[0]), int(fa[0])))
            #print(rt['ForecastId'], int(ca[0]),int(fa[0]),co,st,tdate)
        except:
            tdate = rt['Date']
            ca = f([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt)
            fa = f([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt_)
            sublist.append((rt['ForecastId'], int(ca[0]), int(fa[0])))
            #print(rt['ForecastId'], int(ca[0]),int(fa[0]),co,st,tdate)

print("All done!")
submission = pd.DataFrame(sublist)
submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
submission.to_csv('./submission.csv', index = False)
print("submission ready!")