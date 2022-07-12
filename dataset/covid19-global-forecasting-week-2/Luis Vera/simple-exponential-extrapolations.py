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
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

from scipy.optimize import curve_fit

def exp(x, a, b, c):
    return a*np.exp(b*x) + c

def func(x, a, b, c, d):
    return a*(np.exp(b*x) + c)/(np.exp(b*x) + d)

# Confirmed Cases

df_filtered = df[df['ConfirmedCases'] >= 50]
places = df_filtered[['Country_Region', 'Province_State', 'Date']]
ind = places[['Country_Region', 'Province_State']].drop_duplicates().index
places = places.loc[ind]

def fit_cases(entry):
    if entry['Province_State'] is not np.nan:
        data = df_filtered[df_filtered['Province_State'] == entry['Province_State']]
    else:
        data = df_filtered[df_filtered['Province_State'].isnull()]        
        data = data[data['Country_Region'] == entry['Country_Region']]
    v = data['ConfirmedCases'].values
    day0 = data['Date'].iloc[0]
        
    x = np.arange(len(v))
    v_mean = np.mean(v)
    
    print(entry, day0)

    try:
        if len(v) < 10 or v[-1] < 500:       
            popt, pcov = curve_fit(exp, x, v, [1, 0.1, -1], bounds = [[0, 0, -np.inf], [np.inf, .1, np.inf]],
                       method = 'trf', maxfev = 1000000, xtol = 1e-7, ftol = 1e-7)
            print(popt)
        else:
            popt, pcov = curve_fit(func, x, v, [-v_mean,-.1,-.5,.5], bounds = [[-np.inf,-5,-2,0.001], [-2.,0,-.1,2.]],
                               method = 'trf', maxfev = 100000, xtol = 1e-6, ftol = 1e-6)

            print(popt)
            print(v[-1], func(len(v), *popt), func(len(v) + 1, *popt), func(len(v) + 2, *popt), 
                  func(len(v) + 3, *popt), func(len(v) + 4, *popt), func(len(v) + 5, *popt))
    except Exception as e: 
        print(e)
        return None, None, day0
    
    return popt, pcov, day0

names = []
values = []
days0 = [] 
for i in range(len(places)):
    entry = places.iloc[i]
    popt, pcov, day0 = fit_cases(entry)
    if popt is not None:  
        if entry['Province_State'] is not np.nan:
            name = entry['Province_State']
        else:
            name = entry['Country_Region']
        names.append(name)
        values.append(popt)
        days0.append(pd.to_datetime(day0))

submission_cases = []
for v in df_test.values:
    fid = v[0]
    location = v[1]
    if location is not np.nan:
        data = df[df['Province_State'] == location]
    else:
        location = v[2]
        data = df[df['Province_State'].isnull()]
        data = data[df['Country_Region'] == location]
    date = pd.to_datetime(v[3])
    pred0 = int(data['ConfirmedCases'].values[-1])
    try:
        pred = int(data[data['Date'] == v[3]]['ConfirmedCases'])
        print('READ:',location, date, pred)
    except:
        try:   
            loc_index = names.index(location)
            popt = values[loc_index]
            date0 = days0[loc_index]
            x = (date - date0).days
            if len(popt) == 4:
                pred = func(x, *popt)
            else:
                pred = exp(x, *popt)
            pred = int(np.clip(np.round(pred), pred0, np.inf))
        except:
            pred = pred0
    print(location, date, pred)
    submission_cases.append(pred)
    assert fid == len(submission_cases)


# Fatalities

df_filtered = df[df['Fatalities'] >= 5]

places = df_filtered[['Country_Region', 'Province_State', 'Date']]
ind = places[['Country_Region', 'Province_State']].drop_duplicates().index
places = places.loc[ind]


def fit_fatalities(entry):
    if entry['Province_State'] is not np.nan:
        data = df_filtered[df_filtered['Province_State'] == entry['Province_State']]
    else:
        data = df_filtered[df_filtered['Province_State'].isnull()]        
        data = data[data['Country_Region'] == entry['Country_Region']]
    v = data['Fatalities'].values
    day0 = data['Date'].iloc[0]
        
    x = np.arange(len(v))

    v_mean = np.mean(v)    
    print(entry, day0)

    try:
        if len(v) < 10 or v[-1] < 50:
            popt, pcov = curve_fit(exp, x, v, [1, 0.1, -1], bounds = [[0, 0, -np.inf], [np.inf, .1, np.inf]],
                       method = 'trf', maxfev = 1000000, xtol = 1e-7, ftol = 1e-7)
            print(popt)
        else:
            popt, pcov = curve_fit(func, x, v, [-v_mean,-.1,-.5,.5], bounds = [[-np.inf,-5,-2,0.001], [-2.,0,-.1,2.]],
                               method = 'trf', maxfev = 100000, xtol = 1e-6, ftol = 1e-6)

            print(popt)
            print(v[-1], func(len(v), *popt), func(len(v) + 1, *popt), func(len(v) + 2, *popt), 
                  func(len(v) + 3, *popt), func(len(v) + 4, *popt), func(len(v) + 5, *popt))
    except Exception as e: 
        print(e)
        return None, None, day0
    
    return popt, pcov, day0

names = []
values = []
days0 = [] 
for i in range(len(places)):
    entry = places.iloc[i]
    popt, pcov, day0 = fit_fatalities(entry)
    if popt is not None:  
        if entry['Province_State'] is not np.nan:
            name = entry['Province_State']
        else:
            name = entry['Country_Region']
        names.append(name)
        values.append(popt)
        days0.append(pd.to_datetime(day0))

submission_fatalities = []
for v in df_test.values:
    fid = v[0]
    location = v[1]
    if location is not np.nan:
        data = df[df['Province_State'] == location]
    else:
        location = v[2]
        data = df[df['Province_State'].isnull()]
        data = data[data['Country_Region'] == location]
    date = pd.to_datetime(v[3])
    pred0 = int(data['Fatalities'].values[-1])
    try:
        pred = int(data[data['Date'] == v[3]]['Fatalities'])
        print('READ:',location, date, pred)
    except:
        try:   
            loc_index = names.index(location)
            popt = values[loc_index]
            date0 = days0[loc_index]
            x = (date - date0).days
            if len(popt) == 4:
                pred = func(x, *popt)
            else:
                pred = exp(x, *popt)
            pred = int(np.clip(np.round(pred), pred0, np.inf))
        except:
            pred = pred0
    print(location, date, pred)      
    submission_fatalities.append(pred)
    assert fid == len(submission_fatalities)
    
# Create submission file
with open('submission.csv', 'w') as myfile:
    myfile.write('ForecastId,ConfirmedCases,Fatalities\n')
    
with open('submission.csv', "a") as myfile:
    for i in range(len(submission_fatalities)):
        myfile.write(str(i+1) + ',' + str(submission_cases[i]) + ',' + str(submission_fatalities[i]) + '\n')