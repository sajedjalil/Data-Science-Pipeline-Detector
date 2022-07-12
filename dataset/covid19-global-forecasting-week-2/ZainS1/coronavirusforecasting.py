# Libraries
import numpy as np
import pandas as pd
from fbprophet import Prophet
import os
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

for dirname, _, filenames in os.walk('/kaggle/input'):
    
    for filename in filenames:
        
        print(os.path.join(dirname, filename))
        
# Read
train = pd.read_csv(dirname + '/train.csv')
test = pd.read_csv(dirname + '/test.csv')

# Initialize
casestrainingdata = {}
deathstrainingdata = {}

preds = {}
logpreds = {}
future = pd.DataFrame(pd.date_range(min(test['Date']), max(test['Date'])), columns = ['ds'])

# Prepare - training
for nm, grp in train.groupby('Country_Region'):
    
    casestrainingdata[nm] = grp[['Date', 'ConfirmedCases']].rename({'Date': 'ds', 'ConfirmedCases': 'y'}, axis = 1)
    
for nm, grp in train.groupby('Country_Region'):
    
    deathstrainingdata[nm] = grp[['Date', 'Fatalities']].rename({'Date': 'ds', 'Fatalities': 'y'}, axis = 1)
    
# Prepare - testing
for nm, grp in test.groupby('Country_Region'):
    
    preds[nm] = grp['ForecastId']
    logpreds[nm] = grp['ForecastId']
    
# Function
def prophecy(location):
    
    # Get
    cases = casestrainingdata[location]
    deaths = deathstrainingdata[location]
    
    # Train
    m = Prophet()
    m.fit(cases)
    
    # Predict
    casepreds = abs(round(m.predict(future)['yhat']))
    preds[location]['ConfirmedCases'] = casepreds
    
    # Train
    m = Prophet()
    m.fit(deaths)    
    
    # Predict
    deathpreds = abs(round(m.predict(future)['yhat']))
    preds[location]['Fatalities'] = deathpreds

    # Get
    cases['floor'] = [0] * len(cases)
    cases['cap'] = [max(train.loc[train['Country_Region'] == nm]['ConfirmedCases'])] * len(cases)
        
    # Train
    m = Prophet(growth = 'logistic')
    m.fit(cases)
    
    # Predict
    fut = m.make_future_dataframe(len(future))
    fut['floor'] = 0
    fut['cap'] = [max(train.loc[train['Country_Region'] == nm]['ConfirmedCases'])] * len(fut)
    
    logpreds[location]['ConfirmedCases'] = m.predict(fut)['yhat']

# Execute
for k, v in preds.items():
    
    prophecy(k)
    
# Write
submission = pd.DataFrame()

for k, v in preds.items():
    
    submission = submission.append(v)
    
submission.to_csv(dirname + 'kaggle/output/submission.csv', index = False)


    
