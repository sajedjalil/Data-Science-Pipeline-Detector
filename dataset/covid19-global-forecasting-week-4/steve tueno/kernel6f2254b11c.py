# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet





def forecast(inner_fit_df, dates):

    inner_fit_df.columns = ['ds', 'y']

    modeler = Prophet()
    modeler.fit(inner_fit_df)


    forecast = modeler.predict(dates)#(pd.concat([inner_fit_df[['ds']], future], axis=0))

    final_df = forecast[['yhat', 'yhat_lower', 'yhat_upper', 'ds']]

    print("forecasts on historical data ready...")
    #print(final_df.tail())

    return final_df




idf = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv", sep=',')[['Country_Region', 'Province_State','Date', 'ConfirmedCases','Fatalities']]\
                .astype({'ConfirmedCases': 'float32'}).astype({'Fatalities': 'float32'})
idf["Province_State"].fillna("RAS", inplace = True)


testdf = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv", sep=',')[['ForecastId', 'Date', 'Province_State', 'Country_Region']]\
        .astype({'ForecastId': 'int32'})
testdf["Province_State"].fillna("RAS", inplace = True)
        

idf['Date'] =  pd.to_datetime(idf['Date'],
                             format='%Y-%m-%d')

testdf['Date'] =  pd.to_datetime(testdf['Date'],
                             format='%Y-%m-%d')

result_df_confirmed = pd.DataFrame(columns = ['Country_Region', 'Province_State', 'Date', 'ConfirmedCases'])
result_df_fatality = pd.DataFrame(columns = ['Country_Region', 'Province_State', 'Date', 'Fatalities'])

dates = pd.DataFrame(columns = ['ds'])

dates['ds'] = testdf['Date'].unique()

for key, country_idf in idf.groupby(['Country_Region', 'Province_State']) :
    
    country_idf.columns = ['Country_Region', 'Province_State', 'ds','ConfirmedCases','Fatalities']
    
    forecasted_confirmed_idf = forecast(country_idf[['ds', 'ConfirmedCases']], dates)[['yhat', 'ds']]
    forecasted_confirmed_idf.columns = ['ConfirmedCases',  'Date']    
    forecasted_confirmed_idf['Country_Region'] = key[0]    
    forecasted_confirmed_idf['Province_State'] = key[1]    
    result_df_confirmed = result_df_confirmed.append(forecasted_confirmed_idf[['Country_Region', 'Province_State', 'Date', 'ConfirmedCases']], ignore_index=True)
    
    
    forecasted_fatality_idf = forecast(country_idf[['ds', 'Fatalities']], dates)[['yhat', 'ds']]
    forecasted_fatality_idf.columns = ['Fatalities',  'Date']
    forecasted_fatality_idf['Country_Region'] = key[0]    
    forecasted_fatality_idf['Province_State'] = key[1]
    result_df_fatality = result_df_fatality.append(forecasted_fatality_idf[['Country_Region', 'Province_State', 'Date', 'Fatalities']], ignore_index=True)
    

forecasted_idf = pd.merge(result_df_confirmed, result_df_fatality,  on=['Country_Region', 'Province_State', 'Date'], how='inner').reset_index()
result_df = pd.merge(forecasted_idf, testdf, on=['Country_Region', 'Province_State', 'Date'], how='inner').reset_index()[['ForecastId', 'ConfirmedCases', 'Fatalities']]
    

result_df.to_csv("submission.csv", sep=',', index=False)

