# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import os

def FitReg(train_conf, train_fat, local_pred):
    pred_data = pd.DataFrame(data={'Fatality' : train_fat.flatten(),
                                   'ConfirmedCase' : train_conf.flatten()})
    lm_model = glm('Fatality~0+ConfirmedCase', 
                   data=pred_data, 
                   family = sm.families.Gaussian()).fit()
    
    ## Prediction ##
    local_pred_res = lm_model.predict(pd.DataFrame(data={'ConfirmedCase' : local_pred.flatten()}))
    return np.round(local_pred_res.values)
    
    
# Errors are also returned as 0 predictions
def FitArima(train_data=None):
    pred_res = np.zeros([284, 43])
    err_count = 0
    for i in range(train_data.shape[0]):
        print("Processing State Idx: {}".format(i))
        if np.count_nonzero(train_data[i,]) == 0:
            pred_res[i,] = np.repeat(0.0, 43)
        else:
            try:
                model_fit = ARIMA(train_data[i,].astype(np.float32), order=(2,2,0)).fit(disp=False)
                pred_res[i,] = model_fit.forecast(43)[0]
            except:
                err_count+=1
                pass
    print("Total Error {}".format(err_count))
    return(np.round(pred_res))

    

def LoadData():
    train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
    train_ft = train['Country/Region'].astype(str) + '_' + train['Lat'].astype(str) + '_' + train['Long'].astype(str)
    train_ft = train_ft.drop_duplicates()
    reg_count = train_ft.values
    
    all_ft_conf = []
    all_ft_fat = []
    for r in reg_count:
        local_idx = train['Country/Region'].astype(str) + '_' + train['Lat'].astype(str) + '_' + train['Long'].astype(str) == r
        local_ft = train.iloc[local_idx.values]
        all_ft_conf.append(local_ft['ConfirmedCases'].values)
        all_ft_fat.append(local_ft['Fatalities'].values)
    all_ft_conf_ft = np.stack(all_ft_conf, axis=0)
    all_ft_fat_ft = np.stack(all_ft_fat, axis=0)
    
    local_train_conf = all_ft_conf_ft[::,0:50]
    local_valid_conf = all_ft_conf_ft[::,50:62]
    local_train_fat = all_ft_fat_ft[::,0:50]
    local_valid_fat = all_ft_fat_ft[::,50:62]

    return local_train_conf, local_valid_conf, local_train_fat, local_valid_fat

## Main ##
train_data_conf, test_data_conf, train_data_fat, test_data_fat = LoadData()

# ARIMA Predictions
arima_pred = FitArima(train_data_conf)

#arima_err = np.median(np.mean(np.abs(test_data_conf - arima_pred), axis=1))
#plt.plot(np.abs(test_data[2,] - arima_pred[2,]))

# NB Regression #
lm_pred = FitReg(train_data_conf, train_data_fat, arima_pred).reshape([284, 43])

# Combine
arima_pred_ft = arima_pred.flatten()
lm_pred_ft = lm_pred.flatten()
final_pred = pd.DataFrame(data = {'ForecastId' : range(1, arima_pred_ft.shape[0]+1),
                                  'ConfirmedCases' : arima_pred_ft.astype(np.uint32),
                                  'Fatalities' : lm_pred_ft.astype(np.uint32)})
final_pred.to_csv('submission.csv', index=False)