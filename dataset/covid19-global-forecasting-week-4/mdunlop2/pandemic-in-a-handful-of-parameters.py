'''
Pandemic in a handful of Parameters

SIR model, global parameters optimised using DLIB LIPO optimisation (lipschitz constant optimisation)

r: "r_0", total number of individuals an infected person will infect while sick when S=1
gamma: Inverse of the averageb duration of infection, days^(-1)
mort: Mortality rate, proportion of those infected who die
b1: Influence of population density on the transmission rate
b3: Global quarantine effect - remove susceptible population as a function of time since 21 January 2019

TODO:
Properly implement quarantine effect, economic effect on mortality rate, geographic region constants

'''

# Define column names in case they get changed in a future week
Province_State = "Province_State"
Country_Region = "Country_Region"
Population = "Population"
Density = "Density"
Area = "Area"
ConfirmedCases = "ConfirmedCases"
Fatalities = "Fatalities"
Date= "Date"

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)
import matplotlib.pyplot as plt
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow_probability as tfp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

# read in custom data
country_data = pd.read_csv("../input/covid19-province-pop-area/all_pop_area.csv")

# fill NaN with "null" so that we can use Country_Region, Province_State as key
country_data[[Province_State]] = country_data[[Province_State]].fillna(value="null")
country_data = country_data[[Country_Region, Province_State, Population, Area]]

# get the cases data
# fill NaN with "null" so that we can use Country_Region, Province_State as key
train[[Province_State]] = train[[Province_State]].fillna(value="null")

# get the test data
# fill NaN with "null" so that we can use Country_Region, Province_State as key
test[[Province_State]] = test[[Province_State]].fillna(value="null")


# group the test data by country / region
test_group = test.groupby([Country_Region, Province_State]).agg({"Date": "min"}).reset_index()
# we get the date as this will allow us to join the predictions
# back to test using .loc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Perform Predictions and Submit!                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def forecast(I0,
             F0,
             Population,
             std_density,
             beta0,
             gamma,
             b1,
             b2,
             b3,
             current_t,
             mort,
             forecast_len):
    '''
    Returns I, R at steps of the forecast
    '''
    times = np.arange(forecast_len+1)
    # get estimated recovered proportion
    # print("F0: {} I0: {}".format(F0, I0))
    R0 = (F0/mort)/Population
    I0 = I0/Population
    S0 = 1 - I0 -R0
    y_init = tf.stack((tf.constant(S0), tf.constant(I0), tf.constant(R0)))
    #print("Solving Input: S0: {} I0: {} R0: {}".format(S0, I0, R0))
    #print("forecast length: {}".format(forecast_len))
    #print("Times to solve: {}".format(times))
    def ode_fn(t, y):
        '''
        Requires many global varaibles:
            t_tensor - time since wuhan at start of simulation (Days)
            Beta0    - Number of infections from one infected person per day
            std_density_tensor - Normalised Population Density of Region
            gamma    - inverse of infection duration
        '''
        quarantined = -tf.math.exp(-(current_t+t)*b3)*(-b3)*y[0]
        # current time since Wuhan
        new_infected = tf.math.maximum(beta0 +b1*std_density, 0) * y[0] * y[1]
        S = -tf.math.maximum(tf.math.minimum(new_infected+quarantined, 0.9*y[0]), -0.9*y[0]) # ensure we stay within bounds
        # at most 90% of S get infected/removed in 1 day (keeps model from exploding)
        # at most 90% of infected/recovered move to susceptible
        R = gamma * y[1]
        I = tf.maximum(tf.math.minimum(new_infected - R, 0.9*y[0]), -0.9*y[1])
        return tf.stack((S, I, R))
    results = tfp.math.ode.DormandPrince(rtol = 0.0001,
                                         atol = 0.00001).solve(ode_fn, 0, y_init,
                                   solution_times=times)
    #print("Results Shape: {}".format(tf.shape(results.states)))
    res_np = results.states.numpy()    
    return res_np*Population

def extract_ts(full_pos, target, country, province, days, t_0):
    c   = full_pos[Country_Region] == country
    p   = full_pos[Province_State] == province
    d1  = ((pd.to_datetime(full_pos["Date"])-pd.to_datetime(full_pos["Date"]).min())/pd.offsets.Day(1)).values > t_0
    d2  = ((pd.to_datetime(full_pos["Date"])-pd.to_datetime(full_pos["Date"]).min())/pd.offsets.Day(1)).values <= t_0+days
    return full_pos[c&p&d1&d2][target].values

def second_order_fit(pred, actual, cut=0.8):
    '''
    Attempt to predict the second order change using a time series,
    If r-sq is too low, attempt to predict the first order time series
    '''
    g_2 = np.diff(pred, 2)
    g_1 = np.diff(pred, 1)
    f_2 = np.diff(actual, 2)
    f_1 = np.diff(actual, 1)
    g_2_0 = g_2[:-1]
    g_1_0 = g_1[:-1]
    g_2_1 = g_2[1:]
    g_1_1 = g_1[1:]
    f_2_0 = f_2[:-1]
    f_1_0 = f_1[:-1]
    y_2 = f_2[1:]
    y_1 = f_1[1:]
    
    x = np.vstack((f_2_0, g_2_0, g_2_1))
    A = np.vstack([x, np.ones(len(g_2_0))]).T
    fit = np.linalg.lstsq(A, y_2, rcond=None)
    b1, b2, b3, alpha = fit[0]
    residuals = fit[1]
    print("alpha: {} b1: {} b2: {} b3: {}".format(alpha, b1, b2, b3))
    # calculate r squared
    def predict(f_2_0, g_2_0, g_2_1):
        return b1*f_2_0 +b2*g_2_0 +b3*g_2_1 +alpha
    y_pred = predict(f_2_0, g_2_0, g_2_1)
    ssreg = np.sum((y_pred - np.mean(y_2))**2)
    sstot = np.sum((y_2-np.mean(y_2))**2)
    r_sq =  ssreg/sstot
    print("Second Order R-Squared: {}".format(r_sq))
    if r_sq < cut:
        print("Attempting first order solver")
        x = np.vstack((f_1_0, g_1_0, g_1_1))
        A = np.vstack([x, np.ones(len(g_1_0))]).T
        fit = np.linalg.lstsq(A, y_1, rcond=None)
        b1, b2, b3, alpha = fit[0]
        residuals = fit[1]
        print("alpha: {} b1: {} b2: {} b3: {}".format(alpha, b1, b2, b3))
        # calculate r squared
        def predict(f_1_0, g_1_0, g_1_1):
            return b1*f_1_0 +b2*g_1_0 +b3*g_1_1 +alpha
        y_pred = predict(f_1_0, g_1_0, g_1_1)
        ssreg = np.sum((y_pred - np.mean(y_1))**2)
        sstot = np.sum((y_1-np.mean(y_1))**2)
        r_sq =  ssreg/sstot
        print("First Order R-Squared: {}".format(r_sq))
        if r_sq < cut:
            # neither solver was satisfactory
            return [alpha, b1, b2, b3, False]
        else:
            # first order solver was satisfactory
            return [alpha, b1, b2, b3, "1st"]
        
    else:
        # second order solver was satisfactory
        return [alpha, b1, b2, b3, "2nd"]
    
def preprocess_ts(pred, TS, Country, Province, pred_start, target):
    # check if we recorded a time series model
    if not TS["USE"]:
        return pred
    # get most recent observation
    p1 = train[Country_Region]==Country
    p2 = train[Province_State]==Province
    p3 = pd.to_datetime(train["Date"]) < pred_start
    # read params
    alpha = TS["alpha"]
    b1 = TS["b1"]
    b2 = TS["b2"]
    b3 = TS["b3"]
    # get two most recent values
    n1 = pd.to_datetime(train.loc[p1&p2&p3, "Date"]).max()
    n2 = n1 - pd.offsets.Day(1)
    n3 = n1 - pd.offsets.Day(2)
    f_n1 = train.loc[p1&p2&(pd.to_datetime(train["Date"])==n1),target].values
    f_n2 = train.loc[p1&p2&(pd.to_datetime(train["Date"])==n2),target].values
    f_n3 = train.loc[p1&p2&(pd.to_datetime(train["Date"])==n3),target].values
    def predict(f_2_0, g_2_0, g_2_1):
        return b1*f_2_0 +b2*g_2_0 +b3*g_2_1 +alpha
    if TS["USE"] == "2nd":
        def predict(f_2_0, g_2_0, g_2_1):
            return b1*f_2_0 +b2*g_2_0 +b3*g_2_1 +alpha
        def next_step(f_2_0, f_0_n1, f_0_n2):
            f_0_0 = f_2_0 +2*f_0_n1 -f_0_n2
            return f_0_0
        def two_diff(a, b, c):
            return a -2*b +c
        # use second order solver
        f_n2 = train.loc[p1&p2&(pd.to_datetime(train["Date"])==n2),target]
        # attempt to get the second derivative at this point
        g_2 = np.diff(pred, 2)
        f_2_n1 = g_2.copy()
        f_hat = pred.copy()
        f_2_n1[0] = two_diff(f_n1,f_n2,f_n3)
        f_hat[0] = next_step(f_2_n1[0], f_n1, f_n2)
        f_2_n1[1] = predict(f_2_n1[0],g_2[0],f_2_n1[0])
        f_hat[1] = next_step(f_2_n1[1], f_hat[0], f_n1)
        # now we can iterate over g when g is differenced
        for i in range(len(pred)-2):
            f_2_n1[i+2] = predict(f_2_n1[i+1], g_2[i+1], g_2[i])
            f_hat[i+2] = next_step(f_2_n1[i+2], f_hat[i+1], f_hat[i])
        pred = f_hat # modified time series
    return pred

def preprocess_ts_all(pred, TS, Country, Province, pred_start, target):
    '''
    If there is at least 1 non-zero previous value for ConfirmedCases
    we don't want to return the predicted cases by the model but instead
    scale the last available observation by the change in predicted cases
    in order to best match the Country's own figures
    '''
    # check if there are at least 1 previous days available
    # get most recent observation
    p1 = train[Country_Region]==Country
    p2 = train[Province_State]==Province
    p3 = pd.to_datetime(train["Date"]) < pred_start
    p4 = train[target]>0
    if train.loc[p1&p2&p3&p4,target].empty:
        print("No previous observations found")
        return pred # return normal predictions
    # get most recent value
    n1 = pd.to_datetime(train.loc[p1&p2&p3, "Date"]).max()
    f_n1 = train.loc[p1&p2&(pd.to_datetime(train["Date"])==n1),target].values
    # get the rate of change
    new_pred = pred.copy()
    delta = pred[1:] / pred[0]
    # get new predictions
    new_pred[0:-1] = f_n1*delta
    new_pred[-1] = new_pred[-2]*delta[-1]
    # add one more prediction to the end (delta reduces number of obs by 1)
    return new_pred
    
    

def warmup(row, pred_date, train, country_data,
           min_cases = 30, min_deaths = 10):
    '''
    Performs a warmup for the SIR model
    
    If there is previous data satisfying SIR training requirements,
        takes the first possible entry and forecasts to pred_date
        We then take the number of infected as ground truth for SIR input for forecast.
        The number of deaths will be the most recent figure available (without looking into future)
        
    If there are previous observations but they do not satisy min_cases, min_deaths
        then the most recent ConfirmedCases is used as a proxy for number infected
        and most recent Fatalities is the number of deaths.
    
    Finally, if no previous observations exist we simply predict 0 for both
        n_infected and n_deaths.
        
    If no Area or Population are avaialbe, simply use their respective means in Couintry_Data
    
    INPUTS:
    row : Pandas Dataframe iterrow row
            Country_Region :
            Province_State : 
            Date           : First date in Set for Predictions
    
    OUTPUTS:
    n_infected, n_dead, t_wuhan, population, area
    '''
    Province_State = "Province_State"
    Country_Region = "Country_Region"
    Population = "Population"
    Density = "Density"
    Area = "Area"
    ConfirmedCases = "ConfirmedCases"
    Fatalities = "Fatalities"
    Date= "Date"
    
    




    
    # extract parameters
    r = 1.8065703208219488
    gamma = 0.07100281878756143
    beta = r*gamma
    mort = 0.1545170936811714
    b1 = -0.05680331911241773
    b2 = 0.2538837598408084
    b3 = 0.02862744019932823
    
    # Population Density Macro Parameters
    density_mean = 289.53457697461425
    density_std = 169.01688157501692
    
    # obtain row data
    Country = row[Country_Region]
    Province = row[Province_State]
    # check if a warmup is possible
    p1 = train[Country_Region]==Country
    p2 = train[Province_State]==Province
    p3 = pd.to_datetime(train["Date"]) < pred_date
    # ConfirmedCases becomes unreliable later on in Pandemic
    p4 = train[ConfirmedCases] >= min_cases
    p5 = train[Fatalities] >= min_deaths
    # Get population density
    q1 = country_data[Country_Region] == Country
    q2 = country_data[Province_State] == Province
    population_df = country_data.loc[q1&q2,Population]
    population = population_df.values
    if population_df.empty:
        print("No population data available for {}-{} \nreplacing with global mean.".format(Country, Province))
        population = country_data[Population].mean()
    area_df = country_data.loc[q1&q2,Area]
    area = area_df.values
    if area_df.empty:
        print("No area data available for {}-{} \nreplacing with global mean.".format(Country, Province))
        area = country_data[Area].mean()
    pop_density = population / area
    std_density = (pop_density - density_mean)/density_std
    # check if warmup is necessary or if historical data available
    if len(train.loc[p1&p2&p3&p4&p5])>0:
        # print("Warmup Data Available")
        # get maximum date available
        max_date = train.loc[p1&p2&p3&p4&p5, "Date"].min() # last reliable point
        I0 =  train.loc[p1&p2&(train["Date"] == max_date)&p4&p5, ConfirmedCases].values
        F0 =  train.loc[p1&p2&(train["Date"] == max_date)&p4&p5, Fatalities].values
        t_wuhan = ((pd.to_datetime(train.loc[p1&p2&(train["Date"] == max_date)&p4&p5, "Date"])-pd.to_datetime(train["Date"]).min())/pd.offsets.Day(1)).values
        # perform warmup with SIR model
        # Want to forecast to the day before our prediction
        forecast_len = ((pred_date-pd.to_datetime(max_date))/pd.offsets.Day(1))
        # get Population, Standardised Density
        # NOTE: IMPLEMENT CHECKS FOR AREA AND POPULATION FIGURES!
        res_np = forecast(I0,
                         F0,
                         population,
                         std_density,
                         beta,
                         gamma,
                         b1,
                         b2,
                         b3,
                         t_wuhan,
                         mort,
                         forecast_len)
        I0 = res_np[-1,1,0]
        # proceed to extract most recent mortality
        max_avail_date = train.loc[p1&p2&p3, "Date"].max() # last available point
        F0 =  train.loc[p1&p2&(train["Date"] == max_avail_date), Fatalities].values
        # attempt to get an ARIMA to fine-tune predictions
        
        pred_cases  = res_np[1:,1,0] + res_np[1:,2,0]
        pred_deaths = res_np[1:,1,0]*mort
        # check that there are sufficient observations to fit the curve
        if len(pred_cases) > 10:
            true_cases  = extract_ts(train, ConfirmedCases, Country, Province, len(pred_cases), t_wuhan)
            true_deaths = extract_ts(train, Fatalities, Country, Province, len(pred_cases), t_wuhan)
            # we may have predicted values for which there are no training obs yet.
            # drop these.
            pred_cases = pred_cases[:len(true_cases)]
            pred_deaths = pred_deaths[:len(true_deaths)]
            TS_c_params = second_order_fit(pred_cases, true_cases)
            TS_d_params = second_order_fit(pred_deaths, true_deaths)
            TS_c = {"alpha": TS_c_params[0],
                  "b1": TS_c_params[1],
                  "b2": TS_c_params[2],
                  "b3": TS_c_params[3],
                  "USE": TS_c_params[4]}
            TS_d = {"alpha": TS_d_params[0],
                  "b1": TS_d_params[1],
                  "b2": TS_d_params[2],
                  "b3": TS_d_params[3],
                  "USE": TS_d_params[4]}
        else:
            # there were not sufficient observations to fit a time series
            TS_c = {"alpha": 0,
                  "b1": 0,
                  "b2": 0,
                  "b3": 1,
                  "USE": False}
            TS_d = {"alpha": 0,
                  "b1": 0,
                  "b2": 0,
                  "b3": 1,
                  "USE": False}
        
    elif len(train.loc[p1&p2&p3])>0:
        # Data available but does not need to be warmed up
        # simply take most recent observation
        max_date = train.loc[p1&p2&p3, "Date"].max() # last available point
        I0 =  train.loc[p1&p2&(train["Date"] == max_date), ConfirmedCases].values
        F0 =  train.loc[p1&p2&(train["Date"] == max_date), Fatalities].values
        TS_c = {"alpha": 0,
              "b1": 0,
              "b2": 0,
              "b3": 1,
              "USE": False}
        TS_d = {"alpha": 0,
              "b1": 0,
              "b2": 0,
              "b3": 1,
              "USE": False}
        # print("Warmup not necessary: {}-{} \nI0: {} F0: {}".format(Country, Province, I0, F0))
    else:
        # No previous data available for this Country-Region pair
        I0 = 0
        F0 = 0
        TS_c = {"alpha": 0,
              "b1": 0,
              "b2": 0,
              "b3": 1,
              "USE": False}
        TS_d = {"alpha": 0,
              "b1": 0,
              "b2": 0,
              "b3": 1,
              "USE": False}
    params = {Population : population,
              "std_density": std_density,
              "beta" : beta,
              "gamma" : gamma,
              "b1": b1,
              "b2": b2,
              "b3": b3,
              "mort": mort}
    return I0, F0, params, TS_c, TS_d

def write_res(test, CC, F, Country, Province, pred_start, TS_c, TS_d):
    '''
    Writes the results to the appropriate cell in the test csv
    CC : Vector of Confirmed Cases (Predicted)
    F  : Vector of Fatalities (Predicted)
    '''
    
    date_range = np.arange(len(CC))*pd.offsets.Day(1) + pred_start
    p1 = test[Country_Region]==Country
    p2 = test[Province_State]==Province
    p3 = pd.to_datetime(test["Date"]).isin(date_range)
    # preprocess with time series model
    CC = preprocess_ts_all(CC, TS_c, Country, Province, pred_start, ConfirmedCases)
    # deaths are ok, we predict them correctly
    # F = preprocess_ts(F, TS_d, Country, Province, pred_start, Fatalities)
    test.loc[p1&p2&p3, ConfirmedCases] = CC
    test.loc[p1&p2&p3, Fatalities]  = F
    return test
    
    
valid_pred_date = pd.to_datetime(test["Date"]).min()
valid_forecast_len = 14 # days!
test_forecast_len = (pd.to_datetime(test["Date"]).max()-valid_pred_date)/pd.offsets.Day(1) - valid_forecast_len+1
print("test forecast length: {}".format(test_forecast_len))
# time since Wuhan for valid
valid_wuhan = ((valid_pred_date - pd.to_datetime(train["Date"]).min())/pd.offsets.Day(1))
test_pred_date = valid_pred_date + pd.offsets.Day(valid_forecast_len)
test_wuhan = valid_wuhan + valid_forecast_len
for idx, row in test_group.iterrows():
    # first perform the valid set forecast
    I0_v, F0_v, params, TS_c, TS_d = warmup(row, valid_pred_date, train, country_data,
                    min_cases = 30, min_deaths = 10)
    # generate validation forecast
    valid_np = forecast(I0_v,
                     F0_v,
                     params[Population],
                     params["std_density"],
                     params["beta"],
                     params["gamma"],
                     params["b1"],
                     params["b2"],
                     params["b3"],
                     valid_wuhan,
                     params["mort"],
                     valid_forecast_len)
    # predicted cumulative cases = I_t + R_t
    # ie. those infected now and those who have recovered
    I_v = valid_np[:,1,0]
    R_v = valid_np[:,2,0]
    F_v = R_v * params["mort"]
    CC_v = I_v + R_v - F_v
    print("{}-{}: I0:{:.2f} F0:{:.2f} -> I7:{:.2f} F7:{:.2f} CC7: {:.2f}".format(row[Country_Region],
                                                     row[Province_State],
                                                     float(I0_v),
                                                     float(F0_v),
                                                     float(I_v[-1]),
                                                     float(F_v[-1]),
                                                     float(CC_v[-1])))
    # write these results to the test file
    test = write_res(test, CC_v[1:], F_v[1:], row[Country_Region], row[Province_State], valid_pred_date,
                    TS_c, TS_d)
    print("{}-{} Validation results written successfully".format(row[Country_Region], row[Province_State]))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Attempt to make forecast for test period       #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    I0_f, F0_f, params, TS_c, TS_d = warmup(row, test_pred_date, train, country_data,
                    min_cases = 30, min_deaths = 10)
    # generate validation forecast
    test_np = forecast(I0_f,
                     F0_f,
                     params[Population],
                     params["std_density"],
                     params["beta"],
                     params["gamma"],
                     params["b1"],
                     params["b2"],
                     params["b3"],
                     valid_wuhan,
                     params["mort"],
                     test_forecast_len)
    # predicted cumulative cases = I_t + R_t
    # ie. those infected now and those who have recovered
    I_f = test_np[:,1,0]
    R_f = test_np[:,2,0]
    F_f = R_f * params["mort"]
    CC_f = I_f + R_f - F_f
    # write these results to the test file
    test = write_res(test, CC_f[1:], F_f[1:], row[Country_Region], row[Province_State], test_pred_date, TS_c, TS_d)
    print("{}-{} Test results written successfully".format(row[Country_Region], row[Province_State]))
# Save the test csv in submission format
test[["ForecastId", ConfirmedCases, Fatalities]].to_csv("submission.csv", index=False)

# submit
# !kaggle competitions submit -c covid19-global-forecasting-week-3 -f submission.csv -m "First Submission"