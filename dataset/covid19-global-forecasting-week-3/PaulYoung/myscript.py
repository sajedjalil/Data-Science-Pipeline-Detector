# %% [code]
# Import libraries/packages
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import importlib as imp


sys.path.append("../input/competitionfunctions")
import Competition_Functions

from Competition_Functions import Beta_Estimation_Option1_comp
from Competition_Functions import my_SIR_Model_comp


    

#Specify atomic region
match_by_list = ['Province_State', 'Country_Region'] 


# References
#     Methods for estimating disease transmission rates: Evaluating the precision of Poisson regression and two novel methods
#     Population Key - # Courtesy of Kaggle user dgrechka

# %% [code]
# Import Training Data and Test Data in Dataframe
df_Waveforms_Train = pd.read_csv("../input/competitiondata/train.csv")
df_Waveforms_Test = pd.read_csv("../input/competitiondata/test.csv")
    # remember to zero pad month and day in Excel before importing
    
df_Waveforms_Train["Province_State"].replace({np.nan : '(whole country)'}, inplace=True) # Replace all nans in Province_State Column
df_Waveforms_Test["Province_State"].replace({np.nan : '(whole country)'}, inplace=True) # Replace all nans in Province_State Column


# Population Key - # Courtesy of Kaggle user dgrechka
    # Ms
population_key = pd.read_csv("../input/population-key/my_locations_population.csv")
    # Replace all nans in Province_State Column
population_key["Province_State"].replace({np.nan : '(whole country)'}, inplace=True)

# Generate Consolidated DF for Information
df_Info_Train_noPop = df_Waveforms_Train.drop(columns = ['ConfirmedCases', 'Fatalities'], axis=1) # Not Using Waveform Data in this DF
df_Info_Train_noPop = df_Info_Train_noPop.groupby(match_by_list, as_index=False).aggregate({'Date': 'max'}) #consolidate the many duplicate rows


# Add Populations into Info df.  This will remove some of the countries that aren't presently included in population key
df_Info_Train = pd.merge(left=df_Info_Train_noPop, right=population_key, how='inner', left_on=match_by_list, right_on=match_by_list)


# Add empty Beta and Death rate to df_Info
init_Beta_arr = np.zeros(df_Info_Train.shape[0])
init_DeathRate_arr = np.zeros(df_Info_Train.shape[0])
df_Info_Train["Beta"] = init_Beta_arr
df_Info_Train["DeathRate"] = init_DeathRate_arr

# %% [code]
# Generate Training batch
min_date = '03/01/20'
max_date = '03/26/20'
df_Waveforms_Training_Batch = df_Waveforms_Train[(df_Waveforms_Train["Date"] <= max_date) & (df_Waveforms_Train["Date"] >= min_date)]



# %% [code]
# Loop Through atomic regions and calculate infection rate and death rate
Beta_arr = np.zeros(df_Info_Train.shape[0])
deathrate_arr = np.zeros(df_Info_Train.shape[0])

for row in np.arange(df_Info_Train.shape[0]):
    
    # For each iteration, extract Province/Country Waveforms from Batch
    Province = df_Info_Train.iloc[row][0]
    Country = df_Info_Train.iloc[row][1]
    Population = df_Info_Train.iloc[row][3]

    df_region = df_Waveforms_Training_Batch[(df_Waveforms_Training_Batch["Province_State"]==Province) & (df_Waveforms_Training_Batch["Country_Region"]==Country)]

    # Province/Country Waveforms
    C_waveform = np.expand_dims(np.array(df_region)[:, 4].astype(float), axis=0)
    D_waveform = np.expand_dims(np.array(df_region)[:, 5].astype(float), axis=0)
    
    #Calculate Beta, death rate
    Beta, death_rate = Beta_Estimation_Option1_comp(Infected_arr=C_waveform, Deaths_arr=D_waveform, t0=15, t1=20, Population=Population, Gamma=1/14)
    Beta_arr[row] = Beta
    deathrate_arr[row] = death_rate
    #print('Beta = ' + str(Beta) + ', death rate = ' + str(death_rate))
df_Info_Train["Beta"] = Beta_arr
df_Info_Train["DeathRate"] = deathrate_arr

    #df_Info_Train.iloc[row][4] = Beta
   # df_Info_Train.iloc[row][5] = death_rate
    

# %% [code]
#Validation -  This will only include countries in the population key


# Range of Dates for Validation
min_date = '03/25/20'
max_date = '04/03/20'

# Obtain Validation Batch 
df_Waveforms_Validation_Batch = df_Waveforms_Train[(df_Waveforms_Train["Date"] <= max_date) & (df_Waveforms_Train["Date"] >= min_date)]

# Generate Predictions
   # Append 2 columns in Validation Batch titled "ConfirmedCases (predicted) and "Fatalities" (predicted)"
init_ConfirmedCases_pred = np.zeros(df_Waveforms_Validation_Batch.shape[0])
init_Fatalities_pred = np.zeros(df_Waveforms_Validation_Batch.shape[0])
df_Waveforms_Validation_Batch.loc[:,"ConfirmedCases_pred"]=init_ConfirmedCases_pred
df_Waveforms_Validation_Batch.loc[:, "Fatalities_pred"]=init_Fatalities_pred

#    Loop through regions in Info table
for row in np.arange(df_Info_Train.shape[0]):
    
    # For each iteration, extract Province/Country Waveforms from Batch
    Province = df_Info_Train.iloc[row, 0]
    Country = df_Info_Train.iloc[row, 1]
    Population = df_Info_Train.iloc[row, 3]
    # Take Beta from Info Table
    Beta = df_Info_Train.iloc[row, 4]
    DeathRate =  df_Info_Train.iloc[row, 5]
    
    # Filter by the region.  Calculate number of validation days
    df_validation_region = df_Waveforms_Validation_Batch[(df_Waveforms_Validation_Batch["Province_State"]==Province) & (df_Waveforms_Validation_Batch["Country_Region"]==Country)]
    index_start = df_validation_region.index[0]
    index_stop = df_validation_region.index[-1]
    nDays = index_stop - index_start + 1

    # take last I and D from training batch (first day of validation batch, which must overlap with training batch)
    df_training_region = df_Waveforms_Training_Batch[(df_Waveforms_Training_Batch["Province_State"]==Province) & (df_Waveforms_Training_Batch["Country_Region"]==Country)]
    
    earliest_validation_date = df_validation_region.loc[index_start, "Date"]
    Infected_0 = np.array(df_training_region[df_training_region["Date"]==earliest_validation_date]["ConfirmedCases"]).astype(float)[0]
    Deaths_0 = np.array(df_training_region[df_training_region["Date"]==earliest_validation_date]["Fatalities"]).astype(float)[0]
    


    # Generate I_arr_pred and D_arr_pred
    sird_0 = np.array([1-Infected_0/Population, Infected_0/Population, 0, Deaths_0/Population])
    sird_params = np.array([Beta, 1/14, DeathRate])
    sird_arr = my_SIR_Model_comp(sird_0=sird_0, t_0=0, sird_params=sird_params, nDays=nDays, data_arr=None)
    
    # Insert these arrays in the 2 appended columns of df_Waveforms_Validation_Batch
    df_Waveforms_Validation_Batch.loc[index_start:index_stop, "ConfirmedCases_pred"] = np.round(sird_arr[:,1 ]*Population)
    df_Waveforms_Validation_Batch.loc[index_start:index_stop, "Fatalities_pred"] = np.round(sird_arr[:,3 ]*Population)
#        PLot real vs predicted data if desired

# Cut out first date in validation, as this is actually part of training batch
df_Waveforms_Validation_Batch=df_Waveforms_Validation_Batch[df_Waveforms_Validation_Batch["Date"]> min_date]



# %% [code] {"scrolled":true}

# Calculate Validation Loss Function
nRows = df_Waveforms_Validation_Batch.shape[0]
MSLE = 0
RMSLE_arr = np.zeros(nRows)
for row in np.arange(nRows):
    actual_confirmed  = df_Waveforms_Validation_Batch.iloc[row, 4]
    pred_confirmed  = df_Waveforms_Validation_Batch.iloc[row, 6]
    actual_deaths  = df_Waveforms_Validation_Batch.iloc[row, 5]
    pred_deaths  = df_Waveforms_Validation_Batch.iloc[row, 7]
    
    sum_term = (np.log(pred_confirmed + 1) - np.log(actual_confirmed+1))**2 + (np.log(pred_deaths + 1) - np.log(actual_deaths+1))**2
    if (np.isnan(sum_term)) or (np.isinf(sum_term)) :
        print(df_Waveforms_Validation_Batch.iloc[row,:])
    else:
        MSLE += sum_term/2/nRows
        RMSLE_arr[row] = np.sqrt(sum_term/2)
RMSLE = np.sqrt(MSLE)


# %% [code]
# Generate Test Batch from test.csv

# Append 2 columns in Validation Batch titled "ConfirmedCases (predicted) and "Fatalities" (predicted)"
init_ConfirmedCases_pred = np.zeros(df_Waveforms_Test.shape[0])
init_Fatalities_pred = np.zeros(df_Waveforms_Test.shape[0])
df_Waveforms_Test.loc[:,"ConfirmedCases"]=init_ConfirmedCases_pred
df_Waveforms_Test.loc[:, "Fatalities"]=init_Fatalities_pred

# Generate Predictions

#    Loop through regions in Info table
for row in np.arange(df_Info_Train.shape[0]):
    
    # For each iteration, extract Province/Country, Population, Beta, and DeathRate from Info Table
    Province = df_Info_Train.iloc[row, 0]
    Country = df_Info_Train.iloc[row, 1]
    Population = df_Info_Train.iloc[row, 3]
    Beta = df_Info_Train.iloc[row, 4]
    DeathRate =  df_Info_Train.iloc[row, 5]
    
    # Filter by the region.  Calculate number of validation days
    df_test_region = df_Waveforms_Test[(df_Waveforms_Test["Province_State"]==Province) & (df_Waveforms_Test["Country_Region"]==Country)]
    index_start = df_test_region.index[0]
    index_stop = df_test_region.index[-1]
    nDays = index_stop - index_start + 1

    # take last I and D from training batch (first day of validation batch, which must overlap with training batch)
    df_training_region = df_Waveforms_Training_Batch[(df_Waveforms_Training_Batch["Province_State"]==Province) & (df_Waveforms_Training_Batch["Country_Region"]==Country)]
    
    earliest_test_date = df_test_region.loc[index_start, "Date"]
    Infected_0 = np.array(df_training_region[df_training_region["Date"]==earliest_test_date]["ConfirmedCases"]).astype(float)[0]
    Deaths_0 = np.array(df_training_region[df_training_region["Date"]==earliest_test_date]["Fatalities"]).astype(float)[0]
    


    # Generate I_arr_pred and D_arr_pred
    sird_0 = np.array([1-Infected_0/Population, Infected_0/Population, 0, Deaths_0/Population])
    sird_params = np.array([Beta, 1/14, DeathRate])
    sird_arr = my_SIR_Model_comp(sird_0=sird_0, t_0=0, sird_params=sird_params, nDays=nDays, data_arr=None)
    
    # Insert these arrays in the 2 appended columns of df_Waveforms_Validation_Batch
    df_Waveforms_Test.loc[index_start:index_stop, "ConfirmedCases"] = np.round(sird_arr[:,1 ]*Population).astype(int)
    df_Waveforms_Test.loc[index_start:index_stop, "Fatalities"] = np.round(sird_arr[:,3 ]*Population).astype(int)
#        PLot real vs predicted data if desired

# Cut out first date in validation, as this is actually part of training batch
#df_Waveforms_Validation_Batch=df_Waveforms_Validation_Batch[df_Waveforms_Validation_Batch["Date"]> min_date]
df_Submission = df_Waveforms_Test.drop(columns = ['Province_State', 'Country_Region', 'Date'], axis=1)

# %% [code]
#Submit to csv
df_Submission.to_csv('submission.csv', index=False)

print('complete')
