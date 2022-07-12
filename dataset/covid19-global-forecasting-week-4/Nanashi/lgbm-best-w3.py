
# Parameters - can be changed
BAGS = 25
SEED = 1234
SET_FRAC = 0.01


# Parameters - Other

TRUNCATED = False


DROPS = True
PRIVATE = True
USE_PRIORS = False


SUP_DROP = 0.0
ACTIONS_DROP = 0.0
PLACE_FRACTION = 1.0  # 0.4 

#** FEATURE_DROP = 0.4 # drop random % of features (HIGH!!!, speeds it up)
#** COUNTRY_DROP = 0.35 # drop random % of countries (20-30pct)
#** FIRST_DATE_DROP = 0.5 # Date_f must be after a certain date, randomly applied

# FEATURE_DROP_MAX = 0.3
LT_DECAY_MAX = 0.3
LT_DECAY_MIN = -0.4

SINGLE_MODEL = False
MODEL_Y = 'agg_dff' # 'slope'  # 'slope' or anything else for difference/aggregate log gain


# %% [code]


# %% [markdown]
# 
# ### Init

# %% [code]
import pandas as pd
import numpy as np
import os

# %% [code]
from collections import Counter
from random import shuffle
import math

# %% [code]
from scipy.stats.mstats import gmean


# %% [code]
import datetime

# %% [code]
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns

# %% [code]
pd.options.display.float_format = '{:.8}'.format


# %% [code]
plt.rcParams["figure.figsize"] = (12, 4.75)
 
# %% [code]
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
 

    
pd.options.display.max_rows = 999
    
# %% [code]




# %% [markdown]
# ### Import and Adjust

# %% [markdown]
# #### Import

# %% [code]
path = '/kaggle/input/c19week3'
input_path = '/kaggle/input/covid19-global-forecasting-week-4'

# %% [code]
train = pd.read_csv(input_path + '/train.csv')
test = pd.read_csv(input_path  + '/test.csv')
sub = pd.read_csv(input_path + '/submission.csv')


tt = pd.merge(train, test, on=['Country_Region', 'Province_State', 'Date'], 
              how='right', validate="1:1")\
                    .fillna(method = 'ffill')
public = tt[['ForecastId', 'ConfirmedCases', 'Fatalities']]
      
# %% [raw]
# len(train)
# len(test)

# %% [code]
train.Date.max()

# %% [code]
test_dates = test.Date.unique()
test_dates

# %% [raw]
# # simulate week 1 sort of 
# test = test[ test.Date >=  '2020-03-25']

# %% [raw]
# test

# %% [code]
pp = 'public'

# %% [code]
#FINAL_PUBLIC_DATE = datetime.datetime(2020, 4, 8)

if PRIVATE:
    test = test[ pd.to_datetime(test.Date) >  train.Date.max()]
    pp = 'private'

# %% [code]
test.Date.unique()

# %% [markdown]
# ### Train Fix

# %% [markdown]
# #### Supplement Missing US Data

# %% [code]
revised = pd.read_csv(path + '/outside_data' + 
                          '/covid19_train_data_us_states_before_march_09_new.csv')


# %% [raw]
# revised.Date = pd.to_datetime(revised.Date)
# revised.Date = revised.Date.apply(datetime.datetime.strftime, args= ('%Y-%m-%d',))

# %% [code]
revised = revised[['Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']]

# %% [code]
train.tail()

# %% [code]
revised.head()

# %% [code]
train.Date = pd.to_datetime(train.Date)
revised.Date = pd.to_datetime(revised.Date)

# %% [code]
rev_train = pd.merge(train, revised, on=['Province_State', 'Country_Region', 'Date'],
                            suffixes = ('', '_r'), how='left')

# %% [code]


# %% [code]
rev_train[~rev_train.ConfirmedCases_r.isnull()].head()

# %% [code]


# %% [code]


# %% [code]
rev_train.ConfirmedCases = \
    np.where( (rev_train.ConfirmedCases == 0) & ((rev_train.ConfirmedCases_r > 0 )) &
                 (rev_train.Country_Region == 'US'),
        
        rev_train.ConfirmedCases_r,
            rev_train.ConfirmedCases)


# %% [code]
rev_train.Fatalities = \
    np.where( ~rev_train.Fatalities_r.isnull() & 
                (rev_train.Fatalities == 0) & ((rev_train.Fatalities_r > 0 )) &
                 (rev_train.Country_Region == 'US')
             ,
        
        rev_train.Fatalities_r,
            rev_train.Fatalities)


# %% [code]
rev_train.drop(columns = ['ConfirmedCases_r', 'Fatalities_r'], inplace=True)

# %% [code]
train = rev_train

# %% [raw]
# train[train.Province_State == 'California']

# %% [raw]
# import sys
# def sizeof_fmt(num, suffix='B'):
#     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f %s%s" % (num, 'Yi', suffix)
# 
# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key= lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
# 

# %% [markdown]
# ### Oxford Actions Database

# %% [code]
# contain_data = pd.read_excel(path + '/outside_data' + 
#                           '/OxCGRT_Download_latest_data.xlsx')

contain_data = pd.read_csv(path + '/outside_data' + 
                          '/OxCGRT_Download_070420_160027_Full.csv')

# %% [code] {"scrolled":true}
contain_data = contain_data[[c for c in contain_data.columns if 
                      not any(z in c for z in ['_Notes','Unnamed', 'Confirmed',
                                               'CountryCode',
                                                      'S8', 'S9', 'S10','S11',
                                              'StringencyIndexForDisplay'])] ]\
        

# %% [code]
contain_data.rename(columns = {'CountryName': "Country"}, inplace=True)

# %% [code]
contain_data.Date = contain_data.Date.astype(str)\
    .apply(datetime.datetime.strptime, args=('%Y%m%d', ))

# %% [code]


# %% [code]
contain_data_orig = contain_data.copy()

# %% [code]
contain_data.columns

# %% [raw]
# contain_data.columns

# %% [code]


# %% [code]
cds = []
for country in contain_data.Country.unique():
    cd = contain_data[contain_data.Country==country]
    cd = cd.fillna(method = 'ffill').fillna(0)
    cd.StringencyIndex = cd.StringencyIndex.cummax()  # for now
    col_count = cd.shape[1]
    
    # now do a diff columns
    # and ewms of it
    for col in [c for c in contain_data.columns if 'S' in c]:
        col_diff = cd[col].diff()
        cd[col+"_chg_5d_ewm"] = col_diff.ewm(span = 5).mean()
        cd[col+"_chg_20_ewm"] = col_diff.ewm(span = 20).mean()
        
    # stringency
    cd['StringencyIndex_5d_ewm'] = cd.StringencyIndex.ewm(span = 5).mean()
    cd['StringencyIndex_20d_ewm'] = cd.StringencyIndex.ewm(span = 20).mean()
    
    cd['S_data_days'] =  (cd.Date - cd.Date.min()).dt.days
    for s in [1, 10, 20, 30, 50, ]:
        cd['days_since_Stringency_{}'.format(s)] = \
                np.clip((cd.Date - cd[(cd.StringencyIndex > s)].Date.min()).dt.days, 0, None)
    
    
    cds.append(cd.fillna(0)[['Country', 'Date'] + cd.columns.to_list()[col_count:]])
contain_data = pd.concat(cds)

# %% [raw]
# contain_data.columns

# %% [raw]
# dataset.groupby('Country').S_data_days.max().sort_values(ascending = False)[-30:]

# %% [raw]
# contain_data.StringencyIndex.cummax()

# %% [raw]
# contain_data.groupby('Date').count()[90:]

# %% [code]
contain_data.Date.max()

# %% [code]
contain_data.columns

# %% [code]
contain_data[contain_data.Country == 'Australia']

# %% [code]
contain_data.shape

# %% [raw]
# contain_data.groupby('Country').Date.max()[:50]

# %% [code]
contain_data.Country.replace({ 'United States': "US",
                                 'South Korea': "Korea, South",
                                    'Taiwan': "Taiwan*",
                              'Myanmar': "Burma", 'Slovak Republic': "Slovakia",
                                  'Czech Republic': 'Czechia',

}, inplace=True)

# %% [code]
set(contain_data.Country) - set(test.Country_Region)

# %% [code]


# %% [markdown]
# #### Load in Supplementary Data

# %% [code]
sup_data = pd.read_excel(path + '/outside_data' + 
                          '/Data Join - Copy1.xlsx')


# %% [code]
sup_data.columns = [c.replace(' ', '_') for c in sup_data.columns.to_list()]

# %% [code]
sup_data.drop(columns = [c for c in sup_data.columns.to_list() if 'Unnamed:' in c], inplace=True)

# %% [code]


# %% [code]


# %% [raw]
# sup_data.drop(columns = ['longitude', 'temperature', 'humidity',
#                         'latitude'], inplace=True)

# %% [raw]
# sup_data.columns

# %% [raw]
# sup_data.drop(columns = [c for c in sup_data.columns if 
#                                  any(z in c for z in ['state', 'STATE'])], inplace=True)

# %% [raw]
# sup_data = sup_data[['Province_State', 'Country_Region',
#                      'Largest_City',
#                      'IQ', 'GDP_region', 
#                      'TRUE_POPULATION', 'pct_in_largest_city', 
#                    'Migrant_pct',
#                     'Avg_age',
#                      'latitude', 'longitude',
#                 'abs_latitude', #  'Personality_uai', 'Personality_ltowvs',
#               'Personality_pdi',
# 
#                  'murder',  'real_gdp_growth'
#                     ]]

# %% [raw]
# sup_data = sup_data[['Province_State', 'Country_Region',
#                      'Largest_City',
#                      'IQ', 'GDP_region', 
#                      'TRUE_POPULATION', 'pct_in_largest_city', 
#                    #'Migrant_pct',
#                     # 'Avg_age',
#                      # 'latitude', 'longitude',
#              #    'abs_latitude', #  'Personality_uai', 'Personality_ltowvs',
#             #   'Personality_pdi',
# 
#                  'murder', # 'real_gdp_growth'
#                     ]]

# %% [code]
sup_data.drop(columns = [ 'Date', 'ConfirmedCases',
       'Fatalities', 'log-cases', 'log-fatalities', 'continent'], inplace=True)

# %% [raw]
# sup_data.drop(columns = [ 'Largest_City',  
#                         'continent_gdp_pc', 'continent_happiness', 'continent_generosity',
#        'continent_corruption', 'continent_Life_expectancy', 'TRUE_CHINA',
#                          'Happiness', 'Logged_GDP_per_capita',
#        'Social_support','HDI', 'GDP_pc', 'pc_GDP_PPP', 'Gini',
#                          'state_white', 'state_white_asian', 'state_black',
#        'INNOVATIVE_STATE','pct_urban', 'Country_pop', 
#                         
#                         ], inplace=True)

# %% [raw]
# sup_data.columns

# %% [raw]
# 

# %% [code]
sup_data['Migrants_in'] = np.clip(sup_data.Migrants, 0, None)
sup_data['Migrants_out'] = -np.clip(sup_data.Migrants, None, 0)
sup_data.drop(columns = 'Migrants', inplace=True)

# %% [raw]
# sup_data.loc[:, 'Largest_City'] = np.log(sup_data.Largest_City + 1)

# %% [code]
sup_data.head()

# %% [code]


# %% [code]
sup_data.shape

# %% [raw]
# sup_data.loc[4][:50]

# %% [code]


# %% [markdown]
# #### Revise Columns

# %% [code]
train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)
#contain_data.Date = pd.to_datetime(contain_data.Date)

# %% [code]
train.rename(columns={'Country_Region': 'Country'}, inplace=True)
test.rename(columns={'Country_Region': 'Country'}, inplace=True)
sup_data.rename(columns={'Country_Region': 'Country'}, inplace=True)


# %% [code]
train['Place'] = train.Country + train.Province_State.fillna("")
test['Place'] = test.Country +  test.Province_State.fillna("")









# %% [code]
sup_data['Place'] = sup_data.Country +  sup_data.Province_State.fillna("")

# %% [code]
len(train.Place.unique())

# %% [code]
sup_data = sup_data[    
    sup_data.columns.to_list()[2:]]

# %% [code]
sup_data = sup_data.replace('N.A.', np.nan).fillna(-0.5)

# %% [code]
for c in sup_data.columns[:-1]:
    m = sup_data[c].max() #- sup_data 
    
    if m > 300 and c!='TRUE_POPULATION':
        print(c)
        sup_data[c] = np.log(sup_data[c] + 1)
        assert sup_data[c].min() > -1

# %% [code]
for c in sup_data.columns[:-1]:
    m = sup_data[c].max() #- sup_data 
    
    if m > 300:
        print(c)

# %% [code]


# %% [code]
DEATHS = 'Fatalities'

# %% [code]


# %% [code]
len(train.Place.unique())

# %% [code]


# %% [markdown]
# #### Correct Drop-Offs with interpolation

# %% [raw]
# 
# train[(train.ConfirmedCases.shift(1) > train.ConfirmedCases) & 
#          (train.Place == train.Place.shift(1)) & (train.ConfirmedCases == 0)]

# %% [code]


# %% [code]
train.ConfirmedCases = \
    np.where(
        (train.ConfirmedCases.shift(1) > train.ConfirmedCases) & 
        (train.ConfirmedCases.shift(1) > 0) & (train.ConfirmedCases.shift(-1) > 0) &
         (train.Place == train.Place.shift(1)) & (train.Place == train.Place.shift(-1)) & 
        ~train.ConfirmedCases.shift(-1).isnull(),
        
        np.sqrt(train.ConfirmedCases.shift(1) * train.ConfirmedCases.shift(-1)),
        
        train.ConfirmedCases)



# %% [code]
train.Fatalities = \
    np.where(
        (train.Fatalities.shift(1) > train.Fatalities) & 
        (train.Fatalities.shift(1) > 0) & (train.Fatalities.shift(-1) > 0) &
         (train.Place == train.Place.shift(1)) & (train.Place == train.Place.shift(-1)) & 
        ~train.Fatalities.shift(-1).isnull(),
        
        np.sqrt(train.Fatalities.shift(1) * train.Fatalities.shift(-1)),
        
        train.Fatalities)



# %% [code]


# %% [code]
for i in [0, -1]:
    train.ConfirmedCases = \
        np.where(
            (train.ConfirmedCases.shift(2+ i ) > train.ConfirmedCases) & 
            (train.ConfirmedCases.shift(2+ i) > 0) & (train.ConfirmedCases.shift(-1+ i) > 0) &
         (train.Place == train.Place.shift(2+ i)) & (train.Place == train.Place.shift(-1+ i)) & 
            ~train.ConfirmedCases.shift(-1+ i).isnull(),

            np.sqrt(train.ConfirmedCases.shift(2+ i) * train.ConfirmedCases.shift(-1+ i)),

            train.ConfirmedCases)



# %% [code]



# %% [code]


# %% [code]
train[train.Place=='USVirgin Islands'][-10:]

# %% [code] {"scrolled":true}

train[(train.ConfirmedCases.shift(2) > 2* train.ConfirmedCases) & 
         (train.Place == train.Place.shift(2)) & (train.ConfirmedCases < 100000)]

# %% [code]

train[(train.Fatalities.shift(1) > train.Fatalities) & 

      (train.Place == train.Place.shift(1)) & (train.Fatalities < 10000)]

# %% [code]


# %% [code]


# %% [markdown]
# ### Use Training Set that is Old Predictions

# %% [code]

# %% [code]
train_bk = train.copy()

# %% [raw]
# train.Date.unique()

# %% [markdown]
# #### Possible Truncation for Test Set Prediction

# %% [code]
full_train = train.copy()

# %% [raw]
# full_train[full_train.Place =='USVirgin Islands']

# %% [markdown]
# ### Graphs

# %% [code]
train_c = train[train.Country == 'China']
train_nc = train[train.Country != 'China']
train_us = train[train.Country == 'US']
# train_nc = train[train.Country != 'China']

# %% [raw]
# data.shape
# data[data.ConfirmedCases > 0].shape
# data.ConfirmedCases

# %% [code]
def lplot(data, minDate = datetime.datetime(2000, 1, 1), 
              columns = ['ConfirmedCases', 'Fatalities']):
    return
        

# %% [code]
REAL = datetime.datetime(2020, 2, 10)


# %% [code]
dataset = train.copy()


if TRUNCATED:
    dataset = dataset[dataset.Country.isin(
        ['Italy', 'Spain', 'Germany', 'Portugal', 'Belgium', 'Austria', 'Switzerland' ])]

# %% [code]
dataset.head()

# %% [code]


# %% [code]


# %% [code]


# %% [markdown]
# ### Create Lagged Growth Rates (4, 7, 12, 20 day rates)

# %% [code]
def rollDates(df, i, preserve=False):
    df = df.copy()
    if preserve:
        df['Date_i'] = df.Date
    df.Date = df.Date + datetime.timedelta(i)
    return df

# %% [code]
WINDOWS = [1, 2,  4, 7, 12, 20, 30]

# %% [code]
for window in WINDOWS:
    csuffix = '_{}d_prior_value'.format(window)
    
    base = rollDates(dataset, window)
    dataset = pd.merge(dataset, base[['Date', 'Place',
                'ConfirmedCases', 'Fatalities']], on = ['Date', 'Place'],
            suffixes = ('', csuffix), how='left')
#     break;
    for c in ['ConfirmedCases', 'Fatalities']:
        dataset[c+ csuffix].fillna(0, inplace=True)
        dataset[c+ csuffix] = np.log(dataset[c + csuffix] + 1)
        dataset[c+ '_{}d_prior_slope'.format(window)] = \
                    (np.log(dataset[c] + 1) \
                         - dataset[c+ csuffix]) / window
        dataset[c+ '_{}d_ago_zero'.format(window)] = 1.0*(dataset[c+ csuffix] == 0)     
    
    
    

# %% [code]
for window1 in WINDOWS:
    for window2 in WINDOWS:
        for c in ['ConfirmedCases', 'Fatalities']:
            if window1 * 1.3 < window2 and window1 * 5 > window2:
                dataset[ c +'_{}d_{}d_prior_slope_chg'.format(window1, window2) ] = \
                        dataset[c+ '_{}d_prior_slope'.format(window1)] \
                                - dataset[c+ '_{}d_prior_slope'.format(window2)]
                
                

# %% [raw]
# dataset.tail()

# %% [raw]
# dataset

# %% [markdown]
# #### First Case Etc.

# %% [code]
first_case = dataset[dataset.ConfirmedCases >= 1].groupby('Place').min() 
tenth_case = dataset[dataset.ConfirmedCases >= 10].groupby('Place').min()
hundredth_case = dataset[dataset.ConfirmedCases >= 100].groupby('Place').min()
thousandth_case = dataset[dataset.ConfirmedCases >= 1000].groupby('Place').min()

# %% [code]
first_fatality = dataset[dataset.Fatalities >= 1].groupby('Place').min()
tenth_fatality = dataset[dataset.Fatalities >= 10].groupby('Place').min()
hundredth_fatality = dataset[dataset.Fatalities >= 100].groupby('Place').min()
thousandth_fatality = dataset[dataset.Fatalities >= 1000].groupby('Place').min()


# %% [raw]
# np.isinf(dataset.days_since_hundredth_case).sum()

# %% [raw]
# (dataset.Date - hundredth_case.loc[dataset.Place].Date.values).dt.days

# %% [code]
dataset['days_since_first_case'] = \
        np.clip((dataset.Date - first_case.loc[dataset.Place].Date.values).dt.days\
                            .fillna(-1), -1, None)
dataset['days_since_tenth_case'] = \
        np.clip((dataset.Date - tenth_case.loc[dataset.Place].Date.values).dt.days\
                            .fillna(-1), -1, None)
dataset['days_since_hundredth_case'] = \
        np.clip((dataset.Date - hundredth_case.loc[dataset.Place].Date.values).dt.days\
                            .fillna(-1), -1, None)
dataset['days_since_thousandth_case'] = \
        np.clip((dataset.Date - thousandth_case.loc[dataset.Place].Date.values).dt.days\
                            .fillna(-1), -1, None)


# %% [code]
dataset['days_since_first_fatality'] = \
        np.clip((dataset.Date - first_fatality.loc[dataset.Place].Date.values).dt.days\
                    .fillna(-1), -1, None)
dataset['days_since_tenth_fatality'] = \
        np.clip((dataset.Date - tenth_fatality.loc[dataset.Place].Date.values).dt.days\
                    .fillna(-1), -1, None)
dataset['days_since_hundredth_fatality'] = \
        np.clip((dataset.Date - hundredth_fatality.loc[dataset.Place].Date.values).dt.days\
                    .fillna(-1), -1, None)
dataset['days_since_thousandth_fatality'] = \
        np.clip((dataset.Date - thousandth_fatality.loc[dataset.Place].Date.values).dt.days\
                    .fillna(-1), -1, None)

# %% [code]


# %% [code]
dataset['case_rate_since_first_case'] = \
    np.clip((np.log(dataset.ConfirmedCases + 1) \
             - np.log(first_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)
dataset['case_rate_since_tenth_case'] = \
    np.clip((np.log(dataset.ConfirmedCases + 1) \
             - np.log(tenth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) \
                    / (dataset.days_since_tenth_case+0.01), 0, 1)
dataset['case_rate_since_hundredth_case'] = \
    np.clip((np.log(dataset.ConfirmedCases + 1) \
             - np.log(hundredth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)
dataset['case_rate_since_thousandth_case'] = \
    np.clip((np.log(dataset.ConfirmedCases + 1) \
             - np.log(thousandth_case.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)

# %% [code]
dataset['fatality_rate_since_first_case'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(first_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)
dataset['fatality_rate_since_tenth_case'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(tenth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)
dataset['fatality_rate_since_hundredth_case'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(hundredth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)
dataset['fatality_rate_since_thousandth_case'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(thousandth_case.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_first_case+0.01), 0, 1)


#.plot(kind='hist', bins = 150)

# %% [code]
dataset['fatality_rate_since_first_fatality'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(first_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_first_fatality+0.01), 0, 1)
dataset['fatality_rate_since_tenth_fatality'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(tenth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_tenth_fatality+0.01), 0, 1)
dataset['fatality_rate_since_hundredth_fatality'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(hundredth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_hundredth_fatality+0.01), 0, 1)
dataset['fatality_rate_since_thousandth_fatality'] = \
    np.clip((np.log(dataset.Fatalities + 1) \
             - np.log(thousandth_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1)) \
                    / (dataset.days_since_thousandth_fatality+0.01), 0, 1)
 
#.plot(kind='hist', bins = 150)

# %% [code]


# %% [code]
dataset['first_case_ConfirmedCases'] = \
       np.log(first_case.loc[dataset.Place].ConfirmedCases.values + 1)
dataset['first_case_Fatalities'] = \
       np.log(first_case.loc[dataset.Place].Fatalities.values + 1)

# %% [code]


# %% [code]
dataset['first_fatality_ConfirmedCases'] = \
       np.log(first_fatality.loc[dataset.Place].ConfirmedCases.fillna(0).values + 1) \
            * (dataset.days_since_first_fatality >= 0 )
dataset['first_fatality_Fatalities'] = \
       np.log(first_fatality.loc[dataset.Place].Fatalities.fillna(0).values + 1) \
            * (dataset.days_since_first_fatality >= 0 )

# %% [code]
dataset['first_fatality_cfr'] = \
    np.where(dataset.days_since_first_fatality < 0,
            -8,
        (dataset.first_fatality_Fatalities) -
               (dataset.first_fatality_ConfirmedCases )   )

# %% [code]
dataset['first_fatality_lag_vs_first_case'] = \
    np.where(dataset.days_since_first_fatality >= 0,
                 dataset.days_since_first_case - dataset.days_since_first_fatality , -1)

# %% [code]


# %% [markdown]
# #### Update Frequency, MAs of Change Rates, etc.

# %% [code]
dataset['case_chg'] = \
    np.clip(np.log(dataset.ConfirmedCases + 1 )\
            - np.log(dataset.ConfirmedCases.shift(1) +1), 0, None).fillna(0)

# %% [code]
dataset['case_chg_ema_3d'] = dataset.case_chg.ewm(span = 3).mean() \
                                * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1)
dataset['case_chg_ema_10d'] = dataset.case_chg.ewm(span = 10).mean() \
                             * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1)

# %% [code]
dataset['case_chg_stdev_5d'] = dataset.case_chg.rolling(5).std() \
                                * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/5, 0, 1)
dataset['case_chg_stdev_15d'] = dataset.case_chg.rolling(15).std() \
                                * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/15, 0, 1)

# %% [raw]
# dataset['max_case_chg_3d'] = dataset.case_chg.rolling(3).max() \
#                                  * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1)
# dataset['max_case_chg_10d'] = dataset.case_chg.rolling(10).max() \
#                                  * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1)

# %% [code]
dataset['case_update_pct_3d_ewm'] = (dataset.case_chg > 0).ewm(span = 3).mean() \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
dataset['case_update_pct_10d_ewm'] = (dataset.case_chg > 0).ewm(span = 10).mean() \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1), 2)
dataset['case_update_pct_30d_ewm'] = (dataset.case_chg > 0).ewm(span = 30).mean() \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/30, 0, 1), 2)

 

# %% [code]


# %% [code]
dataset['fatality_chg'] = \
    np.clip(np.log(dataset.Fatalities + 1 )\
            - np.log(dataset.Fatalities.shift(1) +1), 0, None).fillna(0)

# %% [code]
dataset['fatality_chg_ema_3d'] = dataset.fatality_chg.ewm(span = 3).mean() \
                    * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/33, 0, 1)
dataset['fatality_chg_ema_10d'] = dataset.fatality_chg.ewm(span = 10).mean() \
                    * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1)

# %% [code]
dataset['fatality_chg_stdev_5d'] = dataset.fatality_chg.rolling(5).std() \
                                * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/5, 0, 1)
dataset['fatality_chg_stdev_15d'] = dataset.fatality_chg.rolling(15).std() \
                                * np.clip( (dataset.Date - dataset.Date.min() ).dt.days/15, 0, 1)

# %% [code]
dataset['fatality_update_pct_3d_ewm'] = (dataset.fatality_chg > 0).ewm(span = 3).mean() \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
dataset['fatality_update_pct_10d_ewm'] = (dataset.fatality_chg > 0).ewm(span = 10).mean() \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/10, 0, 1), 2)
dataset['fatality_update_pct_30d_ewm'] = (dataset.fatality_chg > 0).ewm(span = 30).mean() \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/30, 0, 1), 2)

# %% [code]


# %% [code]


# %% [code]
dataset.tail()

# %% [code]


# %% [markdown]
# #### Add Supp Data

# %% [code]
# lag containment data as one week behind
contain_data.Date = contain_data.Date + datetime.timedelta(7)

# %% [code]
contain_data.Date.max()

# %% [code]
assert set(dataset.Place.unique()) == set(dataset.Place.unique())
dataset = pd.merge(dataset, sup_data, on='Place', how='left', validate='m:1')
dataset = pd.merge(dataset, contain_data, on = ['Country', 'Date'], how='left', validate='m:1')

# %% [code]
dataset['log_true_population'] =   np.log(dataset.TRUE_POPULATION + 1)

# %% [code]
dataset['ConfirmedCases_percapita'] = np.log(dataset.ConfirmedCases + 1)\
                                        - np.log(dataset.TRUE_POPULATION + 1)
dataset['Fatalities_percapita'] = np.log(dataset.Fatalities + 1)\
                                        - np.log(dataset.TRUE_POPULATION + 1)

# %% [code]


# %% [markdown]
# ##### CFR

# %% [raw]
# np.log( 0 + 0.015/1)

# %% [raw]
# BLCFR = -4.295015257684252

# %% [code] {"scrolled":true}
# dataset['log_cfr_bad'] = np.log(dataset.Fatalities + 1) - np.log(dataset.ConfirmedCases + 1)
dataset['log_cfr'] = np.log(    (dataset.Fatalities \
                                         + np.clip(0.015 * dataset.ConfirmedCases, 0, 0.3)) \
                            / ( dataset.ConfirmedCases + 0.1) )

# %% [code]
def cfr(case, fatality):
    cfr_calc = np.log(    (fatality \
                                         + np.clip(0.015 * case, 0, 0.3)) \
                            / ( case + 0.1) )
#     cfr_calc =np.array(cfr_calc)
    return np.where(np.isnan(cfr_calc) | np.isinf(cfr_calc),
                           BLCFR, cfr_calc)

# %% [code]
BLCFR = np.median(dataset[dataset.ConfirmedCases==1].log_cfr[::10])
dataset.log_cfr.fillna(BLCFR, inplace=True)
dataset.log_cfr = np.where(dataset.log_cfr.isnull() | np.isinf(dataset.log_cfr),
                           BLCFR, dataset.log_cfr)
BLCFR

# %% [code]
dataset['log_cfr_3d_ewm'] = BLCFR + \
                (dataset.log_cfr - BLCFR).ewm(span = 3).mean()  \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/3, 0, 1), 2)
                     
dataset['log_cfr_8d_ewm'] = BLCFR + \
                (dataset.log_cfr - BLCFR).ewm(span = 8).mean()  \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/8, 0, 1), 2)

dataset['log_cfr_20d_ewm'] = BLCFR + \
                (dataset.log_cfr - BLCFR).ewm(span = 20).mean()  \
                     * np.power(np.clip( (dataset.Date - dataset.Date.min() ).dt.days/20, 0, 1), 2)

dataset['log_cfr_3d_20d_ewm_crossover'] = dataset.log_cfr_3d_ewm - dataset.log_cfr_20d_ewm


# %% [code]
dataset.drop(columns = 'log_cfr', inplace=True)



# %% [code]


# %% [markdown]
# ##### Per Capita vs. World and Similar Countries

# %% [code]
date_totals = dataset.groupby('Date').sum()

# %% [code]
mean_7d_c_slope = dataset.groupby('Date')[['ConfirmedCases_7d_prior_slope']].apply(lambda x:
                                        np.mean(x[x > 0]) ).ewm(span = 3).mean() 
mean_7d_f_slope = dataset.groupby('Date')[['Fatalities_7d_prior_slope']].apply(lambda x:
                                        np.mean(x[x > 0]) ).ewm(span = 7).mean()

# %% [raw]
# mean_7d_c_slope.plot()

# %% [raw]
# dataset.columns[:100]

# %% [raw]
# mean_7d_c_slope.plot()

# %% [raw]
# date_totals.Fatalities_7d_prior_slope.plot()

# %% [raw]
# date_counts = dataset.groupby('Date').apply(lambda x:  x > 0)

# %% [raw]
# date_counts

# %% [raw]
# date_totals['world_cases_chg'] = (np.log(date_totals.ConfirmedCases + 1 )\
#                                     - np.log(date_totals.ConfirmedCases.shift(1) + 1) )\
#                                     .fillna(method='bfill')
# date_totals['world_fatalities_chg'] = (np.log(date_totals.Fatalities + 1 )\
#                                     - np.log(date_totals.Fatalities.shift(1) + 1) )\
#                                     .fillna(method='bfill')
# date_totals['world_cases_chg_10d_ewm'] = \
#         date_totals.world_cases_chg.ewm(span=10).mean()
# date_totals['world_fatalities_chg_10d_ewm'] = \
#         date_totals.world_fatalities_chg.ewm(span=10).mean()  

# %% [raw]
# 
# dataset['world_cases_chg_10d_ewm'] = \
#         date_totals.loc[dataset.Date].world_cases_chg_10d_ewm.values
# 
# dataset['world_fatalities_chg_10d_ewm'] = \
#         date_totals.loc[dataset.Date].world_fatalities_chg_10d_ewm.values
# 

# %% [raw]
# dataset.continent

# %% [raw]
# date_totals

# %% [code]
dataset['ConfirmedCases_percapita_vs_world'] = np.log(dataset.ConfirmedCases + 1)\
                                        - np.log(dataset.TRUE_POPULATION + 1) \
                                   -  (
                                   np.log(date_totals.loc[dataset.Date].ConfirmedCases + 1)  
                                       -np.log(date_totals.loc[dataset.Date].TRUE_POPULATION + 1)
                                        ).values

dataset['Fatalities_percapita_vs_world'] = np.log(dataset.Fatalities + 1)\
                                            - np.log(dataset.TRUE_POPULATION + 1) \
                                    -  (
                                   np.log(date_totals.loc[dataset.Date].Fatalities + 1)  
                                       -np.log(date_totals.loc[dataset.Date].TRUE_POPULATION + 1)
                                        ).values
dataset['cfr_vs_world'] = dataset.log_cfr_3d_ewm \
                            -    np.log(    date_totals.loc[dataset.Date].Fatalities   \
                            /   date_totals.loc[dataset.Date].ConfirmedCases ).values

# %% [code]


# %% [markdown]
# #### Nearby Countries

# %% [code]
cont_date_totals = dataset.groupby(['Date', 'continent_generosity']).sum()

# %% [raw]
# cont_date_totals.iloc[dataset.Date]

# %% [code]
len(dataset)

# %% [raw]
# dataset.columns

# %% [raw]
# dataset.TRUE_POPULATION

# %% [raw]
# dataset

# %% [raw]
# dataset

# %% [code]
dataset['ConfirmedCases_percapita_vs_continent_mean'] = 0
dataset['Fatalities_percapita_vs_continent_mean'] = 0
dataset['ConfirmedCases_percapita_vs_continent_median'] = 0
dataset['Fatalities_percapita_vs_continent_median'] = 0

for cg in dataset.continent_generosity.unique():
    ps = dataset.groupby("Place").last()
    tp = ps[ps.continent_generosity==cg].TRUE_POPULATION.sum()
    print(tp / 1e9)
    for Date in dataset.Date.unique():
        cd =  dataset[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg)]\
                               [['ConfirmedCases', 'Fatalities', 'TRUE_POPULATION']]
#         print(cd)
        cmedian = np.median(np.log(cd.ConfirmedCases + 1)\
                                              - np.log(cd.TRUE_POPULATION+1))
        cmean = np.log(cd.ConfirmedCases.sum() + 1) - np.log(tp + 1)
        fmedian = np.median(np.log(cd.Fatalities + 1)\
                                              - np.log(cd.TRUE_POPULATION+1))
        fmean = np.log(cd.Fatalities.sum() + 1) - np.log(tp + 1)
        cfrmean = cfr( cd.ConfirmedCases.sum(),  cd.Fatalities.sum()   ) 
#         print(cmean)
        
#         break;
        
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'ConfirmedCases_percapita_vs_continent_mean'] = \
                                dataset['ConfirmedCases_percapita'] \
                                     - (cmean)
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'ConfirmedCases_percapita_vs_continent_median'] = \
                                dataset['ConfirmedCases_percapita'] \
                                     - (cmedian)
        
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'Fatalities_percapita_vs_continent_mean'] = \
                                dataset['Fatalities_percapita']\
                                    - (fmean)
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'Fatalities_percapita_vs_continent_median'] = \
                                dataset['Fatalities_percapita']\
                                    - (fmedian)
        
        dataset.loc[(dataset.Date == Date) &
                    (dataset.continent_generosity == cg), 
                    'cfr_vs_continent'] = \
                                dataset.log_cfr_3d_ewm \
                            -    cfrmean
#       
#         r.ConfirmedCases
#         r.Fatalities
#         print(continent)
    

# %% [code]


# %% [raw]
# dataset[dataset.Country=='China'][['Place', 'Date', 
#                'ConfirmedCases_percapita_vs_continent_mean',
#                'Fatalities_percapita_vs_continent_mean']][1000::10]

# %% [raw]
# dataset[['Place', 'Date', 
#                'cfr_vs_continent']][10000::5]

# %% [code]


# %% [code]
all_places = dataset[['Place', 'latitude', 'longitude']].drop_duplicates().set_index('Place',
                                                                                    drop=True)
all_places.head()

# %% [code]
def surroundingPlaces(place, d = 10):
    dist = (all_places.latitude - all_places.loc[place].latitude)**2 \
                    + (all_places.longitude - all_places.loc[place].longitude) ** 2 
    return all_places[dist < d**2][1:n+1]

# %% [raw]
# surroundingPlaces('Afghanistan', 5)

# %% [code]
def nearestPlaces(place, n = 10):
    dist = (all_places.latitude - all_places.loc[place].latitude)**2 \
                    + (all_places.longitude - all_places.loc[place].longitude) ** 2
    ranked = np.argsort(dist) 
    return all_places.iloc[ranked][1:n+1]

# %% [code]


# %% [raw]
# dataset.ConfirmedCases_percapita

# %% [code]
dgp = dataset.groupby('Place').last()
for n in [5, 10, 20]:
#     dataset['ConfirmedCases_percapita_vs_nearest{}'.format(n)] = 0
#     dataset['Fatalities_percapita_vs_nearest{}'.format(n)] = 0
    
    for place in dataset.Place.unique():
        nps = nearestPlaces(place, n)
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()
#         print(tp)
        
        
        dataset.loc[dataset.Place==place, 
                    'ratio_population_vs_nearest{}'.format(n)] = \
            np.log(dataset.loc[dataset.Place==place].TRUE_POPULATION.mean() + 1)\
                - np.log(tp+1)
         
#         dataset.loc[dataset.Place==place, 
#                     'avg_distance_to_nearest{}'.format(n)] = \
#             (dataset.loc[dataset.Place==place].latitude.mean() + 1)\
#                 - np.log(tp+1)
        

        nbps =  dataset[(dataset.Place.isin(nps.index))]\
                            .groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

        nppc = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities + 1) - np.log(tp + 1))
        npp_cfr = cfr( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases,
                      nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities)
#         print(npp_cfr)
#         continue;
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_percapita_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].ConfirmedCases_percapita \
                            - nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_percapita_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].Fatalities_percapita \
                            - nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_vs_nearest{}'.format(n)] = \
            dataset[(dataset.Place == place)].log_cfr_3d_ewm \
                            - npp_cfr   
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_nearest{}_percapita'.format(n)] = nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_nearest{}_percapita'.format(n)] = nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_nearest{}'.format(n)] = npp_cfr
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_nearest{}_10d_slope'.format(n)] =   \
                               ( nppc.ewm(span = 1).mean() - nppc.ewm(span = 10).mean() ).values
        dataset.loc[
                (dataset.Place == place),
                    'Fatalities_nearest{}_10d_slope'.format(n)] =   \
                               ( nppf.ewm(span = 1).mean() - nppf.ewm(span = 10).mean() ).values
        
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_nearest{}_10d_slope'.format(n)] = \
                            ( npp_cfr_s.ewm(span = 1).mean()\
                                     - npp_cfr_s.ewm(span = 10).mean() ) .values
        
#         print(( npp_cfr_s.ewm(span = 1).mean()\
#                                      - npp_cfr_s.ewm(span = 10).mean() ).values)
        

# %% [code]


# %% [code]
dgp = dataset.groupby('Place').last()
for d in [5, 10, 20]:
#     dataset['ConfirmedCases_percapita_vs_nearest{}'.format(n)] = 0
#     dataset['Fatalities_percapita_vs_nearest{}'.format(n)] = 0
    
    for place in dataset.Place.unique():
        nps = surroundingPlaces(place, d)
        dataset.loc[dataset.Place==place, 'num_surrounding_places_{}_degrees'.format(d)] = \
            len(nps)
        
        
        tp = dgp.loc[nps.index].TRUE_POPULATION.sum()
        
        dataset.loc[dataset.Place==place, 
                    'ratio_population_vs_surrounding_places_{}_degrees'.format(d)] = \
            np.log(dataset.loc[dataset.Place==place].TRUE_POPULATION.mean() + 1)\
                - np.log(tp+1)
        
        if len(nps)==0:
            continue;
            
#         print(place)
#         print(nps)
#         print(tp)
        nbps =  dataset[(dataset.Place.isin(nps.index))]\
                            .groupby('Date')[['ConfirmedCases', 'Fatalities']].sum()

#         print(nbps)
        nppc = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases + 1) - np.log(tp + 1))
        nppf = (np.log( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities + 1) - np.log(tp + 1))
#         break;
        npp_cfr = cfr( nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).ConfirmedCases,
                      nbps.loc[dataset[dataset.Place==place].Date]\
                                          .fillna(0).Fatalities)
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_percapita_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].ConfirmedCases_percapita \
                            - nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_percapita_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].Fatalities_percapita \
                            - nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_vs_surrounding_places_{}_degrees'.format(d)] = \
            dataset[(dataset.Place == place)].log_cfr_3d_ewm \
                            - npp_cfr   
        
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_surrounding_places_{}_degrees_percapita'.format(d)] = nppc.values
        dataset.loc[ 
                (dataset.Place == place),
                    'Fatalities_surrounding_places_{}_degrees_percapita'.format(d)] = nppf.values
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_surrounding_places_{}_degrees'.format(d)] = npp_cfr
        
        dataset.loc[
                (dataset.Place == place),
                    'ConfirmedCases_surrounding_places_{}_degrees_10d_slope'.format(d)] =   \
                               ( nppc.ewm(span = 1).mean() - nppc.ewm(span = 10).mean() ).values
        dataset.loc[
                (dataset.Place == place),
                    'Fatalities_surrounding_places_{}_degrees_10d_slope'.format(d)] =   \
                               ( nppf.ewm(span = 1).mean() - nppf.ewm(span = 10).mean() ).values
        npp_cfr_s = pd.Series(npp_cfr)
        dataset.loc[ 
                (dataset.Place == place),
                    'cfr_surrounding_places_{}_degrees_10d_slope'.format(d)] = \
                            ( npp_cfr_s.ewm(span = 1).mean()\
                                     - npp_cfr_s.ewm(span = 10).mean() ) .values
        

# %% [code]


# %% [code]
for col in [c for c in dataset.columns if 'surrounding_places' in c and 'num_sur' not in c]:
    dataset[col] = dataset[col].fillna(0)
    n_col = 'num_surrounding_places_{}_degrees'.format(col.split('degrees')[0]\
                                                           .split('_')[-2])

    print(col)
#     print(n_col)
    dataset[col + "_times_num_places"] = dataset[col] * np.sqrt(dataset[n_col])
#     print('num_surrounding_places_{}_degrees'.format(col.split('degrees')[0][-2:-1]))

# %% [code]
dataset[dataset.Country=='US'][['Place', 'Date'] \
                                     + [c for c in dataset.columns if 'ratio_p' in c]]\
                [::50]

# %% [code]


# %% [raw]
# dataset[dataset.Country=="US"].groupby('Place').last()\
#         [[c for c in dataset.columns if 'cfr' in c]].iloc[:10, 8:]

# %% [code]


# %% [raw]
# dataset[dataset.Place=='USAlabama'][['Place', 'Date'] \
#                                      + [c for c in dataset.columns if 'places_5_degree' in c]]\
#                [40::5]

# %% [code]


# %% [code]
dataset.TRUE_POPULATION

# %% [code]
dataset.TRUE_POPULATION.sum()

# %% [code]
dataset.groupby('Date').sum().TRUE_POPULATION

# %% [code]


# %% [raw]
# dataset[dataset.ConfirmedCases>0]['log_cfr'].plot(kind='hist', bins = 250)

# %% [raw]
# dataset.log_cfr.isnull().sum()

# %% [code]
dataset['first_case_ConfirmedCases_percapita'] = \
       np.log(dataset.first_case_ConfirmedCases + 1) \
          - np.log(dataset.TRUE_POPULATION + 1)

dataset['first_case_Fatalities_percapita'] = \
       np.log(dataset.first_case_Fatalities + 1) \
          - np.log(dataset.TRUE_POPULATION + 1)

dataset['first_fatality_Fatalities_percapita'] = \
       np.log(dataset.first_fatality_Fatalities + 1) \
          - np.log(dataset.TRUE_POPULATION + 1)

dataset['first_fatality_ConfirmedCases_percapita'] = \
        np.log(dataset.first_fatality_ConfirmedCases + 1)\
            - np.log(dataset.TRUE_POPULATION + 1)

# %% [code]


# %% [code]
 
dataset['days_to_saturation_ConfirmedCases_4d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_4d_prior_slope         
dataset['days_to_saturation_ConfirmedCases_7d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_7d_prior_slope         

    
dataset['days_to_saturation_Fatalities_20d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_20d_prior_slope         
dataset['days_to_saturation_Fatalities_12d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1)) \
                            / dataset.ConfirmedCases_12d_prior_slope         
 

# %% [code]
dataset['days_to_3pct_ConfirmedCases_4d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 3.5) \
                            / dataset.ConfirmedCases_4d_prior_slope         
dataset['days_to_3pct_ConfirmedCases_7d'] = \
                                ( - np.log(dataset.ConfirmedCases + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 3.5) \
                            / dataset.ConfirmedCases_7d_prior_slope         

    
dataset['days_to_0.3pct_Fatalities_20d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 5.8) \
                            / dataset.ConfirmedCases_20d_prior_slope         
dataset['days_to_0.3pct_Fatalities_12d_cases'] = \
                                ( - np.log(dataset.Fatalities + 1)\
                                        + np.log(dataset.TRUE_POPULATION + 1) - 5.8) \
                            / dataset.ConfirmedCases_12d_prior_slope         
 

# %% [code]


# %% [raw]
# 

# %% [code]


# %% [code]
dataset.tail()

# %% [code]


# %% [markdown]
# ### Build Intervals into Future

# %% [code]


# %% [code]


# %% [code]
dataset = dataset[dataset.ConfirmedCases > 0]

len(dataset)

# %% [code]
datas = []
for window in range(1, 35):
    base = rollDates(dataset, window, True)
    datas.append(pd.merge(dataset[['Date', 'Place',
                 'ConfirmedCases', 'Fatalities']], base, on = ['Date', 'Place'],
                          how = 'right', 
            suffixes = ('_f', '')))
data = pd.concat(datas, axis =0).astype(np.float32, errors ='ignore')

# %% [code]
len(data)

# %% [raw]
# data[data.Place=='USNew York']

# %% [code]
data['Date_f'] = data.Date
data.Date = data.Date_i

# %% [code]
data['elapsed'] = (data.Date_f - data.Date_i).dt.days

# %% [code]
data['CaseChgRate'] = (np.log(data.ConfirmedCases_f + 1) - np.log(data.ConfirmedCases + 1))\
                            / data.elapsed;
data['FatalityChgRate'] = (np.log(data.Fatalities_f + 1) - np.log(data.Fatalities + 1))\
                            / data.elapsed;


# %% [code]


# %% [code]
data.elapsed

# %% [code]


# %% [raw]
# data[slope_cols]

# %% [raw]
# [c for c in data.columns if any(z in c for z in [ 'rate']) ]

# %% [code]
falloff_hash = {}

# %% [code]


# %% [code]
def true_agg(rate_i, elapsed, bend_rate):
#     print(elapsed); 
    elapsed = int(elapsed)
#     ar = 0
#     rate = rate_i
#     for i in range(0, elapsed):
#         rate *= bend_rate
#         ar += rate
#     return ar

    if (bend_rate, elapsed) not in falloff_hash:
        falloff_hash[(bend_rate, elapsed)] = \
            np.sum( [  np.power(bend_rate, e) for e in range(1, elapsed+1)] )
    return falloff_hash[(bend_rate, elapsed)] * rate_i
     

# %% [code]
true_agg(0.3, 30, 0.9)

# %% [raw]
# %timeit true_agg(0.3, 30, 0.9)

# %% [code]
slope_cols = [c for c in data.columns if 
                      any(z in c for z in ['prior_slope', 'chg', 'rate'])
           and not any(z in c for z in ['bend', 'prior_slope_chg', 'Country', 'ewm', 
                                        ]) ] # ** bid change; since rate too stationary
print(slope_cols)
bend_rates = [1, 0.95, 0.90]
for bend_rate in bend_rates:
    bend_agg = data[['elapsed']].apply(lambda x: true_agg(1, *x, bend_rate), axis=1)
     
    for sc in slope_cols:
        if bend_rate < 1:
            data[sc+"_slope_bend_{}".format(bend_rate)] =  data[sc]  \
                                    * np.power((bend_rate + 1)/2, data.elapsed)
         
            data[sc+"_true_slope_bend_{}".format(bend_rate)] = \
                          bend_agg *  data[sc] / data.elapsed
            
        data[sc+"_agg_bend_{}".format(bend_rate)] =  data[sc] * data.elapsed \
                                * np.power((bend_rate + 1)/2, data.elapsed)
         
        data[sc+"_true_agg_bend_{}".format(bend_rate)] = \
                        bend_agg *  data[sc]
#                       data[[sc, 'elapsed']].apply(lambda x: true_agg(*x, bend_rate), axis=1) 
        
         
#         print(data[sc+"_true_agg_bend_{}".format(bend_rate)])

# %% [raw]
# data[[c for c in data.columns if 'Fatalities_7d_prior_slope' in c and 'true_agg' in c]]

# %% [code]


# %% [code]


# %% [raw]
# data[data.Place=='USNew York'][['elapsed'] +[c for c in data.columns if 'ses_4d_prior_slope' in c]]

# %% [code]
slope_cols[:5]

# %% [raw]
# data

# %% [code]
for col in [c for c in data.columns if any(z in c for z in 
                               ['vs_continent', 'nearest', 'vs_world', 'surrounding_places'])]:
#     print(col)
    data[col + '_times_days'] = data[col] * data.elapsed

# %% [code]
data['saturation_slope_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1)) \
                                                    / data.elapsed
data['saturation_slope_Fatalities'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1)) \
                                                    / data.elapsed

data['dist_to_ConfirmedCases_saturation_times_days'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1)) \
                                                    * data.elapsed
data['dist_to_Fatalities_saturation_times_days'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1)) \
                                                    * data.elapsed
        


data['slope_to_1pct_ConfirmedCases'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1) - 4.6) \
                                                    / data.elapsed
data['slope_to_0.1pct_Fatalities'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1) - 6.9) \
                                                    / data.elapsed

data['dist_to_1pct_ConfirmedCases_times_days'] = (- np.log(data.ConfirmedCases + 1)\
                                                        + np.log(data.TRUE_POPULATION + 1) - 4.6) \
                                                    * data.elapsed
data['dist_to_0.1pct_Fatalities_times_days'] = (- np.log(data.Fatalities + 1)\
                                                + np.log(data.TRUE_POPULATION + 1) - 6.9) \
                                                    * data.elapsed

# %% [raw]
# data.ConfirmedCases_12d_prior_slope.plot(kind='hist')

# %% [code]
data['trendline_per_capita_ConfirmedCases_4d_slope'] = ( np.log(data.ConfirmedCases + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_4d_prior_slope * data.elapsed)
data['trendline_per_capita_ConfirmedCases_7d_slope'] = ( np.log(data.ConfirmedCases + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_7d_prior_slope * data.elapsed)
 

data['trendline_per_capita_Fatalities_12d_slope'] = ( np.log(data.Fatalities + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_12d_prior_slope * data.elapsed)
data['trendline_per_capita_Fatalities_20d_slope'] = ( np.log(data.Fatalities + 1)\
                                                        - np.log(data.TRUE_POPULATION + 1)) \
                                       + (data.ConfirmedCases_20d_prior_slope * data.elapsed)

 

# %% [code]


# %% [raw]
# data[data.Place == 'USNew York']

# %% [code]
len(data)

# %% [raw]
# data.CaseChgRate.plot(kind='hist', bins = 250);

# %% [code]


# %% [raw]
# data_bk = data.copy()

# %% [code]


# %% [code]
data.groupby('Place').last()

# %% [code]


# %% [code]


# %% [raw]
# # data['log_days_since_first_case'] =  np.log(data.days_since_first_case + 1)
# # data['log_days_since_first_fatality'] = np.log(data.days_since_first_fatality + 1)
# 
# data['sqrt_days_since_first_case'] = np.sqrt(data.days_since_first_case)
# data['sqrt_days_since_first_fatality'] = np.sqrt(data.days_since_first_fatality)
# 
# 
# 

 
# %% [code]
def logHist(x, b = 150):
    return

# %% [raw]
# np.std(x.log_cases)

# %% [raw]
# np.std(x.log_fatalities)

# %% [code]


# %% [code]
data['log_fatalities'] = np.log(data.Fatalities + 1) #  + 0.4 * np.random.normal(0, 1, len(data))
data['log_cases'] = np.log(data.ConfirmedCases + 1) # + 0.2 *np.random.normal(0, 1, len(data))



# %% [raw]
# data.log_cases.plot(kind='hist', bins = 250)

# %% [code]
data['is_China'] = (data.Country=='China') & (~data.Place.isin(['Hong Kong', 'Macau']))

# %% [code]
for col in [c for c in data.columns if 'd_ewm' in c]:
    data[col] += np.random.normal(0, 1, len(data)) * np.std(data[col]) * 0.2
    

# %% [raw]
# data[data.log_cfr>-11].log_fatalities.plot(kind='hist', bins = 150)

# %% [code]
data['is_province'] = 1.0* (~data.Province_State.isnull() )

# %% [code]
data['log_elapsed'] = np.log(data.elapsed + 1)

# %% [code]
data.columns

# %% [code]
data.columns[::19]

# %% [code]
data.shape

# %% [code]
logHist(data.ConfirmedCases)

# %% [code]


# %% [code]


# %% [markdown]
# ### Data Cleanup

# %% [code]
data.drop(columns = ['TRUE_POPULATION'], inplace=True)

# %% [code]
data['final_day_of_week'] = data.Date_f.apply(datetime.datetime.weekday)

# %% [code]
data['base_date_day_of_week'] = data.Date.apply(datetime.datetime.weekday)

# %% [code]
data['date_difference_modulo_7_days'] = (data.Date_f - data.Date).dt.days % 7

# %% [raw]
# for c in data.columns.to_list():
#     if 'days_since' in c:
#         data[c] = np.log(data[c]+1)

# %% [code]


# %% [code]
for c in data.columns.to_list():
    if 'days_to' in c:
#         print(c)
        data[c] = data[c].where(~np.isinf(data[c]), 1e3)
        data[c] = np.clip(data[c], 0, 365)
        data[c] = np.sqrt(data[c])


        
        
new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &
      (train.ConfirmedCases == 0)
     ].Place

        
        
        # %% [code]


# %% [markdown]
# ## II. Modeling

# %% [markdown]
# ### Data Prep

# %% [code]
model_data = data[ (( len(test) ==0 ) | (data.Date_f < test.Date.min()))
                  & 
                  (data.ConfirmedCases > 0) &
                 (~data.ConfirmedCases_f.isnull())].copy()

# %% [raw]
# data.Date_f

# %% [code]
test.Date.min()

# %% [code]
model_data.Date_f.max()

# %% [code]
model_data.Date_f.max()

# %% [code]
model_data.Date.max()

# %% [code]
model_data.Date_f.min()

# %% [code]


# %% [code]
model_data = model_data[~( 
                            ( np.random.rand(len(model_data)) < 0.8 )  &
                          ( model_data.Country == 'China') &
                              (model_data.Date < datetime.datetime(2020, 2, 15)) )]

# %% [code]
x_dates = model_data[['Date_i', 'Date_f', 'Place']]

# %% [code]
x = model_data[    
    model_data.columns.to_list()[
            model_data.columns.to_list().index('ConfirmedCases_1d_prior_value'):]]\
            .drop(columns = ['Date_i', 'Date_f', 'CaseChgRate', 'FatalityChgRate'])

# %% [raw]
# x.columns

# %% [raw]
# x




test.Date

# %% [code]
if PRIVATE:
    data_test = data[ (data.Date_i == train.Date.max() ) & 
                     (data.Date_f.isin(test.Date.unique() ) ) ].copy()
else:
    data_test = data[ (data.Date_i == test.Date.min() - datetime.timedelta(1) ) & 
                     (data.Date_f.isin(test.Date.unique() ) ) ].copy()

# %% [code]
data_test.Date.unique()

# %% [code]
test.Date.unique()

# %% [raw]
# data_test.Date_f

# %% [code]
x_test =  data_test[x.columns].copy()

# %% [code]
train.Date.max()

# %% [code]
test.Date.max()

# %% [raw]
# data_test[data_test.Place=='San Marino'].Date_f

# %% [raw]
# data_test.groupby('Place').Date_f.count().sort_values()

# %% [raw]
# x_test

# %% [code]


# %% [raw]
# x.columns

# %% [code]


# %% [code]
if MODEL_Y is 'slope':
    y_cases = model_data.CaseChgRate 
    y_fatalities = model_data.FatalityChgRate 
else:
    y_cases = model_data.CaseChgRate * model_data.elapsed
    y_fatalities = model_data.FatalityChgRate * model_data.elapsed
    
y_cfr = np.log(    (model_data.Fatalities_f \
                                         + np.clip(0.015 * model_data.ConfirmedCases_f, 0, 0.3)) \
                            / ( model_data.ConfirmedCases_f + 0.1) )

# %% [code]
groups = model_data.Country
places = model_data.Place

# %% [raw]
# y_cfr

# %% [code]


# %% [markdown]
# #### Model Setup

# %% [code]
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, PredefinedSplit
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import HuberRegressor, ElasticNet
import lightgbm as lgb


# %% [code]
np.random.seed(SEED)

# %% [code]
enet_params = { 'alpha': [   3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,  ],
                'l1_ratio': [  0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.97, 0.99 ]}

# %% [code]
et_params = {        'n_estimators': [50, 70, 100, 140],
                    'max_depth': [3, 5, 7, 8, 9, 10],
                      'min_samples_leaf': [30, 50, 70, 100, 130, 165, 200, 300, 600],
                     'max_features': [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                    'min_impurity_decrease': [0, 1e-5 ], #1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                    'bootstrap': [ True, False], # False is clearly worse          
                 #   'criterion': ['mae'],
                   }

# %% [code]
lgb_params = {
                'max_depth': [5, 12],
                'n_estimators': [ 100, 200, 300, 500],   # continuous
                'min_split_gain': [0, 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
                'min_child_samples': [ 7, 10, 14, 20, 30, 40, 70, 100, 200, 400, 700, 1000, 2000],
                'min_child_weight': [0], #, 1e-3],
                'num_leaves': [5, 10, 20, 30],
                'learning_rate': [0.05, 0.07, 0.1],   #, 0.1],       
                'colsample_bytree': [0.1, 0.2, 0.33, 0.5, 0.65, 0.8, 0.9], 
                'colsample_bynode':[0.1, 0.2, 0.33, 0.5, 0.65, 0.81],
                'reg_lambda': [1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000,   ],
                'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 30, 1000,], # 1, 10, 100, 1000, 10000],
                'subsample': [  0.8, 0.9, 1],
                'subsample_freq': [1],
                'max_bin': [ 7, 15, 31, 63, 127, 255],
  #               'extra_trees': [True, False],
#                 'boosting': ['gbdt', 'dart'],
    #     'subsample_for_bin': [200000, 500000],
               }    

# %% [code]
MSE = 'neg_mean_squared_error'
MAE = 'neg_mean_absolute_error'

# %% [code]


# %% [code]
def trainENet(x, y, groups, cv = 0, **kwargs):
    return trainModel(x, y, groups, 
                      clf = ElasticNet(normalize = True, selection = 'random', 
                                       max_iter = 3000),
                      params = enet_params, 
                      cv = cv, **kwargs)

# %% [code]
def trainETR(x, y, groups, cv = 0, n_jobs = 5,  **kwargs):
    clf = ExtraTreesRegressor(n_jobs = 1)
    params = et_params
    return trainModel(x, y, groups, clf, params, cv, n_jobs, **kwargs)

# %% [code]
def trainLGB(x, y, groups, cv = 0, n_jobs = 4, **kwargs):
    clf = lgb.LGBMRegressor(verbosity=-1, hist_pool_size = 1000,  
                      )
    params = lgb_params
    
    return trainModel(x, y, groups, clf, params, cv, n_jobs,  **kwargs)

# %% [code]
def trainModel(x, y, groups, clf, params, cv = 0, n_jobs = None, 
                   verbose=0, splits=None, **kwargs):
#     if cv is 0:
#         param_sets = list(ParameterSampler(params, n_iter=1))
#         clf = clf.set_params(**param_sets[0] )
#         if n_jobs is not None:
#             clf = clf.set_params(** {'n_jobs': n_jobs } )
#         f = clf.fit(x, y)
#         return clf 
#     else:
        if n_jobs is None:
            n_jobs = 4
        if np.random.rand() < 0.8: # all shuffle, don't want overfit models, just reasonable
            folds = GroupShuffleSplit(n_splits=4, 
                                                   test_size= 0.2 + 0.10 * np.random.rand())
        else:
            folds = GroupKFold(4)
        clf = RandomizedSearchCV(clf, params, 
                            cv=  folds, 
#                                  cv = GroupKFold(4),
                                 n_iter=12, 
                                verbose = 0, n_jobs = n_jobs, scoring = MSE)
        f = clf.fit(x, y, groups)
        #if verbose > 0:
        print(pd.DataFrame(clf.cv_results_['mean_test_score'])); print();  
     #   print(pd.DataFrame(clf.cv_results_).to_string()); print();  
        
        
        best = clf.best_estimator_;  print(best)
        print("Best Score: {}".format(np.round(clf.best_score_,4)))
        
        return best

# %% [code] {"scrolled":true}
np.mean(y_cases)

# %% [code]


# %% [code]
def getSparseColumns(x, verbose = 0):
    sc = []
    for c in x.columns.to_list():
        u = len(x[c].unique())
        if u > 10 and u < 0.01*len(x) :
            sc.append(c)
            if verbose > 0:
                print("{}: {}".format(c, u))

    return sc

# %% [code]
def noisify(x, noise = 0.1):
    x = x.copy()
   # cols = x.columns.to_list()
    cols = getSparseColumns(x)
    for c in cols:
        u = len(x[c].unique())
        if u > 50:
            x[c].values[:] = x[c].values + np.random.normal(0, noise, len(x)) * np.std(x[c])
    return x;

# %% [raw]
# cols = getSparseColumns(x)
# for c in cols:
#     u = len(x[c].unique())
#     if u > 50:
#         print("{}: {}".format(c, u)) #x[c].values[:] = x[c].values + np.random.normal(0, noise, len(x)) * np.std(x[c])
# # return x;

# %% [raw]
# [c for c in x.columns if any(z in c for z in 
#                                  ['prior_slope', 'prior_value'])]

# %% [raw]
# getSparseColumns(x, verbose = 0)

# %% [raw]
# x.columns[::19]

# %% [code]
def getMaxOverlap(row, df):
#     max_overlap_frac = 0

    df_place = df[df.Place == row.Place]
    if len(df_place)==0:
        return 0
#     print(df_place)
    overlap = \
        (np.clip( df_place.Date_f, None, row.Date_f) \
                            - np.clip( df_place.Date_i, row.Date_i, None) ).dt.days
    overlap = np.clip(overlap, 0, None)
    length = np.clip(  (df_place.Date_f - df_place.Date_i).dt.days, 
                        (row.Date_f - row.Date_i).days,  None)
#     print(overlap)
#     print(length)
#     print(overlap)
#     print(length)
    return np.amax(overlap / length) 
#     print(row)
#     print(df_place)
#     return
    
#     for i in range(0, len(df_place)):
#         selected = df_place.iloc[i]
#        # if row.Place == selected.Place:
#         overlap = (np.min((row.Date_f, selected.Date_f))\
#                      - np.max((row.Date_i, selected.Date_i )) ).days
#         overlap_frac = overlap / (selected.Date_f - selected.Date_i).days 
#         if overlap_frac > max_overlap_frac:
#             max_overlap_frac = overlap_frac
#     return max_overlap_frac
     

# %% [raw]
# 

# %% [code]
def getSampleWeight(x, groups):
 
    
    counter = Counter(groups)
    median_count = np.median( [counter[group] for group in groups.unique()])
#     print(median_count)
    c_count = [counter[group] for group in groups]
    
    e_decay = np.round(LT_DECAY_MIN + np.random.rand() * ( LT_DECAY_MAX - LT_DECAY_MIN), 1) 
    print("LT weight decay: {:.2f}".format(e_decay));
    ssr =  np.power(  1 / np.clip( c_count / median_count , 0.1,  30) , 
                        0.1 + np.random.rand() * 0.6) \
                /   np.power(x.elapsed / 3, e_decay) \
                    *  SET_FRAC * np.exp(  -    np.random.rand()  )
    
#     print(np.power(  1 / np.clip( c_count / median_count , 1,  10) , 
#                         0.1 + np.random.rand() * 0.3))
#     print(np.power(x.elapsed / 3, e_decay))
#     print(np.exp(  1.5 * (np.random.rand() - 0.5) ))
        
    # drop % of groups at random
    group_drop = dict([(group, np.random.rand() < 0.15) for group in groups.unique()])
    ssr = ssr * (  [ 1 -group_drop[group] for group in groups])
#     print(ssr[::171])
#     print(np.array([ 1 -group_drop[group] for group in groups]).sum() / len(groups))

#     pd.Series(ssr).plot(kind='hist', bins = 100)
    return ssr;

# %% [raw]
# group_drop = dict([(group, np.random.rand() < 0.20) for group in groups.unique()])
#      
# np.array([ 1 -group_drop[group] for group in groups]).sum() / len(groups)
# 

# %% [raw]
# [c for c in x.columns if 'continent' in c]

# %% [raw]
# x.columns[::10]

# %% [raw]
# x.shape

# %% [raw]
# contain_data.columns

# %% [code]
def runBags(x, y, groups, cv, bags = 3, model_type = trainLGB, 
            noise = 0.1, splits = None, weights = None, **kwargs):
    models = []
    for bag in range(bags):
        print("\nBAG {}".format(bag+1))
        
        x = x.copy()  # copy X to modify it with noise
        
        if DROPS:
            # drop 0-70% of the bend/slope/prior features, just for speed and model diversity
            for col in [c for c in x.columns if any(z in c for z in ['bend', 'slope', 'prior'])]:
                if np.random.rand() < np.sqrt(np.random.rand()) * 0.7:
                    x[col].values[:] = 0
            
        # 00% of the time drop all 'rate_since' features 
#         if np.random.rand() < 0.00:
#             print('dropping rate_since features')
#             for col in [c for c in x.columns if 'rate_since' in c]:    
#                 x[col].values[:] = 0
        
        # 20% of the time drop all 'world' features 
#         if np.random.rand() < 0.00:
#             print('dropping world features')
#             for col in [c for c in x.columns if 'world' in c]:    
#                 x[col].values[:] = 0
        
        # % of the time drop all 'nearest' features 
        if DROPS and (np.random.rand() < 0.30):
            print('dropping nearest features')
            for col in [c for c in x.columns if 'nearest' in c]:    
                x[col].values[:] = 0
        
        #  % of the time drop all 'surrounding_places' features 
        if DROPS and (np.random.rand() < 0.25):
            print('dropping \'surrounding places\' features')
            for col in [c for c in x.columns if 'surrounding_places' in c]:    
                x[col].values[:] = 0
        
        
        # 20% of the time drop all 'continent' features 
#         if np.random.rand() < 0.20:
#             print('dropping continent features')
#             for col in [c for c in x.columns if 'continent' in c]:    
#                 x[col].values[:] = 0
        
        # drop 0-50% of all features
#         if DROPS:
        col_drop_frac = np.sqrt(np.random.rand()) * 0.5
        for col in [c for c in x.columns if 'elapsed' not in c ]:
            if np.random.rand() < col_drop_frac:
                x[col].values[:] = 0

        
        x = noisify(x, noise)
        
        
        if DROPS and (np.random.rand() < SUP_DROP):
            print("Dropping supplemental country data")
            for col in x[[c for c in x.columns if c in sup_data.columns]]:  
                x[col].values[:] = 0
                
        if DROPS and (np.random.rand() < ACTIONS_DROP): 
            for col in x[[c for c in x.columns if c in contain_data.columns]]:  
                x[col].values[:] = 0
#             print(x.StringencyIndex_20d_ewm[::157])
        else:
            print("*using containment data")
            
        if np.random.rand() < 0.6: 
            x.S_data_days = 0
            
        ssr = getSampleWeight(x, groups)
        
        date_falloff = 0 + (1/30) * np.random.rand()
        if weights is not None:
            ssr = ssr * np.exp(-weights * date_falloff)
        
        ss = ( np.random.rand(len(y)) < ssr  )
        print("n={}".format(len(x[ss])))
        
        p1 =x.elapsed[ss].plot(kind='hist', bins = int(x.elapsed.max() - x.elapsed.min() + 1))
        p1 = plt.figure();
#         break
#        print(Counter(groups[ss]))
        print((ss).sum())
        models.append(model_type(x[ss], y[ss], groups[ss], cv,   **kwargs))
    return models

# %% [code]
x = x.astype(np.float32)

# %% [raw]
# x.elapsed

# %% [code]


# %% [code]
BAG_MULT = 1

# %% [code]


# %% [code]
x.shape

# %% [code]


# %% [code]
lgb_c_clfs = []; lgb_c_noise = []

# %% [code] {"scrolled":true}
date_weights =  np.abs((model_data.Date_f - test.Date.min()).dt.days) 

# %% [code]
for iteration in range(0, int(math.ceil(1.1 * BAGS))):
    for noise in [ 0.05, 0.1, 0.2, 0.3, 0.4  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < PLACE_FRACTION:
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
             
        
        lgb_c_clfs.extend(runBags(x, y_cases, 
                          cv_group, #groups
                          MSE, num_bags, trainLGB, verbose = 0, 
                                          noise = noise, weights = date_weights

                                 ))
        lgb_c_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

# %% [code]


# %% [raw]
# np.isinf(x).sum().sort_values()

# %% [code]


# %% [raw]
# enet_c_clfs = runBags(x, y_cases, groups, MSE, 1, trainENet, verbose = 1)

# %% [code]


# %% [code]
lgb_f_clfs = []; lgb_f_noise = []

# %% [code]
for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in [  0.5,  1, 2, 3,  ]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * int(np.ceil(np.sqrt(BAG_MULT)))
        if np.random.rand() < PLACE_FRACTION  :
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
            
   
        lgb_f_clfs.extend(runBags(x, y_fatalities, 
                                  cv_group, #places, # groups, 
                                  MSE, num_bags, trainLGB, 
                                  verbose = 0, noise = noise,
                                  weights = date_weights
                                 ))
        lgb_f_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

# %% [raw]
# lgb_f_noise = lgb_f_noise[0:3]
# lgb_f_clfs = lgb_f_clfs[0:3]

# %% [raw]
# lgb_f_noise = lgb_f_noise[2:]
# lgb_f_clfs = lgb_f_clfs[2:]

# %% [raw]
# et_f_clfs = runBags(x, y_fatalities, groups, MSE, 1, trainETR, verbose = 1)
# 
# 

# %% [raw]
# enet_f_clfs = runBags(x, y_fatalities, groups, MSE, 1, trainENet, verbose = 1)
# 
# 

# %% [raw]
# y_cfr.plot(kind='hist', bins = 250)

# %% [code]
lgb_cfr_clfs = []; lgb_cfr_noise = [];

# %% [code]
for iteration in range(0, int(np.ceil(np.sqrt(BAGS)))):
    for noise in [    0.4, 1, 2, 3]:
        print("\n---\n\nNoise of {}".format(noise));
        num_bags = 1 * BAG_MULT;
        if np.random.rand() < 0.5 * PLACE_FRACTION :
            cv_group = places
            print("CV by Place")
        else:
            cv_group = groups
            print("CV by Country")
 
        lgb_cfr_clfs.extend(runBags(x, y_cfr, 
                          cv_group, #groups
                          MSE, num_bags, trainLGB, verbose = 0, 
                                          noise = noise, 
                                          weights = date_weights

                                 ))
        lgb_cfr_noise.extend([noise] * num_bags)
        if SINGLE_MODEL:
            break;

# %% [raw]
# x_test

# %% [code]
lgb_cfr_clfs[0].predict(x_test)

# %% [raw]
# 

# %% [code]
# full sample, through 03/28 (avail on 3/30), lgb only: 0.0097 / 0.0036;   0.0092 / 0.0042
#                                                       

# %% [markdown]
# ##### Feature Importance

# %% [code]
def show_FI(model, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fis = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1][:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    

# %% [code]
def avg_FI(all_clfs, featNames, featCount):
    # 1. Sum
    clfs = []
    for clf_set in all_clfs:
        for clf in clf_set:
            clfs.append(clf);
    print("{} classifiers".format(len(clfs)))
    fi = np.zeros( (len(clfs), len(clfs[0].feature_importances_)) )
    for idx, clf in enumerate(clfs):
        fi[idx, :] = clf.feature_importances_
    avg_fi = np.mean(fi, axis = 0)

    # 2. Plot
    fis = avg_fi
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(fis)[::-1]#[:featCount]
    #print(indices)
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fis[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    
    return pd.Series(fis[indices], featNames[indices])

# %% [code]

def linear_FI_plot(fi, featNames, featCount):
   # show_FI_plot(model.feature_importances_, featNames, featCount)
    fig, ax = plt.subplots(figsize=(6, 5))
    indices = np.argsort(np.absolute(fi))[::-1]#[:featCount]
    g = sns.barplot(y=featNames[indices][:featCount],
                    x = fi[indices][:featCount] , orient='h' )
    g.set_xlabel("Relative importance")
    g.set_ylabel("Features")
    g.tick_params(labelsize=12)
    g.set_title( " feature importance")
    return pd.Series(fi[indices], featNames[indices])

# %% [code]


# %% [raw]
# fi_list = []
# for clf in enet_c_clfs:
#     fi = clf.coef_ * np.std(x, axis=0).values 
#     fi_list.append(fi)
# fis = np.mean(np.array(fi_list), axis = 0)
# fis = linear_FI_plot(fis, x.columns.values,25)

# %% [raw]
# lgb_c_clfs

# %% [code]
f = avg_FI([lgb_c_clfs], x.columns, 25)

# %% [code]
for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
             'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# %% [code]
f[:100:3]

# %% [code]
print("{}: {:.2f}".format('sup_data', 
                       f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
print("{}: {:.2f}".format('contain_data', 
                   f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))

# %% [raw]
# I used a very simple Week 2 model like many. For Week 3:
# 
# The right target is total change in the logged counts. This exactly mirrors the final evaluation metric, and using change rather than raw logged countes keeps it stationary.
# 
# These can be put into a regressor for all windows from 1-30 days; ideally lightgbm or xgboost.  
# 
# Cross-validation works well by place (country-level struggles to understand China's magic numbers; time-series would be too 1-2 week centric and couldn't be done for a full month). Each place is it's own outbreak so this works reasonably well.
# 
# Feature Importance:
# ~30-40%: current and past outbreak information (slopes and rates calculated *many* ways)
# ~20-30%: nearby outbreak information, e.g. per capita rates vs. nearest 5, 10, 20 regions or within a specified latitude and longitude range--indicates not just spread but propensity to be tracking and reporting, severity, gov't management, likelihood of flattening, etc.
# ~10-20%: place attributes (average age, personality, tfr, percent in largest city, etc)
# ~10%: comparisons with world or continent, typically per capita prevalance compared with world or continent figures
# ~5%: containment actions taken 
# ~5%: other 
# 
# The models started to get good once I put in world and continent and then proximity information--this 'state of the world' information gives a clue to where the country is compared to others and its likely pace that may mirror recent trends for similar countries. 
# 
# It might be possible to get better 1-10 day figures with time series models, but a lot of the error is in long-term drift, so these 1-30 day interval total aggregate change models are best suited to the competition overall.
# 
# 

# %% [code]
f = avg_FI([lgb_f_clfs], x.columns, 25)

# %% [code]
for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
            'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# %% [code]
print("{}: {:.2f}".format('sup_data', 
                       f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
print("{}: {:.2f}".format('contain_data', 
                   f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))


# %% [raw]
# x.days_since_Stringency_1.plot(kind='hist', bins = 100)

# %% [raw]
# len(x.log_fatalities.unique())

# %% [code]
f = avg_FI([lgb_cfr_clfs], x.columns, 25)

# %% [code]
for feat in ['bend', 'capita', 'cfr', 'slope', 'since', 'chg', 'ersonal', 
            'world', 'continent', 'nearest', 'surrounding']:
    print("{}: {:.2f}".format(feat, f.filter(like=feat).sum() / f.sum()))

# %% [code]
print("{}: {:.2f}".format('sup_data', 
                       f[[c for c in f.index if c in sup_data.columns]].sum() / f.sum()))
print("{}: {:.2f}".format('contain_data', 
                   f[[c for c in f.index if c in contain_data.columns]].sum() / f.sum()))




# %% [code]
all_c_clfs = [lgb_c_clfs, ]#  enet_c_clfs]
all_f_clfs = [lgb_f_clfs] #, enet_f_clfs]
all_cfr_clfs = [lgb_cfr_clfs]


# %% [code]
all_c_noise = [lgb_c_noise]
all_f_noise = [lgb_f_noise]
all_cfr_noise = [lgb_cfr_noise]

# %% [code]


# %% [code]
NUM_TEST_RUNS = 1

# %% [code]
c_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_c_clfs]), len(x_test)))
f_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_f_clfs]), len(x_test)))
cfr_preds = np.zeros((NUM_TEST_RUNS * sum([len(x) for x in all_cfr_clfs]), len(x_test)))


# %% [code]
def avg(x):
    return (np.mean(x, axis=0) + np.median(x, axis=0))/2

# %% [code]
count = 0

for idx, clf in enumerate(lgb_c_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_c_noise[idx]
        c_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -1 , 10)
        count += 1
#y_cases_pred_blended_full = avg(c_preds)

# %% [code]
count = 0

for idx, clf in enumerate(lgb_f_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_f_noise[idx]
        f_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -1 , 10)
        count += 1
#y_fatalities_pred_blended_full = avg(f_preds)

# %% [code]
count = 0

for idx, clf in enumerate(lgb_cfr_clfs):
    for i in range(0, NUM_TEST_RUNS):
        noise = lgb_cfr_noise[idx]
        cfr_preds[count,:] = np.clip(clf.predict(noisify(x_test, noise)), -10 , 10)
        count += 1
#y_cfr_pred_blended_full = avg(cfr_preds)

# %% [code]


# %% [code]


# %% [code]
def qPred(preds, pctile, simple=False):
    q = np.percentile(preds, pctile, axis = 0)
    if simple:
        return q;
    resid = preds - q
    resid_wtg = 2/100/len(preds)* ( np.clip(resid, 0, None) * (pctile) \
                        + np.clip(resid, None, 0) * (100- pctile) )
    adj = np.sum(resid_wtg, axis = 0)
#     print(q)
#     print(adj)
#     print(q+adj)
    return q + adj

# %% [code]


# %% [code]
q = 50

# %% [code]
y_cases_pred_blended_full = qPred(c_preds, q) #avg(c_preds)
y_fatalities_pred_blended_full = qPred(f_preds, q) # avg(f_preds)
y_cfr_pred_blended_full = qPred(cfr_preds, q) #avg(cfr_preds)

# %% [code]


# %% [raw]
# cfr_preds

# %% [raw]
# lgb_cfr_noise

# %% [raw]
# lgb_cfr_clfs[0].predict(noisify(x_test, 0.4))

# %% [raw]
# cfr_preds[0][0:500]

# %% [raw]
# x.log_cfr.plot(kind='hist', bins = 250)

# %% [code]


# %% [code]
print(np.mean(np.corrcoef(c_preds[::NUM_TEST_RUNS]),axis=0))

# %% [code]
print(np.mean(np.corrcoef(f_preds[::NUM_TEST_RUNS]), axis=0))

# %% [code]
print(np.mean(np.corrcoef(cfr_preds[::NUM_TEST_RUNS]), axis = 0))

# %% [raw]
# cfr_preds

# %% [code]
pd.Series(np.std(c_preds, axis = 0)).plot(kind='hist', bins = 50)

# %% [code]
pd.Series(np.std(f_preds, axis = 0)).plot(kind='hist', bins = 50)

# %% [code]
pd.Series(np.std(cfr_preds, axis = 0)).plot(kind='hist', bins = 50)

# %% [code]
y_cfr

# %% [code]
(groups == 'Sierra Leone').sum()

# %% [code]
pred = pd.DataFrame(np.hstack((np.transpose(c_preds),
                              np.transpose(f_preds))), index=x_test.index)
pred['Place'] = data_test.Place


pred['Date'] = data_test.Date
pred['Date_f'] = data_test.Date_f

# %% [code]
pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][30: 60]

# %% [code]
(pred.Place=='Sierra Leone').sum()

# %% [code]
np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())], 2)[190:220:]

# %% [code] {"scrolled":false}
np.round(pred[(pred.Date == pred.Date.max()) & (pred.Date_f == pred.Date_f.max())][220:-20],2)

# %% [code]
c_preds.shape
x_test.shape

# %% [raw]
# 
# data_test.shape

# %% [raw]
# pd.DataFrame({'c_mean': np.mean(c_preds, axis =0 ),
#                   'c_median': np.median(c_preds, axis =0 ),
#              }, index=data_test.Place)[::7]

# %% [raw]
# np.median(c_preds, axis =0 )[::71]

# %% [code]


# %% [markdown]
# ### III. Other

# %% [code]


# %% [raw]
# MAX_DATE = np.max(train.Date)

# %% [raw]
# final = train[train.Date == MAX_DATE]

# %% [code]


# %% [code]


# %% [raw]
# train.groupby('Place')[['ConfirmedCases','Fatalities']].apply(lambda x: np.sum(x >0))

# %% [raw]
# num_changes = train.groupby('Place')[['ConfirmedCases','Fatalities']].apply(lambda x: np.sum(x - x.shift(1) >0))

# %% [raw]
# num_changes.Fatalities.plot(kind='hist', bins = 50);

# %% [raw]
# num_changes.ConfirmedCases.plot(kind='hist', bins = 50);

# %% [code]


# %% [code]


# %% [markdown]
# ### Rate Calculation

# %% [raw]
# def getRate(train, window = 5):
#     joined = pd.merge(train[train.Date == 
#                                     np.max(train.Date) - datetime.timedelta(window)], 
#                       final,  on=['Place'])
#     joined['FatalityRate'] = (np.log(joined.Fatalities_y + 1)\
#                                   - np.log(joined.Fatalities_x + 1)) / window
#     joined['CasesRate'] = (np.log(joined.ConfirmedCases_y + 1)\
#                                    - np.log(joined.ConfirmedCases_x + 1)) / window
#     joined.set_index('Place', inplace=True)
# 
#     rates = joined[[c for c in joined.columns.to_list() if 'Rate' in c]] 
#     return rates

# %% [raw]
# ltr = getRate(train, 14)

# %% [raw]
# lm = pd.merge(ltr, num_changes, on='Place')

# %% [raw]
# lm.filter(like='China', axis='rows')

# %% [raw]
# 

# %% [raw]
# flat = lm[
#     (lm.CasesRate < 0.01) & (lm.ConfirmedCases > 5)]

# %% [raw]
# flat

# %% [raw]
# 

# %% [raw]
# c_rate = pd.Series(
#     np.where(num_changes.ConfirmedCases >= 0, 
#          getRate(train, 7).CasesRate, 
#          getRate(train, 5).CasesRate),
#     index = num_changes.index, name = 'CasesRate')
# 
# f_rate = pd.Series(
#     np.where(num_changes.Fatalities >= 0, 
#          getRate(train, 7).FatalityRate, 
#          getRate(train, 4).CasesRate),
#     index = num_changes.index, name = 'FatalityRate')

# %% [code]


# %% [markdown]
# ### Plot of Changes

# %% [raw]
# def rollDates(df, i):
#     df = df.copy()
#     df.Date = df.Date + datetime.timedelta(i)
#     return df

# %% [raw]
# m = pd.merge(rollDates(train, 7), train, on=['Place', 'Date'])
# m['CaseChange'] = (np.log(m.ConfirmedCases_y + 1) - np.log(m.ConfirmedCases_x + 1))/7

# %% [raw]
# m[m.Place=='USMaine']

# %% [code]


# %% [markdown]
# #### Histograms of Case Counts

# %% [code]


# %% [raw]
# m = pd.merge(rollDates(full_train, 1), full_train, on=['Place', 'Date'])
# 

# %% [code]


# %% [markdown]
# ##### CFR Charts

# %% [raw]
# joined.Fatalities_y

# %% [raw]
# withcases = joined[joined.ConfirmedCases_y > 300]

# %% [raw]
# withcases.sort_values(by = ['Fatalities_y'])

# %% [raw]
# (withcases.Fatalities_y / withcases.ConfirmedCases_x).plot(kind='hist', bins = 150);

# %% [raw]
# (final.Fatalities / final.ConfirmedCases).plot(kind='hist', bins = 250);

# %% [code]


# %% [code]


# %% [markdown]
# ### Predict on Test Set

# %% [code]
data_wp = data_test.copy()

# %% [code]
if MODEL_Y is 'slope':
    data_wp['case_slope'] = y_cases_pred_blended_full 
    data_wp['fatality_slope'] = y_fatalities_pred_blended_full 
else:
    data_wp['case_slope'] = y_cases_pred_blended_full / x_test.elapsed
    data_wp['fatality_slope'] = y_fatalities_pred_blended_full / x_test.elapsed

data_wp['cfr_pred'] = y_cfr_pred_blended_full

# %% [raw]
# data_wp.head()

# %% [raw]
# data_wp.shape

# %% [raw]
# data_wp.Date_f.unique()

# %% [code]
train.Date.max()

# %% [raw]
# data_wp.Date

# %% [code]
test.Date.min()

# %% [raw]
# test

# %% [code]
if len(test) > 0:
    base_date = test.Date.min() - datetime.timedelta(1)
else:
    base_date = train.Date.max()

# %% [raw]
# train

# %% [raw]
# len(test)

# %% [code]
base_date

# %% [code]
data_wp_ss = data_wp[data_wp.Date == base_date]
data_wp_ss = data_wp_ss.drop(columns='Date').rename(columns = {'Date_f': 'Date'})

# %% [raw]
# base_date

# %% [raw]
# data_wp_ss.head()

# %% [raw]
# test

# %% [raw]
# data_wp_ss.columns

# %% [code]


# %% [raw]
# len(test);
# len(x_test)

# %% [code]
test_wp = pd.merge(test, data_wp_ss[['Date', 'Place', 'case_slope', 'fatality_slope', 'cfr_pred',
                                    'elapsed']], 
            how='left', on = ['Date', 'Place'])

# %% [raw]
# test_wp[test_wp.Country == 'US']

# %% [raw]
# test_wp

# %% [code]
first_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').first()
last_c_slope = test_wp[~test_wp.case_slope.isnull()].groupby('Place').last()

first_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').first()
last_f_slope = test_wp[~test_wp.fatality_slope.isnull()].groupby('Place').last()

first_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').first()
last_cfr_pred = test_wp[~test_wp.cfr_pred.isnull()].groupby('Place').last()

# %% [raw]
# test_wp

# %% [raw]
# first_c_slope

# %% [raw]
# test_wp

# %% [raw]
# test_wp

# %% [code]
test_wp.case_slope = np.where(  test_wp.case_slope.isnull() & 
                     (test_wp.Date < first_c_slope.loc[test_wp.Place].Date.values),
                   
                  first_c_slope.loc[test_wp.Place].case_slope.values,
                     test_wp.case_slope
                  )

test_wp.case_slope = np.where(  test_wp.case_slope.isnull() & 
                     (test_wp.Date > last_c_slope.loc[test_wp.Place].Date.values),
                   
                  last_c_slope.loc[test_wp.Place].case_slope.values,
                     test_wp.case_slope
                  )

# %% [code]
test_wp.fatality_slope = np.where(  test_wp.fatality_slope.isnull() & 
                     (test_wp.Date < first_f_slope.loc[test_wp.Place].Date.values),
                   
                  first_f_slope.loc[test_wp.Place].fatality_slope.values,
                     test_wp.fatality_slope
                  )

test_wp.fatality_slope = np.where(  test_wp.fatality_slope.isnull() & 
                     (test_wp.Date > last_f_slope.loc[test_wp.Place].Date.values),
                   
                  last_f_slope.loc[test_wp.Place].fatality_slope.values,
                     test_wp.fatality_slope
                  )

# %% [code]
test_wp.cfr_pred = np.where(  test_wp.cfr_pred.isnull() & 
                     (test_wp.Date < first_cfr_pred.loc[test_wp.Place].Date.values),
                   
                  first_cfr_pred.loc[test_wp.Place].cfr_pred.values,
                     test_wp.cfr_pred
                  )

test_wp.cfr_pred = np.where(  test_wp.cfr_pred.isnull() & 
                     (test_wp.Date > last_cfr_pred.loc[test_wp.Place].Date.values),
                   
                  last_cfr_pred.loc[test_wp.Place].cfr_pred.values,
                     test_wp.cfr_pred
                  )

# %% [code]


# %% [code]
test_wp.case_slope = test_wp.case_slope.interpolate('linear')
test_wp.fatality_slope = test_wp.fatality_slope.interpolate('linear')
test_wp.cfr_pred = test_wp.cfr_pred.interpolate('linear')

# %% [code]
test_wp.case_slope = test_wp.case_slope.fillna(0)
test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

# test_wp.fatality_slope = test_wp.fatality_slope.fillna(0)

# %% [raw]
# test_wp.cfr_pred.isnull().sum()

# %% [markdown]
# #### Convert Slopes to Aggregate Counts

# %% [code]
LAST_DATE = test.Date.min() - datetime.timedelta(1)

# %% [code]
final = train_bk[train_bk.Date == LAST_DATE  ]

# %% [raw]
# train

# %% [raw]
# final

# %% [code]
test_wp = pd.merge(test_wp, final[['Place', 'ConfirmedCases', 'Fatalities']], on='Place', 
                   how ='left', validate='m:1')

# %% [raw]
# test_wp

# %% [code]
LAST_DATE

# %% [raw]
# test_wp

# %% [code]
test_wp.ConfirmedCases = np.exp( 
                            np.log(test_wp.ConfirmedCases + 1) \
                                + test_wp.case_slope * 
                                   (test_wp.Date - LAST_DATE).dt.days )- 1

test_wp.Fatalities = np.exp(
                            np.log(test_wp.Fatalities + 1) \
                              + test_wp.fatality_slope * 
                                   (test_wp.Date - LAST_DATE).dt.days )  -1

# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1
                                     

# %% [code]
LAST_DATE

# %% [raw]
# final[final.Place=='Italy']

# %% [code]
test_wp[ (test_wp.Country == 'Italy')].groupby('Date').sum()[:10]


# %% [code]
test_wp[ (test_wp.Country == 'US')].groupby('Date').sum().iloc[-5:]


# %% [code]


# %% [markdown]
# ### Final Merge

# %% [code]
final = train_bk[train_bk.Date == test.Date.min() - datetime.timedelta(1) ]

# %% [code]
final.head()

# %% [code]
test['elapsed'] = (test.Date - final.Date.max()).dt.days 

# %% [raw]
# test.Date

# %% [code]
test.elapsed

# %% [code]


# %% [markdown]
# ### CFR Caps

# %% [code]
full_bk = test_wp.copy()

# %% [code]
full = test_wp.copy()

# %% [code]


# %% [code]
BASE_RATE = 0.01

# %% [code]
CFR_CAP = 0.13

# %% [code]


# %% [code]
lplot(full_bk)

# %% [code]
lplot(full_bk, columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [code]
full['cfr_imputed_fatalities_low'] = full.ConfirmedCases * np.exp(full.cfr_pred) / np.exp(0.5)
full['cfr_imputed_fatalities_high'] = full.ConfirmedCases * np.exp(full.cfr_pred) * np.exp(0.5)
full['cfr_imputed_fatalities'] = full.ConfirmedCases * np.exp(full.cfr_pred)  

# %% [code]


# %% [raw]
# full[(full.case_slope > 0.02) & 
#           (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
#                 (full.cfr_imputed_fatalities_low > 0.3) &
#                 ( full.Fatalities < 100 ) &
#     (full.Country!='China')] \
#      .groupby('Place').count()\
#     .sort_values('ConfirmedCases', ascending=False).iloc[:, 9:]

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
                (full.cfr_imputed_fatalities_low > 0.3) &
                ( full.Fatalities < 100000 ) &
    (full.Country!='China') &
     (full.Date == datetime.datetime(2020, 4,15))] \
     .groupby('Place').last()\
    .sort_values('Fatalities', ascending=False).iloc[:, 9:]

# %% [code]
(np.log(full.Fatalities + 1) -np.log(full.cfr_imputed_fatalities) ).plot(kind='hist', bins = 250)

# %% [raw]
# full[  
#                    (np.log(full.Fatalities + 1) < np.log(full.cfr_imputed_fatalities_high + 1) -0.5    ) 
#     & (~full.Country.isin(['China', 'Korea, South']))
#                 ][full.Date==train.Date.max()]\
#      .groupby('Place').first()\
#     .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities < full.cfr_imputed_fatalities_low    ) &
                (full.cfr_imputed_fatalities_low > 0.3) &
                ( full.Fatalities < 100000 ) &
    (~full.Country.isin(['China', 'Korea, South']))][full.Date==train.Date.max()]\
     .groupby('Place').first()\
    .sort_values('cfr_imputed_fatalities', ascending=False).iloc[:, 9:]

# %% [code]
full.Fatalities = np.where(   
    (full.case_slope > 0.02) & 
                   (full.Fatalities <= full.cfr_imputed_fatalities_low    ) &
                (full.cfr_imputed_fatalities_low > 0.3) &
                ( full.Fatalities < 100000 ) &
    (~full.Country.isin(['China', 'Korea, South'])) ,
                        
                        (full.cfr_imputed_fatalities_high + full.cfr_imputed_fatalities)/2,
                                    full.Fatalities)
    

# %% [raw]
# assert len(full) == len(data_wp)

# %% [raw]
# x_test.shape

# %% [code]
full['elapsed'] = (test_wp.Date - LAST_DATE).dt.days

# %% [code]
full[ (full.case_slope > 0.02) & 
          (np.log(full.Fatalities + 1) < np.log(full.ConfirmedCases * BASE_RATE + 1) - 0.5) &
                           (full.Country != 'China')]\
            [full.Date == datetime.datetime(2020, 4, 5)] \
            .groupby('Place').last().sort_values('ConfirmedCases', ascending=False).iloc[:,8:]

# %% [raw]
# full.Fatalities.max()

# %% [code]
full.Fatalities = np.where((full.case_slope > 0.02) & 
                      (full.Fatalities < full.ConfirmedCases * BASE_RATE) &
                           (full.Country != 'China'), 
                                            
            np.exp(   
                    np.log( full.ConfirmedCases * BASE_RATE + 1) \
                           * np.clip(   0.5* (full.elapsed - 1) / 30, 0, 1) \
                           
                     +  np.log(full.Fatalities +1 ) \
                           * np.clip(1 - 0.5* (full.elapsed - 1) / 30, 0, 1)
            ) -1
                           
                           ,
                                               full.Fatalities)  

# %% [raw]
# full.elapsed

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
    (full.Country!='China')]\
     .groupby('Place').count()\
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# %% [raw]
# full[full.Place=='United KingdomTurks and Caicos Islands']

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high * 2   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
    (full.Country!='China')  ]\
     .groupby('Place').last()\
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# %% [code]
full[(full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high * 1.5   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
    (full.Country!='China')][full.Date==train.Date.max()]\
     .groupby('Place').first()\
    .sort_values('ConfirmedCases', ascending=False).iloc[:, 8:]

# %% [code]


# %% [code]
full.Fatalities =  np.where(  (full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high      * 2   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
                (full.Country!='China') ,
                            
                     full.cfr_imputed_fatalities,
                            
                            full.Fatalities)

full.Fatalities =  np.where(  (full.case_slope > 0.02) & 
                   (full.Fatalities > full.cfr_imputed_fatalities_high   ) &
                (full.cfr_imputed_fatalities_low > 0.4) &
                (full.Country!='China') ,
                    np.exp(        
                            0.6667 * np.log(full.Fatalities + 1) \
                        + 0.3333 * np.log(full.cfr_imputed_fatalities + 1)
                                ) - 1,
                            
                            full.Fatalities)

# %% [code]


# %% [code]
full[(full.Fatalities > full.ConfirmedCases * CFR_CAP) &
                                          (full.ConfirmedCases > 1000)

    ]                        .groupby('Place').last().sort_values('Fatalities', ascending=False)

# %% [raw]
# full.Fatalities =  np.where( (full.Fatalities > full.ConfirmedCases * CFR_CAP) &
#                                           (full.ConfirmedCases > 1000)
#                                         , 
#                              full.ConfirmedCases * CFR_CAP\
#                                            * np.clip((full.elapsed - 5) / 15, 0, 1) \
#                                  +  full.Fatalities * np.clip(1 - (full.elapsed - 5) / 15, 0, 1)
#                             , 
#                                                full.Fatalities)

# %% [raw]
# train[train.Country=='Italy']

# %% [raw]
# final[final.Country=='US'].sum()

# %% [code]
(np.log(full.Fatalities + 1) -np.log(full.cfr_imputed_fatalities) ).plot(kind='hist', bins = 250)

# %% [code]


# %% [markdown]
# ### Fix Slopes now

# %% [raw]
# final

# %% [code]
assert len(pd.merge(full, final, on='Place', suffixes = ('', '_i'), validate='m:1')) == len(full)

# %% [code]
ffm = pd.merge(full, final, on='Place', suffixes = ('', '_i'), validate='m:1')
ffm['fatality_slope'] = (np.log(ffm.Fatalities + 1 )\
                             - np.log(ffm.Fatalities_i + 1 ) ) \
                                 / ffm.elapsed
ffm['case_slope'] = (np.log(ffm.ConfirmedCases + 1 ) \
                             - np.log(ffm.ConfirmedCases_i + 1 ) ) \
                                 / ffm.elapsed

# %% [markdown]
# #### Fix Upward Slopers

# %% [raw]
# final_slope = (ffm.groupby('Place').last().case_slope)
# final_slope.sort_values(ascending=False)
# 
# high_final_slope = final_slope[final_slope > 0.1].index

# %% [raw]
# slope_change = (ffm.groupby('Place').last().case_slope - ffm.groupby('Place').first().case_slope)
# slope_change.sort_values(ascending = False)
# high_slope_increase = slope_change[slope_change > 0.05].index

# %% [code]


# %% [raw]
# test.Date.min()

# %% [raw]
# set(high_slope_increase) & set(high_final_slope)

# %% [raw]
# ffm.groupby('Date').case_slope.median()

# %% [code]


# %% [markdown]
# ### Fix Drop-Offs

# %% [code]
ffm[np.log(ffm.Fatalities+1) < np.log(ffm.Fatalities_i+1) - 0.2]\
    [['Place', 'Date', 'elapsed', 'Fatalities', 'Fatalities_i']]

# %% [code]
ffm[np.log(ffm.ConfirmedCases + 1) < np.log(ffm.ConfirmedCases_i+1) - 0.2]\
    [['Place', 'elapsed', 'ConfirmedCases', 'ConfirmedCases_i']]

# %% [code]


# %% [raw]
# (ffm.groupby('Place').last().fatality_slope - ffm.groupby('Place').first().fatality_slope)\
#     .sort_values(ascending = False)[:10]

# %% [markdown]
# ### Display

# %% [raw]
# full[full.Country=='US'].groupby('Date').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'mean',
#         'fatality_slope': 'mean',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     })

# %% [code]
full_bk[(full_bk.Date == test.Date.max() ) & 
   (~full_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)

# %% [raw]
# full[full.Country=='China'].groupby('Date').agg(
#     {'ForecastId': 'count',
#      'case_slope': 'mean',
#         'fatality_slope': 'mean',
#             'ConfirmedCases': 'sum',
#                 'Fatalities': 'sum',
#                     })[::5]

# %% [code]


# %% [raw]
# ffc = pd.merge(final, full, on='Place', validate = '1:m')
# ffc[(np.log(ffc.Fatalities_x) - np.log(ffc.ConfirmedCase_x)) / ffc.elapsed_y ]

# %% [raw]
# ffm.groupby('Place').case_slope.last().sort_values(ascending = False)[:30]

# %% [raw]
# lplot(test_wp)

# %% [raw]
# lplot(test_wp, columns = ['case_slope', 'fatality_slope'])

# %% [code]

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)])

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [raw]
# test.Date.min()

# %% [code]
ffm.fatality_slope = np.clip(ffm.fatality_slope, None, 0.5)

# %% [raw]
# ffm.case_slope = np.clip(ffm.case_slope, None, 0.25)

# %% [code]


# %% [raw]
# for lr in [0.05, 0.02, 0.01, 0.007, 0.005, 0.003]:
# 
#     ffm.loc[ (ffm.Place==ffm.Place.shift(1) )
#          & (ffm.Place==ffm.Place.shift(-1) ) &
#      ( np.abs ( (ffm.case_slope.shift(-1) + ffm.case_slope.shift(1) ) / 2
#                        - ffm.case_slope).fillna(0)
#                     > lr ), 'case_slope'] = \
#                      ( ffm.case_slope.shift(-1) + ffm.case_slope.shift(1) ) / 2
# 

# %% [code]
for lr in [0.2, 0.14, 0.1, 0.07, 0.05, 0.03, 0.01 ]:

    ffm.loc[ (ffm.Place==ffm.Place.shift(4) )
         & (ffm.Place==ffm.Place.shift(-4) ), 'fatality_slope'] = \
         ( ffm.fatality_slope.shift(-2) * 0.25 \
              + ffm.fatality_slope.shift(-1) * 0.5 \
                + ffm.fatality_slope \
                  + ffm.fatality_slope.shift(1) * 0.5 \
                    + ffm.fatality_slope.shift(2) * 0.25 ) / 2.5


# %% [code]


# %% [code]
ffm.ConfirmedCases = np.exp( 
                            np.log(ffm.ConfirmedCases_i + 1) \
                                + ffm.case_slope * 
                                   ffm.elapsed ) - 1

ffm.Fatalities = np.exp(
                            np.log(ffm.Fatalities_i + 1) \
                              + ffm.fatality_slope * 
                                   ffm.elapsed ) - 1
# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1
                                     

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)])

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [code]
ffm[(ffm.Date == test.Date.max() ) & 
   (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)

# %% [code]


# %% [code]
ffm_bk = ffm.copy()

# %% [code]


# %% [code]


# %% [code]
ffm = ffm_bk.copy()

# %% [code]
counter = Counter(data.Place)
# counter.most_common()
median_count = np.median([ counter[group] for group in ffm.Place])
# [ (group, np.round( np.power(counter[group] / median_count, -1),3) ) for group in 
#      counter.keys()]
c_count = [ np.clip(
            np.power(counter[group] / median_count, -1.5), None, 2.5) for group in ffm.Place]
 

# %% [code]
RATE_MULT = 0.00
RATE_ADD = 0.003
LAG_FALLOFF = 15

ma_factor = np.clip( ( ffm.elapsed - 14) / 14 , 0, 1)

ffm.case_slope = np.where(ffm.elapsed > 0,
    0.7 * ffm.case_slope * (1+ ma_factor * RATE_MULT) \
         + 0.3 * (  ffm.case_slope.ewm(span=LAG_FALLOFF).mean()\
                                                      * np.clip(ma_factor, 0, 1)
                      + ffm.case_slope    * np.clip( 1 - ma_factor, 0, 1)) 
                          
                          + RATE_ADD * ma_factor * c_count,
         ffm.case_slope)

# --

RATE_MULT = 0
RATE_ADD = 0.015
LAG_FALLOFF = 15

ma_factor = np.clip( ( ffm.elapsed - 10) / 14 , 0, 1)


ffm.fatality_slope = np.where(ffm.elapsed > 0,
    0.3 * ffm.fatality_slope * (1+ ma_factor * RATE_MULT) \
         + 0.7* (  ffm.fatality_slope.ewm(span=LAG_FALLOFF).mean()\
                                                              * np.clip( ma_factor, 0, 1)
                      + ffm.fatality_slope    * np.clip( 1 - ma_factor, 0, 1)   )
                              
                              + RATE_ADD * ma_factor * c_count \
                              
                              
                              * (ffm.Country != 'China')
                              ,
         ffm.case_slope)

# %% [code]
ffm.ConfirmedCases = np.exp( 
                            np.log(ffm.ConfirmedCases_i + 1) \
                                + ffm.case_slope * 
                                   ffm.elapsed ) - 1

ffm.Fatalities = np.exp(
                            np.log(ffm.Fatalities_i + 1) \
                              + ffm.fatality_slope * 
                                   ffm.elapsed ) - 1
# test_wp.Fatalities = np.exp(
#                             np.log(test_wp.ConfirmedCases + 1) \
#                               + test_wp.cfr_pred  )  -1
                                     

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)])

# %% [code]


# %% [code]
lplot(ffm[~ffm.Place.isin(new_places)], columns = ['case_slope', 'fatality_slope'])

# %% [code]


# %% [raw]
# LAST_DATE

# %% [code]
ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
   (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[:15]

# %% [code]
ffm[(ffm.Date == test.Date.max() ) & 
   (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[:15]

# %% [code]


# %% [code]


# %% [code]


# %% [code]
ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
   (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[-50:]

# %% [code]
ffm[(ffm.Date == test.Date.max() ) & 
   (~ffm.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).loc[ffm_bk[(ffm_bk.Date == test.Date.max() ) & 
   (~ffm_bk.Place.isin(new_places))].groupby('Country').agg(
    {'ForecastId': 'count',
     'case_slope': 'last',
        'fatality_slope': 'last',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    }
).sort_values('ConfirmedCases', ascending=False)[-50:].index]

# %% [code]


# %% [code]
# use country-specific CFR !!!!  helps cap US and raise up Italy !
# could also use lagged CFR off cases as of 2 weeks ago...
 # ****  keep everything within ~0.5 order of magnitude of its predicted CFR.. !!


# %% [code]


# %% [markdown]
# ### Join

# %% [raw]
# assert len(test_wp) == len(full)
# 

# %% [raw]
# full = pd.merge(test_wp, full[['Place', 'Date', 'Fatalities']], on = ['Place', 'Date'],
#             validate='1:1')

# %% [code]


# %% [markdown]
# ### Fill in New Places with Ramp Average

# %% [code]
NUM_TEST_DATES = len(test.Date.unique())

base = np.zeros((2, NUM_TEST_DATES))
base2 = np.zeros((2, NUM_TEST_DATES))

# %% [code]
for idx, c in enumerate(['ConfirmedCases', 'Fatalities']):
    for n in range(0, NUM_TEST_DATES):
        base[idx,n] = np.mean(
            np.log(  train[((train.Date < test.Date.min())) & 
              (train.ConfirmedCases > 0)].groupby('Country').nth(n)[c]+1))

# %% [code]
base = np.pad( base, ((0,0), (6,0)), mode='constant', constant_values = 0)

# %% [code]
for n in range(0, base2.shape[1]):
    base2[:, n] = np.mean(base[:, n+0: n+7], axis = 1)

# %% [code]
new_places = train[(train.Date == test.Date.min() - datetime.timedelta(1)) &
      (train.ConfirmedCases == 0)
     ].Place

# %% [code]
# fill in new places 
ffm.ConfirmedCases = \
    np.where(   ffm.Place.isin(new_places),
          base2[ 0, (ffm.Date - test.Date.min()).dt.days],
                 ffm.ConfirmedCases)
ffm.Fatalities = \
    np.where(   ffm.Place.isin(new_places),
          base2[ 1, (ffm.Date - test.Date.min()).dt.days],
                 ffm.Fatalities)

# %% [code]


# %% [code]
ffm[ffm.Country=='US'].groupby('Date').agg(
    {'ForecastId': 'count',
     'case_slope': 'mean',
        'fatality_slope': 'mean',
            'ConfirmedCases': 'sum',
                'Fatalities': 'sum',
                    })

# %% [raw]
# train[train.Country == 'US'].Province_State.unique()

# %% [markdown]
# ### Save

# %% [code]


# %% [code]


# %% [code]
sub = pd.read_csv(input_path + '/submission.csv')

# %% [code]
scl = sub.columns.to_list()

# %% [code]

# print(full_bk.groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])
# print(ffm.groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])


# %% [code]
if ffm[scl].isnull().sum().sum() == 0:
    out = full_bk[scl] * 0.3 + ffm_bk[scl] * 0.3 + full[scl] * 0.3 + ffm[scl] * 0.1
else:
    print('using full-bk')
    out = full_bk[scl]

out.ForecastId = np.round(out.ForecastId, 0).astype(int) 

print(pd.merge(out, test[['ForecastId', 'Date', 'Place']], on='ForecastId')\
      .sort_values('ForecastId')\
          .groupby('Place').last()[['Date', 'ConfirmedCases', 'Fatalities']])

out = np.round(out, 2)
private = out[sub.columns.to_list()]
  


full_pred = pd.concat((private, public[~public.ForecastId.isin(private.ForecastId)]),
     ignore_index=True).sort_values('ForecastId')

full_pred.to_csv('submission.csv', index=False)