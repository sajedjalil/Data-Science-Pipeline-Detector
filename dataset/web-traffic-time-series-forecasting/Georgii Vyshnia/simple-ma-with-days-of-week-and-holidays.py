# Project/Competition: https://www.kaggle.com/c/web-traffic-time-series-forecasting/
# Simple benchmark prediction with median (median by page, weekdays, and holidays)
#
# Notes:
# - this is inspired by https://www.kaggle.com/clustifier/weekend-weekdays/ partially
# - language feature engineering reused from https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration/
# - You should insall Workalendar from its github repo directly
# >>> pip install git+https://github.com/novafloss/workalendar.git


import pandas as pd
import pandas.tseries.holiday as hol
import re
import datetime as dt
import numpy as np
from workalendar.europe import France, Germany, Spain
from workalendar.asia import Taiwan, Japan

def get_language(page):
    """Parse the language abbreviation out of the page url of a Wikipedia page"""
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'

def is_holiday_fr(date_of_interest):
    cal = France()
    result = cal.is_holiday(date_of_interest) # this will return a boolean True/False value
    # cast to integer
    result = int(result)
    return result

def is_holiday_de(date_of_interest):
    cal = Germany()
    result = cal.is_holiday(date_of_interest) # this will return a boolean True/False value
    # cast to integer
    result = int(result)
    return result

def is_holiday_es(date_of_interest):
    cal = Spain()      # this is a surrogate solution as the current data do not distinguish the 'es' traffic from
                       # Spain and other countries where Spanish is the official language
    result = cal.is_holiday(date_of_interest) # this will return a boolean True/False value
    # cast to integer
    result = int(result)
    return result

def is_holiday_ja(date_of_interest):
    cal = Japan()
    result = cal.is_holiday(date_of_interest) # this will return a boolean True/False value
    # cast to integer
    result = int(result)
    return result

def is_holiday_zh(date_of_interest):
    # this may be used for 'zh' pages as there is no implementation for China in workalendar yet
    cal = Taiwan()
    result = cal.is_holiday(date_of_interest) # this will return a boolean True/False value
    # cast to integer
    result = int(result)
    return result

def is_holiday_ru(date_of_interest):
    # Ru calendar is not implemented in workalendar - do it on the low-level
    # the current implementation will be bogus , via manually contstructed array of Russian state holidays in 2015-17
    # TODO:
    # - re-implement it with normal OOP patterns and reuse of calendar infrastructure
    # - the professional way of doing it relates to one of the options below
    # (1) extending workalendar.europe with appropriate contribution from your end, or
    # (2) implementing a class inheriting Pandas AbstractHolidayCalendar per the suggestions at
    #     https://stackoverflow.com/questions/33094297/create-trading-holiday-calendar-with-pandas

    # note: 2015 holidays are not complete as we are only interested for dates since Jul 1, 2015
    ru_holidays = {dt.date(2017, 1, 1), dt.date(2017, 1, 2), dt.date(2017, 1, 3),
        dt.date(2017, 1, 4), dt.date(2017, 1, 5), dt.date(2017, 1, 6),
        dt.date(2017, 1, 7), dt.date(2017, 2, 23), dt.date(2017, 2, 24),
        dt.date(2017, 3, 8),
        dt.date(2017, 5, 1), dt.date(2017, 5, 8), dt.date(2017, 5, 9),
        dt.date(2017, 6, 12), dt.date(2017, 11, 4), dt.date(2017, 11, 6),
        dt.date(2017, 12, 31),
        dt.date(2016, 1, 1), dt.date(2016, 1, 4), dt.date(2016, 1, 5),
        dt.date(2016, 1, 6), dt.date(2016, 1, 7),
        dt.date(2016, 2, 22), dt.date(2016, 2, 23), dt.date(2016, 3, 8),
        dt.date(2016, 5, 1), dt.date(2016, 5, 9),
        dt.date(2016, 6, 12), dt.date(2016, 6, 13), dt.date(2016, 11, 4),
        dt.date(2016, 12, 31),
        dt.date(2015, 11, 4), dt.date(2015, 12, 31)}

    if date_of_interest in ru_holidays:
        return 1
    else:
        return 0

########################################################################
# Main execusion loop
########################################################################

# US holidays
us_cal = hol.USFederalHolidayCalendar()
dr = pd.date_range(start='2015-07-01', end='2017-08-01')
us_holidays = us_cal.holidays(start=dr.min(), end=dr.max())

print('Reading train data...')
train = pd.read_csv("../input/train_1.csv")

print('Pre-processing and feature engineering train data...')
train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)
train_flattened['lang'] = train_flattened.Page.map(get_language)

print('Pre-processing: started featuring holidays by locales ... ')

lang_sets = {}
lang_sets['en'] = train_flattened[train_flattened.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train_flattened[train_flattened.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train_flattened[train_flattened.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train_flattened[train_flattened.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train_flattened[train_flattened.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train_flattened[train_flattened.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train_flattened[train_flattened.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train_flattened[train_flattened.lang=='es'].iloc[:,0:-1]

lang_sets['en']['holiday'] = ((lang_sets['en'].date.isin(us_holidays))).astype(float)
lang_sets['na']['holiday'] = ((lang_sets['na'].date.isin(us_holidays))).astype(float)  # assumption
lang_sets['fr']['holiday'] = ((lang_sets['fr'].date.map(is_holiday_fr))).astype(float)
lang_sets['de']['holiday'] = ((lang_sets['de'].date.map(is_holiday_de))).astype(float)
lang_sets['es']['holiday'] = ((lang_sets['es'].date.map(is_holiday_es))).astype(float)
lang_sets['zh']['holiday'] = ((lang_sets['zh'].date.map(is_holiday_zh))).astype(float)
lang_sets['ja']['holiday'] = ((lang_sets['ja'].date.map(is_holiday_ja))).astype(float)
lang_sets['ru']['holiday'] = ((lang_sets['ru'].date.map(is_holiday_ru))).astype(float)

# concatenate it back into a single training df
train_flattened = pd.concat(lang_sets)

print('Reading key data...')
test = pd.read_csv("../input/key_1.csv")

print('Processing key data...')
test['date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['date'] = test['date'].astype('datetime64[ns]')
test['weekend'] = ((test.date.dt.dayofweek) // 5 == 1).astype(float)
test['holiday'] = ((test.date.isin(us_holidays))).astype(float)
test['lang'] = test.Page.map(get_language)

print('Pre-processing: started featuring holidays by locales in key data ... ')

lang_sets_test = {}
lang_sets_test['en'] = test[test.lang=='en'].iloc[:,0:-1]
lang_sets_test['ja'] = test[test.lang=='ja'].iloc[:,0:-1]
lang_sets_test['de'] = test[test.lang=='de'].iloc[:,0:-1]
lang_sets_test['na'] = test[test.lang=='na'].iloc[:,0:-1]
lang_sets_test['fr'] = test[test.lang=='fr'].iloc[:,0:-1]
lang_sets_test['zh'] = test[test.lang=='zh'].iloc[:,0:-1]
lang_sets_test['ru'] = test[test.lang=='ru'].iloc[:,0:-1]
lang_sets_test['es'] = test[test.lang=='es'].iloc[:,0:-1]

lang_sets_test['en']['holiday'] = ((lang_sets_test['en'].date.isin(us_holidays))).astype(float)
lang_sets_test['na']['holiday'] = ((lang_sets_test['na'].date.isin(us_holidays))).astype(float)  # assumption
lang_sets_test['fr']['holiday'] = ((lang_sets_test['fr'].date.map(is_holiday_fr))).astype(float)
lang_sets_test['de']['holiday'] = ((lang_sets_test['de'].date.map(is_holiday_de))).astype(float)
lang_sets_test['es']['holiday'] = ((lang_sets_test['es'].date.map(is_holiday_es))).astype(float)
lang_sets_test['zh']['holiday'] = ((lang_sets_test['zh'].date.map(is_holiday_zh))).astype(float)
lang_sets_test['ja']['holiday'] = ((lang_sets_test['ja'].date.map(is_holiday_ja))).astype(float)
lang_sets_test['ru']['holiday'] = ((lang_sets_test['ru'].date.map(is_holiday_ru))).astype(float)

# concatenate it back into a single test df
test = pd.concat(lang_sets_test)

print('Calculating medians...')
train_page_per_dow = train_flattened.groupby(['Page','weekend', 'holiday']).median().reset_index()

print('Prepare submission dataframe...')
test = test.merge(train_page_per_dow, how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

print('Output submission dataframe...')
test[['Id','Visits']].to_csv('output/mad_holidays.csv', index=False)