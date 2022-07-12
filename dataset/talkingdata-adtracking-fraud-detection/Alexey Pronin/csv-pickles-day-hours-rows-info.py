"""
This kernel creates 6 csv and 6 pickle files containing some useful information 
about days and hours included  in train, test, and test supplement. The files 
show what hours occured on what day and what rows correspond to the beginning 
and the end of each hour.

The summary files contain the beginning and ending rows for each day. This might 
be useful if you are training your model in chunks with each chunk equal to one 
day. Processing the 'click_time' column of the train data set takes time. It makes 
sense to do this computation elsewhere and then load the resulting csv every time 
you need it.

"""
############################################################################

import pandas as pd
import numpy as np

############################################################################

path = "../input/"

############################################################################
""" If debug = 1 (the debugging mode) then only 100000 rows will be processed; 
if debug = 0 then the whole data set will be processed.
"""
debug = 0 

if debug:
    nrows=100000
else:
    nrows=None

############################################################################

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

############################################################################

def extract_day_hour(name):
    print("Processing '{}'...\n".format(name))
    path_name = path + name + '.csv'
    print("Loading the data from '{}'...\n".format(path_name))
    df = pd.read_csv(path_name, dtype=dtypes, nrows=nrows, usecols=['click_time'], \
                     parse_dates=['click_time'])
    
    print("Building new features: 'day' and 'hour'...\n")
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
     
    df_days = df.day.unique()
    day_hour = pd.DataFrame()
    
    for day in df_days:
        new_hours = df.hour[df.day == day].unique()
        new_day = np.repeat(day, len(new_hours))
        new_start = [df.index[((df.day == day) & (df.hour == hour))].min() \
                              for hour in new_hours]
        new_end = [df.index[((df.day == day) & (df.hour == hour))].max() \
                              for hour in new_hours]
        new_df = pd.DataFrame.from_items([("day", new_day), \
                                          ("hour", new_hours), \
                                          ("start", new_start), \
                                          ("end", new_end)])
        day_hour = pd.concat((day_hour, new_df), ignore_index = True)
    
    print("The day/hour table for {}.".format(name))
    print(day_hour)
    path_day_hour_csv = 'day_hour_' + name + '.csv'
    path_day_hour_pickle = 'day_hour_' + name + '.pkl'
    day_hour.to_csv(path_day_hour_csv, index=False)
    day_hour.to_pickle(path_day_hour_pickle)

    summary = day_hour['start'].groupby(day_hour['day']). \
                        describe()[['count', 'min', 'max']].astype('uint32')
                        
    summary.rename(columns={'count': 'n_hours'}, inplace=True)

    summary['max'] = day_hour['end'].groupby(day_hour['day']). \
                        describe()[['max']].astype('uint32')

    summary['n_rows'] = summary['max'] - summary['min'] + 1
                        
    print("\nThe summary table for {}.".format(name))
    print(summary)
    
    path_summary = 'summary_' + name + '.csv'
    path_summary_pickle = 'summary_' + name + '.pkl'
    summary.to_csv(path_summary)
    summary.to_pickle(path_summary_pickle)
    
    return day_hour, summary

############################################################################
    

day_hour_train, summary_train = extract_day_hour('train')
day_hour_test, summary_test = extract_day_hour('test')
day_hour_train, summary_train = extract_day_hour('test_supplement')

############################################################################

# =============================================================================
# # CLEAR THE WORKSPACE
# 
# for name in dir():
#     if not name.startswith('_'):
#         del globals()[name]
# del(name)
# =============================================================================