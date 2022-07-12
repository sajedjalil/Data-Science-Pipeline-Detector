
import numpy
import pandas
import os
import json
from pandas.io.json import json_normalize

print(os.listdir("../input"))


cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']


def flat_dataframe(data_df):
    for col in cols_to_parse:
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    return data_df



chunksize = 100000



delcol = [
'socialEngagementType',
'device_browserSize', 
'device_browserVersion', 
'device_flashVersion', 
'device_language', 
'device_mobileDeviceBranding', 
'device_mobileDeviceInfo', 
'device_mobileDeviceMarketingName', 
'device_mobileDeviceModel', 
'device_mobileInputSelector', 
'device_operatingSystemVersion', 
'device_screenColors', 
'device_screenResolution', 
'geoNetwork_cityId', 
'geoNetwork_latitude', 
'geoNetwork_longitude', 
'geoNetwork_networkLocation', 
'totals_visits', 
'trafficSource_adwordsClickInfo.criteriaParameters',
]




df = pandas.DataFrame()
for dfpart in pandas.read_csv('../input/test_v2.csv', chunksize=chunksize, iterator=True, converters={column: json.loads for column in cols_to_parse}, dtype={'fullVisitorId': 'str'}):
    dfpart.drop(['hits','customDimensions'], axis=1, inplace=True)
    df = pandas.concat([df, dfpart])
    del dfpart

print(df.info())
df = flat_dataframe(df)
print(df.info())
df.drop(delcol, axis=1, inplace=True)
df.to_csv('test_v2_flat.csv', sep=',', encoding='utf-8', index=False)
del df







df = pandas.DataFrame()

for dfpart in pandas.read_csv('../input/train_v2.csv', chunksize=chunksize, iterator=True, converters={column: json.loads for column in cols_to_parse}, dtype={'fullVisitorId': 'str'}):
    dfpart.drop(['hits','customDimensions'], axis=1, inplace=True)
    df = pandas.concat([df, dfpart])
    del dfpart

print(df.info())
df = flat_dataframe(df)
print(df.info())
df.drop(delcol, axis=1, inplace=True)

df.to_csv('train_v2_flat.csv', sep=',', encoding='utf-8', index=False)




