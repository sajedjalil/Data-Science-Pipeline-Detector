from datetime import datetime
import pandas as pd
import numpy as np

start = datetime.now()

path = '/Users/mas/Documents/workspace/Avito/input/'
N = None #Use None if you want to read the whole file or 300000 to only read 300000 lines
how_value ='left' #parameter for Merge method
#Read data files
df_trainSearchStream = pd.read_csv(path +'trainSearchStream.tsv',nrows = N,delimiter = '\t')

df_AdsInfo = pd.read_csv(path + 'AdsInfo.tsv', delimiter = '\t',nrows = N, encoding='utf-8')

df_Category = pd.read_csv( path + 'Category.tsv', delimiter = '\t',nrows = N)

df_Location = pd.read_csv(path + 'Location.tsv', delimiter = '\t',nrows = N)

df_VisitsStream = pd.read_csv( path + 'VisitsStream.tsv', delimiter = '\t',nrows = N)

df_PhoneRequestsStream = pd.read_csv(path + 'PhoneRequestsStream.tsv', delimiter = '\t',nrows = N)

df_SearchInfo = pd.read_csv(path + 'SearchInfo.tsv', delimiter = '\t',nrows = N,encoding='utf-8')

df_UserInfo = pd.read_csv(path + 'UserInfo.tsv', delimiter = '\t',nrows = N)

#print df_trainSearchStream[:5]

#print df_AdsInfo[:5]

#print df_Category[:5]

#print df_Location[:5]

#print df_VisitsStream[:5]

#print df_PhoneRequestsStream[:5]

#print df_SearchInfo[:5]

#print df_UserInfo[:5]

df_SearchInfo = df_SearchInfo.merge(df_Location, on='LocationID',how = how_value)

df_SearchInfo = df_SearchInfo.merge(df_Category, on='CategoryID',suffixes=('_Location', '_Category'),how = how_value)

df_AdsInfo = df_AdsInfo.merge(df_Location, on='LocationID',how = how_value)

df_AdsInfo = df_AdsInfo.merge(df_Category, on='CategoryID',suffixes=('_Location', '_Category'),how= how_value)

df = df_trainSearchStream.merge(df_SearchInfo, on='SearchID',how= how_value)

df = df.merge(df_UserInfo, on='UserID',how= how_value)

df = df.merge(df_AdsInfo, on='AdID',suffixes=('_Search', '_Ad'),how = how_value )

df = df.merge(df_VisitsStream, on=['UserID', 'AdID'],suffixes=('_SearchInfo', '_VisitsStream'),how= how_value)

df = df.merge(df_PhoneRequestsStream, on=['UserID', 'AdID'],how= how_value)

print df[:10]
#print df.info

print('\a') #Play a notification sound 

print 'Elapsed time: %s' % str(datetime.now() - start)