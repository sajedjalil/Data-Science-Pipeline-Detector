#Baris Kanbar Stuff but generates libffm

import gc
import pandas as pd
import numpy as np

print('Loading Raw Data')
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'}
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip','app','device','os', 'channel', 'click_time']
print('Depending on your memory change nrows or add skiprows')
train_df = pd.read_csv('../input/train.csv',nrows=10000,usecols = train_cols,dtype = dtypes)
test_df = pd.read_csv('../input/test_supplement.csv',nrows=10000,usecols = test_cols,dtype = dtypes)
test_df['is_attributed'] = -1
print('Finished Loading Raw Data')

print('Munge by Date')
train_df = pd.concat([train_df[train_df.columns],test_df[train_df.columns]])
train_df['click_time'] = pd.to_datetime(train_df.click_time)
train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')
train_df['day'] = train_df.click_time.dt.day.astype('uint8')
train_df['epochtime'] = train_df.click_time - train_df.click_time.min()
train_df['epochtime'] = train_df['epochtime'].dt.total_seconds()
del test_df
train_df.drop(['click_time'],inplace=True,axis=1)
gc.collect()
print('Finished Munge by Date')

print('Deltas')

train_df.sort_values(by='epochtime',inplace=True,ascending=False)
train_df = train_df.reset_index(drop=True)
D=2**26

print('Product')
train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) + "_" + train_df['os'].astype(str)).apply(hash) % D
train_df['delta'] = -train_df[['category','epochtime']].groupby(by=['category']).diff().fillna(1)
train_df['delta_p1'] = -train_df[['category','epochtime']].groupby(by=['category']).diff(2).fillna(1)
train_df.drop(['category'],inplace=True,axis=1)
gc.collect()

print('Ip')
train_df['delta_ip'] = -train_df[['ip','epochtime']].groupby(by=['ip']).diff().fillna(1)
train_df['delta_ip_p1'] = -train_df[['ip','epochtime']].groupby(by=['ip']).diff(2).fillna(1)

train_df.sort_values(by='epochtime',inplace=True)
train_df = train_df.reset_index(drop=True)
print('Finished Deltas')
print('Creating Features')
naddfeat=9
for i in range(0,naddfeat):
    if i==0: selcols=['ip', 'channel']; QQ=4;
    if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
    if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
    if i==3: selcols=['ip', 'app']; QQ=4;
    if i==4: selcols=['ip', 'app', 'os']; QQ=4;
    if i==5: selcols=['ip', 'device']; QQ=4;
    if i==6: selcols=['app', 'channel']; QQ=4;
    if i==7: selcols=['ip', 'os']; QQ=5;
    if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
    print('selcols',selcols,'QQ',QQ)

    if QQ==0:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==1:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==2:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==3:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==4:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
            rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
        train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
    if QQ==5:
        gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
        train_df['X'+str(i)]=gp.values

    del gp
    gc.collect()    


print('grouping by ip-day-hour combination...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print('grouping by ip-app combination...')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

print('grouping by ip-app-os combination...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

# Adding features with var and mean hour (inspired from nuhsikander's script)
print('grouping by : ip_day_chl_var_hour')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()

print('grouping by : ip_app_os_var_hour')
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print('grouping by : ip_app_channel_var_day')
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print('grouping by : ip_app_chl_mean_hour')
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
print("merging...")
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print("vars and data type: ")
train_df.info()
train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')


train_df.fillna(-1,inplace=True,axis=1)
print('Finished Features')

bincols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
           'delta','delta_p1','delta_ip','delta_ip_p1',
           'ip_tcount',
           'ip_app_count', 'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
           'ip_app_channel_var_day', 'ip_app_channel_mean_hour']


for c in bincols:
    print('Binning: ', c, train_df.loc[train_df[c]!=-1,c].min())
    train_df.loc[train_df[c]!=-1,c] = pd.cut(np.log1p(train_df.loc[train_df[c]!=-1,c]), 100, labels=False)
    train_df[c] = train_df[c].astype('int16')
    gc.collect()

test_df = train_df[train_df.is_attributed==-1]
train_df = train_df[train_df.is_attributed>-1]
test_df.is_attributed=0
gc.collect()

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


gc.collect()

categories = ['app', 'device', 'os', 'channel', 'hour', 'day',
              'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8',
              'delta','delta_p1','delta_ip','delta_ip_p1',
              'ip_tcount',
              'ip_app_count', 'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
              'ip_app_channel_var_day', 'ip_app_channel_mean_hour']

numerics = []

currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train_df.shape[0]
print('Train',noofrows)
with open("alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%1000000)==0):
            print('Row',n)
        datastring = ""
        datarow = train_df.iloc[r].to_dict()
        datastring += str(int(datarow['is_attributed']))

        for i, x in enumerate(catdict.keys()):
            if((catdict[x]==0)):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)
        
noofrows = test_df.shape[0]
print('Test',noofrows)
with open("alltestffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n%1000000)==0):
            print('Row',n)
        datastring = ""
        datarow = test_df.iloc[r].to_dict()
        datastring += str(int(datarow['is_attributed']))

        for i, x in enumerate(catdict.keys()):
            if((catdict[x]==0)):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)

