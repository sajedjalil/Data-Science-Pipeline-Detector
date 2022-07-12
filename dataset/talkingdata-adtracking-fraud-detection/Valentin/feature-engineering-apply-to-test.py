import pandas as pd
import numpy as np
import gc

# Type's conversions
def conversion (var):
    if df[var].dtype != object:
        maxi = df[var].max()
        if maxi < 255:
            df[var] = df[var].astype(np.uint8)
            print(var,"converted to uint8")
        elif maxi < 65535:
            df[var] = df[var].astype(np.uint16)
            print(var,"converted to uint16")
        elif maxi < 4294967295:
            df[var] = df[var].astype(np.uint32)
            print(var,"converted to uint32")
        else:
            df[var] = df[var].astype(np.uint64)
            print(var,"converted to uint64")

# Importation
df = pd.read_csv('../input/test.csv')
for v in ['ip', 'app', 'device','os', 'channel'] :
    conversion(v)

# Number of clicks from the IP
IP_ready_to_merge = pd.DataFrame(df['ip'].value_counts()).reset_index()
IP_ready_to_merge.columns=['ip', 'freq_ip']
df = pd.merge(df, IP_ready_to_merge, on ='ip')
conversion('freq_ip')
del IP_ready_to_merge
gc.collect()

# Hour, minute and second
df.set_index(pd.to_datetime(df['click_time']), inplace = True)
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['second'] = df.index.second
[conversion (i) for i in ['hour', 'minute', 'second']]

# Number of clicks during the hour
by_hour = pd.DataFrame(df['hour'].value_counts())
by_hour.columns = ['clicks_by_hour']
by_hour['hour'] = by_hour.index
df = pd.merge(df, by_hour, on ='hour')
conversion('clicks_by_hour')
del by_hour
gc.collect()

# Number of click of the OS, application and device
clicks_by_os = df.groupby('os').ip.count().reset_index()
clicks_by_os.columns = ['os', 'clicks_by_os']
clicks_by_app = df.groupby('app').ip.count().reset_index()
clicks_by_app.columns = ['app', 'clicks_by_app']
clicks_by_device = df.groupby('device').ip.count().reset_index()
clicks_by_device.columns = ['device', 'clicks_by_device']
df = pd.merge(df, clicks_by_os, on = 'os')
conversion('clicks_by_os')
df = pd.merge(df, clicks_by_app, on = 'app')
conversion('clicks_by_app')
df = pd.merge(df, clicks_by_device, on = 'device')
conversion('clicks_by_device')
del clicks_by_os, clicks_by_app, clicks_by_device
gc.collect()

# Last click from the user 
df['id'] = range(1, len(df) + 1)
last_clicks = df.groupby('ip').id.last().reset_index()
last_clicks['last_click'] = 1
last_clicks.drop('ip', axis=1, inplace = True)
df = pd.merge(df, last_clicks, on ='id', how = 'left').set_index(pd.to_datetime(df['click_time']))
df['last_click'].fillna(0, inplace = True)
conversion('last_click')
del df['id'], df['click_time'], last_clicks
gc.collect()

# Number of clicks during the last minute
df['click_time'] = df.index
df.sort_values(['click_time', 'ip'], inplace = True)
clicks_minute = pd.DataFrame(df.groupby('ip')['app'].rolling('min').count())
clicks_minute.reset_index(inplace = True)
clicks_minute.sort_values(['click_time', 'ip'], inplace = True)
del clicks_minute['click_time'], clicks_minute['ip']
gc.collect()
df['clicks_minute'] = clicks_minute.values
conversion('clicks_minute')
del clicks_minute
gc.collect()

# Next click
df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
df['nextClick'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time).astype(np.float32)
df['nextClick'].fillna(df['nextClick'].mean(), inplace = True)
del df['click_time']
gc.collect()

# Download rate by app, device and OS
nrows = 57 * 10 ** 6
start_importation = 184903890 - nrows - 1
train = pd.read_csv('../input/train.csv', skiprows = 9308568, nrows = nrows, usecols = [1,2,3,5,7])
header = pd.read_csv('../input/train.csv', nrows = 0, usecols = [1,2,3,5,7]) 
train.columns = header.columns

dl_rate_app = pd.DataFrame(train.groupby('app').is_attributed.mean()).reset_index()
dl_rate_app.columns = ['app', 'dl_rate_app']
df = pd.merge(df, dl_rate_app, on = 'app', how = 'left')
del dl_rate_app
gc.collect()

dl_rate_os = pd.DataFrame(train.groupby('os').is_attributed.mean()).reset_index()
dl_rate_os.columns = ['os', 'dl_rate_os']
df = pd.merge(df, dl_rate_os, on = 'os', how = 'left')
del dl_rate_os
gc.collect()

dl_rate_device = pd.DataFrame(train.groupby('device').is_attributed.mean()).reset_index()
dl_rate_device.columns = ['device', 'dl_rate_device']
df = pd.merge(df, dl_rate_device, on = 'device', how = 'left')
del dl_rate_device, header

df['dl_rate_app'] = df['dl_rate_app'].fillna(df['dl_rate_app'].mean())
df['dl_rate_device'] = df['dl_rate_device'].fillna(df['dl_rate_device'].mean())
df['dl_rate_os'] = df['dl_rate_os'].fillna(df['dl_rate_os'].mean())
gc.collect()

# Download rate by hour
train.set_index(pd.to_datetime(train['click_time']), inplace = True)
train['hour'] = train.index.hour
dl_hour = pd.DataFrame(train.groupby('hour').is_attributed.mean())
dl_hour.reset_index(inplace = True)
dl_hour.columns = ['hour', 'DL_by_hour']
df = pd.merge(df, dl_hour, on = 'hour')
del dl_hour, train
gc.collect()
df[['dl_rate_app', 'dl_rate_os', 'dl_rate_device', 'DL_by_hour']] = df[['dl_rate_app', 'dl_rate_os', 'dl_rate_device', 'DL_by_hour']].apply(lambda x : x * 10000)
[conversion(v) for v in ['dl_rate_app', 'dl_rate_os', 'dl_rate_device', 'DL_by_hour']] 
print("Download rate by hour", len(df) - 18790469)

# Crossed features
def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
df = do_countuniq(df, ['ip'], 'channel', 'channel_by_ip', 'uint8', show_max=True )
gc.collect()
df = do_countuniq(df, ['ip', 'device', 'os'], 'app', 'app_by_use', show_max=True )
gc.collect()
df = do_countuniq(df, ['ip'], 'app', 'app_by_ip', 'uint8', show_max=True )
gc.collect()

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

df = do_count(df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
df = do_count(df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()

print(df.shape)
print(df.columns)

# Exportation
print("Exportation is running")
#df.drop(['ip', 'app', 'device', 'os'], axis = 1, inplace = True)
df.to_csv('test_up.csv', compression = 'gzip')