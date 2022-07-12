import pandas as pd
import numpy as np
import gc
import time

### Functions ###
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

### Parameter's setting ###
full_day_importation_start = 9308568
fullday_importation_nrows = 59633310
nrows = 65 * 10 ** 6
start_importation = 184903890 - nrows - 1
        
### Train importation ###
start_time = time.time()
df = pd.read_csv('../input/train.csv', skiprows = start_importation, nrows = nrows)
header = pd.read_csv('../input/train.csv', nrows = 0) 
df.columns = header.columns

del header, df['attributed_time']
gc.collect()    

[conversion(v) for v in ['ip', 'app', 'device','os', 'channel', 'is_attributed']]
print("Importation completed :", round(time.time() - start_time,0),"seconds")

### Feature engineering ###
print("Feature engineering is processing")
start_time = time.time()

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
#df['minute'] = df.index.minute
#df['second'] = df.index.second
#[conversion (i) for i in ['hour', 'minute', 'second']]
conversion('hour')

# Number of clicks during the hour
by_hour = pd.DataFrame(df['hour'].value_counts())
by_hour.columns = ['clicks_by_hour']
by_hour['hour'] = by_hour.index
df = pd.merge(df, by_hour, on ='hour')
conversion('clicks_by_hour')
del by_hour
gc.collect()

# Number of click of the OS, application and device
#clicks_by_os = df.groupby('os').ip.count().reset_index()
#clicks_by_os.columns = ['os', 'clicks_by_os']
#clicks_by_app = df.groupby('app').ip.count().reset_index()
#clicks_by_app.columns = ['app', 'clicks_by_app']
clicks_by_device = df.groupby('device').ip.count().reset_index()
clicks_by_device.columns = ['device', 'clicks_by_device']
#df = pd.merge(df, clicks_by_os, on = 'os')
#conversion('clicks_by_os')
#df = pd.merge(df, clicks_by_app, on = 'app')
#conversion('clicks_by_app')
df = pd.merge(df, clicks_by_device, on = 'device')
conversion('clicks_by_device')
#del clicks_by_os, clicks_by_app,
del clicks_by_device
gc.collect()

# Last click from the user 
df['id'] = range(1, len(df) + 1)
last_clicks = df.groupby('ip').id.last().reset_index()
last_clicks['last_click'] = 1
last_clicks.drop('ip', axis=1, inplace = True)
df = pd.merge(df, last_clicks, on ='id', how = 'left').set_index(pd.to_datetime(df['click_time']))
df['last_click'].fillna(0, inplace = True)
conversion('last_click')
del df['id'], last_clicks
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
dl_rate_app = pd.DataFrame(df.groupby('app').is_attributed.mean()).reset_index()
dl_rate_app.columns = ['app', 'dl_rate_app']
df = pd.merge(df, dl_rate_app, on = 'app', how = 'left')
del dl_rate_app
gc.collect()
dl_rate_os = pd.DataFrame(df.groupby('os').is_attributed.mean()).reset_index()
dl_rate_os.columns = ['os', 'dl_rate_os']
df = pd.merge(df, dl_rate_os, on = 'os')
del dl_rate_os
gc.collect()
dl_rate_device = pd.DataFrame(df.groupby('device').is_attributed.mean()).reset_index()
dl_rate_device.columns = ['device', 'dl_rate_device']
df = pd.merge(df, dl_rate_device, on = 'device')
del dl_rate_device

df[['dl_rate_app', 'dl_rate_os', 'dl_rate_device']].apply(lambda x : x.fillna(x.mean(), inplace = True))
#df['dl_rate_app'] = df['dl_rate_app'].fillna(df['dl_rate_app'].mean())

gc.collect()

# Download rate by hour
dl_hour = pd.DataFrame(df.groupby('hour').is_attributed.mean())
dl_hour.reset_index(inplace = True)
dl_hour.columns = ['hour', 'DL_by_hour']
df = pd.merge(df, dl_hour, on = 'hour')
del dl_hour
gc.collect()

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
conversion('app_by_use')
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

df = do_count(df, ['ip', 'app'], 'ip_app_count', show_max=True )
conversion('ip_app_count')
gc.collect()
df = do_count(df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ) 
gc.collect()


print("Feature engineering completed :", round(time.time()-start_time,0)/60,"minutes")

print("df size :",round(sum(df.memory_usage())/1000000000,2),"GB")

df[['dl_rate_app', 'dl_rate_os', 'dl_rate_device', 'DL_by_hour']] = df[['dl_rate_app', 'dl_rate_os', 'dl_rate_device', 'DL_by_hour']].apply(lambda x : x * 10000)
[conversion (v) for v in ['dl_rate_app', 'dl_rate_os', 'dl_rate_device', 'DL_by_hour']]

print("df size after optimisation:",round(sum(df.memory_usage())/1000000000,2),"GB")
print(df.head(3))
# Exportation
#df.to_csv('train_up.csv', index = False, compression ='gzip')

df[['ip', 'app', 'os', 'channel', 'is_attributed', 'freq_ip', 'hour', 'clicks_by_hour']].to_csv('p1.csv', index = False, compression ='gzip')
print("Part 1 exported")
df[['clicks_by_device', 'last_click', 'channel_by_ip', 'app_by_use', 'nextClick']].to_csv('p2.csv', index = False, compression ='gzip')
print("Part 2 exported")
df[['dl_rate_app', 'dl_rate_os', 'DL_by_hour', 'app_by_ip', 'ip_app_os_count', 'ip_app_count']].to_csv('p3.csv', index = False, compression ='gzip')
print("Part 3 exported")