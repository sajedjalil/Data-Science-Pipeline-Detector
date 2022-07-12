import os
import glob
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import time
from scipy.signal import find_peaks, savgol_filter
from numba import njit
from scipy.spatial.distance import cdist
import gc
import warnings
warnings.filterwarnings('ignore')





######################################################################################
# part 1 - read all data #############################################################
######################################################################################

# init timer
start_time = time.time()

# data structure
@dataclass
class ReadData:
    acce:           np.ndarray
    ahrs:           np.ndarray
    wifi:           np.ndarray
    waypoint:       np.ndarray
    SiteID:         str
    FileName:       str
    FloorNum:       int

# site decode dictionary
site_di = {'5a0546857ecc773753327266':0,'5c3c44b80379370013e0fd2b':1,'5d27075f03f801723c2e360f':2,'5d27096c03f801723c31e5e0':3,
    '5d27097f03f801723c320d97':4,'5d27099f03f801723c32511d':5,'5d2709a003f801723c3251bf':6,'5d2709b303f801723c327472':7,
    '5d2709bb03f801723c32852c':8,'5d2709c303f801723c3299ee':9,'5d2709d403f801723c32bd39':10,'5d2709e003f801723c32d896':11,
    '5da138274db8ce0c98bbd3d2':12,'5da1382d4db8ce0c98bbe92e':13,'5da138314db8ce0c98bbf3a0':14,'5da138364db8ce0c98bc00f1':15,
    '5da1383b4db8ce0c98bc11ab':16,'5da138754db8ce0c98bca82f':17,'5da138764db8ce0c98bcaa46':18,'5da1389e4db8ce0c98bd0547':19,
    '5da138b74db8ce0c98bd4774':20,'5da958dd46f8266d0737457b':21,'5dbc1d84c1eb61796cf7c010':22,'5dc8cea7659e181adb076a3f':23}

# all train sites
test_bldg = list(site_di.keys())

# floor decode dictionary
fl_di = {'F1':0, 'F2':1, 'F3':2, 'F4':3, 'F5':4, 'F6':5, 'F7':6, 'F8':7, '1F':0, '2F':1, '3F':2,
    '4F':3, '5F':4, '6F':5, '7F':6, '8F':7, '9F':8, 'B1':-1, 'B2':-2}

# BSSID decode dictionary - construct it as data is read
BSSID_di = {}

# this function reads one data file at a time
def read_data_file(data_filename, call_type):# call_type: 0=train, 1=test
    acce        = []
    ahrs        = []
    wifi        = []
    waypoint    = []
    FloorNum    = -99
    ts          = 0
    wifi_c      = 0

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # assign vals from filename
    data_filename = str(data_filename).split('/')
    FileName      = data_filename[-1].split('.')[0]

    if call_type == 0: # train data, infer from path
        SiteID   = data_filename[-3]
        FloorNum = fl_di[data_filename[-2]]
    
    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            # read metadata
            if 'startTime' in line_data:
                ld2 = line_data[10 + line_data.find('startTime'):]
                ld2 = ld2.split('\t')
                ld2 = ld2[0].split(':')
                startTime = int(ld2[0])
            if 'SiteID' in line_data:
                ld2 = line_data.split(':')
                ld2 = ld2[1].split('\t')
                SiteID = ld2[0]
            if 'FloorName' in line_data:
                ld2 = line_data[line_data.find('FloorName'):]
                ld2 = ld2.split(':')
                if FloorNum == -99 and ld2[1] != '':
                    FloorNum = fl_di[ld2[1]]
            continue

        line_data = line_data.split('\t')

        if len(line_data) < 5: # correct data error
            line_data.append(0)

        if call_type > 0 and line_data[1] == 'TYPE_ACCELEROMETER': # only need this for test data. Get tot acce - that is all i need
            a = np.sqrt(float(line_data[2])**2 + float(line_data[3])**2 + float(line_data[4])**2)
            acce.append([int(line_data[0])-startTime, a])
            continue

        if call_type > 0 and line_data[1] == 'TYPE_ROTATION_VECTOR': # only need this for test data
            ahrs.append([int(line_data[0])-startTime, float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts  = int(line_data[0])-startTime
            bssid_t = line_data[3]
            rssi    = line_data[4]

            #skip wifis after 20 per timestamp
            if sys_ts == ts:
                wifi_c += 1
            else:
                wifi_c = 0
                ts = sys_ts
            if wifi_c > 20:
                continue

            bssid = (BSSID_di.get(bssid_t) or -1)
            if bssid  == -1: # add each new bssid to the dictionary
                BSSID_di[bssid_t] = 1 + len(BSSID_di)
                bssid = BSSID_di[bssid_t]
            
            wifi_data = [int(sys_ts), bssid, int(rssi)]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0])-startTime, float(line_data[2]), float(line_data[3])])

    acce     = np.array(acce, dtype=np.float32)
    ahrs     = np.array(ahrs, dtype=np.float32)
    wifi     = np.array(wifi, dtype=np.int32)
    waypoint = np.array(waypoint, dtype=np.float32)
    return ReadData(acce, ahrs, wifi, waypoint, SiteID, FileName, FloorNum)



# read train data - prepare data objects
misc_tr = pd.DataFrame()
waypoint_tr = np.zeros([75278, 5], dtype=np.float32)
wifi_tr = np.zeros([5385467, 6], dtype=np.int32)
train_waypoints = pd.DataFrame()
misc_tr = pd.DataFrame({'waypoint_s':np.zeros(10877, dtype=np.int32)})
misc_tr['wifi_s']   = 0
misc_tr['ahrs_s']   = 0
misc_tr['Floor']    = 0
misc_tr['Site']     = ''
misc_tr['PathName'] = ''
misc_tr['path']     = 0
wifi_s = i = waypoint_s = 0

# read train data
data_path = Path('../input/indoor-location-navigation/train')
floorplans = []

# select buildings in test
for f in sorted(glob.glob(f'{data_path}/*/*')):
    if f.split('/')[-2] in test_bldg:
        floorplans.append(f)
paths = {fp:glob.glob(f'{fp}/*.txt') for fp in floorplans}

# loop over all sites
for p in paths:
    for f in os.listdir(p):
        data = read_data_file(os.path.join(p, f), 0)
    
        if data.waypoint.shape[0] > 0:
            df = pd.DataFrame({'x':data.waypoint[:,1], 'y':data.waypoint[:,2], 'site':data.SiteID, 'floor':data.FloorNum, 'path':i, 'pathName':data.FileName})
            train_waypoints = train_waypoints.append(df)
            
            waypoint_tr[waypoint_s:waypoint_s + data.waypoint.shape[0], 0:3] = data.waypoint
            waypoint_tr[waypoint_s:waypoint_s + data.waypoint.shape[0], 3]   = i
            waypoint_tr[waypoint_s:waypoint_s + data.waypoint.shape[0], 4]   = data.FloorNum
            waypoint_s += data.waypoint.shape[0]

        if data.wifi.shape[0] > 0:
            wifi_tr[wifi_s:wifi_s + data.wifi.shape[0], 0]   = data.wifi[:,0]
            wifi_tr[wifi_s:wifi_s + data.wifi.shape[0], 2]   = data.wifi[:,1]
            wifi_tr[wifi_s:wifi_s + data.wifi.shape[0], 3]   = data.wifi[:,2]
            wifi_tr[wifi_s:wifi_s + data.wifi.shape[0], 4]   = i
            wifi_tr[wifi_s:wifi_s + data.wifi.shape[0], 5]   = data.FloorNum
            wifi_s += data.wifi.shape[0]

        misc_tr['wifi_s'].iat[i]      = wifi_s
        misc_tr['waypoint_s'].iat[i]  = waypoint_s
        misc_tr['Floor'].iat[i]       = data.FloorNum
        misc_tr['Site'].iat[i]        = data.SiteID
        misc_tr['PathName'].iat[i]    = data.FileName
        misc_tr['path'].iat[i]        = i

        if i > 0 and i%1000 == 0:
            print(i, int(time.time() - start_time), 'sec')
        i += 1
print('read train data', int(time.time() - start_time), 'sec')



# read test data - prepare data objects
misc_te = pd.DataFrame()
ahrs = np.zeros([3819802, 9], dtype=np.float32)
acce = np.zeros([3819802, 2], dtype=np.float32)
wifi_te = np.zeros([790894, 6], dtype=np.int32)
misc_te = pd.DataFrame({'waypoint_s':np.zeros(626, dtype=np.int32)})
misc_te['wifi_s']   = 0
misc_te['ahrs_s']   = 0
misc_te['Floor']    = 0
misc_te['Site']     = ''
misc_te['PathName'] = ''
misc_te['path']     = 0
path_di = {}
wifi_s = i = ahrs_s = 0

# read test data
data_path = Path('../input/indoor-location-navigation/test')
for f in os.listdir(data_path):
    data = read_data_file(os.path.join(data_path, f), 1)
    path_di[f[:-4]] = i # need this for encoding final submission

    if data.ahrs.shape[0] > 0:
        ahrs[ahrs_s:ahrs_s + data.ahrs.shape[0], 8]   = site_di[data.SiteID]
        ahrs[ahrs_s:ahrs_s + data.ahrs.shape[0], 0:4] = data.ahrs
        ahrs[ahrs_s:ahrs_s + data.ahrs.shape[0], 4]   = i
        acce[ahrs_s:ahrs_s + data.ahrs.shape[0], :]   = data.acce
        ahrs_s += data.ahrs.shape[0]

    if data.wifi.shape[0] > 0:
        wifi_te[wifi_s:wifi_s + data.wifi.shape[0], 0] = data.wifi[:,0]
        wifi_te[wifi_s:wifi_s + data.wifi.shape[0], 2] = data.wifi[:,1]
        wifi_te[wifi_s:wifi_s + data.wifi.shape[0], 3] = data.wifi[:,2]
        wifi_te[wifi_s:wifi_s + data.wifi.shape[0], 4] = i + 100000 # to separate test from train
        wifi_s += data.wifi.shape[0]

    misc_te['wifi_s'].iat[i]    = wifi_s
    misc_te['ahrs_s'].iat[i]    = ahrs_s
    misc_te['Site'].iat[i]      = data.SiteID
    misc_te['PathName'].iat[i]  = data.FileName
    misc_te['path'].iat[i]      = i + 100000 # to make path unique
    i += 1
print('read test data', int(time.time() - start_time), 'sec')



# read sample submission
sub = pd.read_csv('../input/indoor-location-navigation/sample_submission.csv')
tss = sub['site_path_timestamp'].str.split('_')
sub['path'] = tss.apply(lambda x: x[1]).map(path_di).astype('int32')
sub['ts'] = tss.apply(lambda x: x[2]).astype('int32')
sub = sub.sort_values(by=['path', 'ts']).reset_index(drop=True)
misc_te['waypoint_s'] = sub.groupby('path').size().reset_index()[0].cumsum()





######################################################################################
# part 2 - make relative prediction (dead reckoning) for test paths###################
######################################################################################

# dead reckoning parameters
ang_lim     = 19.5
h           = 10.59
min_dist    = 22
step_length = 0.6717
v_min       = 0.02666536
window      = 33
v_max       = 1.4798
p1          = 0.116315
p2          = 0.03715

# process acceleration data to get steps and speed
acce[:,1] = savgol_filter(acce[:,1], 15, 1)
peak_times, _ = find_peaks(acce[:,1], height=h, distance=min_dist)
peak_times = np.round(peak_times, 0).astype(np.int32)
print('steps per minute:', int(.5 + 60 * peak_times.shape[0] * 50 / acce.shape[0]))

# set speed
v = np.zeros(ahrs.shape[0], dtype=np.float32)
i = v0 = 0
for j in range(peak_times.shape[0] - 1):
    v[i:peak_times[j]] = v0
    i  = peak_times[j]
    f  = acce[peak_times[j]:peak_times[j+1],1]
    f  = f.std()
    v0 = 50 * (p1 * f + step_length - p2 * np.sqrt(peak_times[j+1] - peak_times[j])) / (peak_times[j+1] - peak_times[j])
v[i:] = v0
v = savgol_filter(v, window, 1) # smooth speed
v = np.minimum(v_max, np.maximum(v_min, v)) # cap/floor
print('process acceleration data to get steps and speed', int(time.time() - start_time), 'sec')

# process ahrs data
cos  = np.sqrt(1 - np.minimum(0.9999999, (ahrs[:,1:4] * ahrs[:,1:4]).sum(axis=1)))
x    = 2 * (ahrs[:,1] * ahrs[:,2] - ahrs[:,3] * cos)
y    = 1 - 2 * (ahrs[:,1] * ahrs[:,1] + ahrs[:,3] * ahrs[:,3])
norm = np.sqrt(x * x + y * y)
x    = x / norm
y    = y / norm

# rotate by an angle
ang = np.arctan2(x,y) * 180 / 3.14159 # degrees

# this is rotation that places most points into +-10 degrees of cardinal directions
ang_rot_site =  [ 15, -33, -5, -33, -25, 6, 11, 3, -17, 1, 11, -2, -39, -1, 0, -44, 8, 1, 2,  0, -14, 5, 40, -27]
for i in range(24): # loop over sites
    ang_rot = ang_rot_site[i]
    idx2 = (i == ahrs[:,8])
    # if close to cardinal direction, assume it is equal to that direction
    # north
    idx = idx2 & (np.abs(ang-ang_rot) < ang_lim)
    ang[idx] = 0 + ang_rot
    # south
    idx = idx2 & (np.abs(np.abs(ang-ang_rot) - 180) < ang_lim)
    ang[idx] = 180 + ang_rot
    # east
    idx = idx2 & (np.abs(ang-ang_rot - 90) < ang_lim)
    ang[idx] = 90 + ang_rot
    # west
    idx = idx2 & (np.abs(ang-ang_rot + 90) < ang_lim)
    ang[idx] = -90 + ang_rot
ang_inc_site = [-2.0, 10.0, -1.5, -7.0, -4.5, -7.5, -2.5, -9.0, -10.0, -3.0, -6.5, -9.5, -7.0, -0.5, -5.0, -6.0, -8.0, 0.5, -5.0, -1.5, -1.0, -10.0, 0.0, -0.5]
for i in range(24): # loop over sites
    idx = (i == ahrs[:,8])
    ang[idx] = ang[idx] + ang_inc_site[i]
  
# restate x/y using rotated coords
x = np.sin(ang / 180 * 3.14159)
y = np.cos(ang / 180 * 3.14159)

# get projected position
ahrs[:,5] = (v * np.append(0, x[:-1] * (ahrs[1:,0] - ahrs[:-1,0]) / 1000)).cumsum()
ahrs[:,6] = (v * np.append(0, y[:-1] * (ahrs[1:,0] - ahrs[:-1,0]) / 1000)).cumsum()
print('got projected position', int(time.time() - start_time), 'sec')



# indices of waypoints - only use them for finding intersecting paths
path1 = np.array(sub['path'].astype('int64'))
ts1 = np.array(sub['ts'].astype('int64'))
i1 = path1 * 10000000 + ts1
path2 = ahrs[:,4].astype(np.int64)
ts2 = ahrs[:,0].astype(np.int64)
i2 = path2 * 10000000 + ts2
indices = []
m0 = 0
for i in range(sub.shape[0]):
    m = m0 + (i2[m0:m0 + 20000] >= i1[i]).argmax()
    if np.abs(i1[i] - i2[m]) > 100000: # use last point from correct path
        m -= 1
    indices.append(m)
    m0 = m

# select waypoints only, and get the closest position to each one.
subl = sub.copy()
subl['x'] = ahrs[indices,5] # projected position for waypoints
subl['y'] = ahrs[indices,6]

# find intersecting paths
misc_te2 = misc_te.copy()
misc_te2['path'] = misc_te['path'] - 100000 # turn it into normal path, for merging
subl = subl.merge(misc_te2[['path','waypoint_s','ahrs_s']], how='left', on='path')
res = []
for i1 in range(subl.shape[0] - 2):
    for j1 in range(i1+2, subl.shape[0]):
        if subl['path'].iat[i1] != subl['path'].iat[j1]:
            break
        dt = subl['ts'].iat[j1] - subl['ts'].iat[i1]
        if dt > 3700:
            dt = max(1, dt) / 1000
            d = np.sqrt((subl['x'].iat[i1] - subl['x'].iat[j1])**2 + (subl['y'].iat[i1] - subl['y'].iat[j1])**2)
            if d < 6.54 and d / dt < 0.064:
                res.append([i1, j1, subl['path'].iat[i1], subl['waypoint_s'].iat[i1], indices[i1], indices[j1], subl['ahrs_s'].iat[i1]])
                break # no tripples - move on to next i
res = pd.DataFrame(res)
res.columns = ['i', 'j', 'path', 'waypoint_s', 'i2', 'j2', 'ahrs_s']

# correct intersecting paths
for k1 in range(res.shape[0]):
    i, j, path, waypoint_s , i2, j2, ahrs_s = res.iloc[k1]
    ts = np.array(subl['ts'].iloc[i:j+1])
    ts = ts - ts[0]
    ts = ts / ts[-1]
    mult = np.append(ts, np.ones(waypoint_s - 1 - j))
    subl['x'].iloc[i:waypoint_s] += (subl['x'].iloc[i] - subl['x'].iloc[j]) * mult
    subl['y'].iloc[i:waypoint_s] += (subl['y'].iloc[i] - subl['y'].iloc[j]) * mult

    ts = np.array(ahrs[i2:j2+1, 0])
    ts = ts - ts[0]
    ts = ts / ts[-1]
    mult = np.append(ts, np.ones(ahrs_s - 1 - j2))
    ahrs[i2:ahrs_s, 5] += (ahrs[i2, 5] - ahrs[j2, 5]) * mult
    ahrs[i2:ahrs_s, 6] += (ahrs[i2, 6] - ahrs[j2, 6]) * mult





######################################################################################
# part 3 - fingerprinting ############################################################
######################################################################################

# assign coordinates to each train wifi point (interpolate between waypoints)
wifi_s = waypoint_s = 0
wifi_xy = np.zeros([wifi_tr.shape[0], 2], dtype=np.float32)
for i in range(misc_tr.shape[0]):
    wifi = misc_tr['wifi_s'].iat[i] - wifi_s
    waypoint = misc_tr['waypoint_s'].iat[i] - waypoint_s
    waypoints = waypoint_tr[waypoint_s:waypoint_s + waypoint, :]
    waypoints_t = waypoints[:,0].astype(np.int32)
    # here each t is repeated many times - loop over distinct t values
    values, counts = np.unique(wifi_tr[wifi_s:wifi_s+wifi,0], return_counts=True)
    j = 0
    for c in range(values.shape[0]):
        t = values[c]
        if t <= waypoints_t[0]:
            k1 = 0
            k2 = k1
            w1 = 1
        elif t >= waypoints_t[-1]:
            k1 = waypoints_t.shape[0] - 1
            k2 = k1
            w1 = 1
        else:
            k2 = ((waypoints_t - t) > 0).argmax()
            k1 = k2 - 1
            w1 = (waypoints_t[k2] - t)/ (waypoints_t[k2] - waypoints_t[k1])
        wifi_xy[wifi_s:wifi_s+counts[c], 0] = waypoint_tr[waypoint_s + k1, 1] * w1 + waypoint_tr[waypoint_s + k2, 1] * (1 - w1)
        wifi_xy[wifi_s:wifi_s+counts[c], 1] = waypoint_tr[waypoint_s + k1, 2] * w1 + waypoint_tr[waypoint_s + k2, 2] * (1 - w1)
        j += counts[c]
        wifi_s +=counts[c]
    waypoint_s += waypoint
print('prepared train coordinates', int(time.time() - start_time), 'sec')

# function for data formatting: construct unique index
@njit
def f2id(ts, path):# ts, path: count unique combos of ts/path
    j = 0
    index = np.zeros(ts.shape[0], dtype=np.int32)
    for i in range(1, ts.shape[0]):
        if ts[i] != ts[i-1] or path[i] != path[i-1]:
            j = j + 1
        index[i] = j
    return index

# put id in wifi data
wifi_tr[:,1] = f2id(wifi_tr[:,0], wifi_tr[:,4])
wifi_te[:,1] = 1000000 + f2id(wifi_te[:,0], wifi_te[:,4]) # make test separable from train by adding 1M

# only keep bssids that are in train data
bssids = set(wifi_tr[:,2])
rows = [i for i in range(wifi_te.shape[0]) if wifi_te[i,2] in bssids]
wifi_te = wifi_te[rows,:]

# combine train and test data
wifi_xy = np.append(wifi_xy, np.zeros([wifi_tr.shape[0], wifi_xy.shape[1]], dtype=np.float32)).reshape(-1, wifi_xy.shape[1])
wifi_tr = np.append(wifi_tr, wifi_te).reshape(-1, wifi_tr.shape[1])    
misc_tr = misc_tr.append(misc_te)

# save
wifi_tr0 = wifi_tr.copy()
wifi_xy0 = wifi_xy.copy()

# loop over all sites******************************************************************
df1_tot = pd.DataFrame()
site_id = 0
for site in misc_tr['Site'].unique():
    if site == '':
        break
    print(site_id, site, 'site', int(time.time() - start_time), 'sec')
    site_id += 1
    
    # select current site only
    paths = set(misc_tr['path'].loc[misc_tr['Site'] == site])
    rows = [i for i in range(wifi_tr0.shape[0]) if wifi_tr0[i,4] in paths]
    wifi_tr = wifi_tr0[rows,:].copy()
    wifi_xy = wifi_xy0[rows,:].copy()

    # only keep bssids that are present in both train and val
    bssids = set(wifi_tr[wifi_tr[:,1] >= 1000000,2])
    bssids2 = set(wifi_tr[wifi_tr[:,1] < 1000000,2])
    bssids = bssids.intersection(bssids2)
    rows = [i for i in range(wifi_tr.shape[0]) if wifi_tr[i,2] in bssids]
    wifi_tr = wifi_tr[rows,:]
    wifi_xy = wifi_xy[rows,:]

    # renumber bssids
    bssids = pd.DataFrame({'bssid':wifi_tr[:,2]})
    wifi_tr[:,2] = np.array(bssids['bssid'].astype('category').cat.codes)

    # format data
    df = pd.DataFrame(wifi_tr[:,[0, 1, 2, 3, 4, 5]])
    df.columns = ['ts', 'id', 'bssid','rssi','path','f']
    df['x'] = wifi_xy[:,0]
    df['y'] = wifi_xy[:,1]
    x = pd.pivot_table(df, values='rssi', index='id', columns='bssid', aggfunc=np.sum, fill_value=-10000).reset_index()

    # split into train/valid
    x_tr = np.array(x.loc[x['id'] < 1000000], dtype=np.int32)
    x_val = np.array(x.loc[x['id'] >= 1000000], dtype=np.int32)

    # process all val points in 1 pass
    x_val2 = x_val.reshape(-1)
    x_val2[x_val2 == -10000] = 10000
    x_val = x_val2.reshape(x_val.shape)

    # process in chunks
    x_val0 = x_val.copy() # save
    chunk_size = int(5.e9 / 3. / 4. / x_tr.shape[0] / x_tr.shape[1]) # back into 5 Gb total
    id1  = x_tr[:,0]   # id of tr points
    x_tr = x_tr[:,1:]  # drop id
    x_tr = x_tr.reshape(x_tr.shape[0], 1, x_tr.shape[1])
    for i in range(1 + x_val.shape[0]//chunk_size): # loop over chunks
        if i%20 == 0:
            print('   ', i * chunk_size, x_val0.shape, int(time.time() - start_time), 'sec')
    
        x_val = x_val0[i*chunk_size:(i+1)*chunk_size,:].copy()
        
        id0   = x_val[:,0]  # id of val points
        x_val = x_val[:,1:] # drop id
        x_val = x_val.reshape(1, x_val.shape[0], x_val.shape[1])

        # find closest match of rec in x_tr
        x1 = np.abs(x_tr - x_val)
        x1a = x1 < 200
        # count of bssid matches
        x2 = x1a.sum(axis=-1)
        # diff for matched bssids
        x3 = (x1a * x1).sum(axis=-1)
        
        # turn results into dataframe
        df1 = pd.DataFrame({'id0':np.tile(id0, id1.shape[0]), 'cc':x2.ravel(), 'id':np.repeat(id1, id0.shape[0]), 'diff':x3.ravel()})
        
        # select closest matches for each match count
        df1['m'] = 28 * df1['cc'] - df1['diff']
        df2 = df1.groupby(['id0'])['m'].max().reset_index()
        df2.columns = ['id0','m2']
        df1 = df1.merge(df2, how='left', on='id0')
        df1 = df1.loc[df1['m'] >= df1['m2']].reset_index(drop=True)
        df1.drop(['m2', 'm'], axis=1, inplace=True)

        # append to total
        df1_tot = df1_tot.append(df1)
print('finish main fingerprinting loop', df1_tot.shape, int(time.time() - start_time), 'sec')
del x3, x2, x1a, x1, x_val, x_tr, x_val0
gc.collect()

# bring in coordinates
df = pd.DataFrame(wifi_tr0[:,[0, 1, 2, 3, 4, 5]])
df.columns = ['ts', 'id', 'bssid','rssi', 'path','f']
df['x'] = wifi_xy0[:df.shape[0],0]
df['y'] = wifi_xy0[:df.shape[0],1]
df_xy = df.groupby('id')[['x','y','f']].mean().reset_index()
df1_tot = df1_tot.merge(df_xy, how='left', on='id')

# weight parameters
cc_di = {} # multiple of cc, tabulated
cc_l = [1,1,1,1,1,1,1,1,1,1,1.2,37,60,60,230,260,260,273,440,440,720,720]
for i in range(22):
    cc_di[i] = cc_l[i]
diff_mult = 23.9

# make predicted floor the same for all points on the same path
def f_pred_path(dft):# this replaces f1 with average floor per path
    dft1 = pd.DataFrame(wifi_tr0[:,[1, 4]])
    dft1.columns = ['id0', 'path']
    dft1 = dft1.loc[dft1['path'] >= 100000] # select test from total
    dft2 = dft1.groupby('id0').mean().reset_index()
    dft3 = dft[['id0','f1']].merge(dft2, how='left', on='id0')
    dft4 = dft3.groupby('path')['f1'].mean().reset_index()
    dft4['f1'] = np.round(dft4['f1'], 0).astype('int32') # round to nearest. path, f1 - no dups.
    dft.drop('f1', axis=1, inplace=True)
    dft5 = dft2.merge(dft4, how='inner', on='path') # id0, path, f1
    dft = dft.merge(dft5[['id0','f1']], how='left', on='id0')
    return dft
    
# bring in relative prediction into df_xy_pred: id, x, y
dft = pd.DataFrame(wifi_tr0[:,[0, 1, 4]])
dft.columns = ['ts', 'id', 'path']
df_xy_pred = dft.groupby('id').mean().reset_index()
dtypes = {'ts':'int32', 'x_p':'float32', 'y_p':'float32', 'path':'int32'}
df_xy_pred = df_xy_pred.loc[df_xy_pred['path'] >= 100000].reset_index(drop=True) # select test from total
df_dr = pd.DataFrame(ahrs[:,[0, 5, 6, 4]]) # relative prediction *********************************
paths = np.array(df_xy_pred['path'], dtype=np.int32) - 100000
tss = np.array(df_xy_pred['ts'], dtype=np.int32)
df_xy_pred.drop(['ts', 'path'], axis=1, inplace=True)
y_te = np.zeros([tss.shape[0], 2])

# now only select data for wifi points (relative prediction was for sensor timestamps)
path0 = -1
df3a_np = np.array(df_dr)
for i in range(y_te.shape[0]):
    path = paths[i]
    ts   = tss[i]
    if path != path0:
        d = df3a_np[df3a_np[:,3] == path,:]
        offset = (df3a_np[:,3] == path).argmax()
        path0 = path
    if ts <= d[0,0]:
        y_te[i,0] = d[0, 1]
        y_te[i,1] = d[0, 2]
    elif ts >= d[-1,0]:
        y_te[i,0] = d[-1, 1]
        y_te[i,1] = d[-1, 2]
    else:# interpolate between 2 surrounding points
        k2 = ((d[:,0] - ts) > 0).argmax()
        k1 = k2 - 1
        w1 = (d[k2,0] - ts)/ (d[k2,0] - d[k1,0])
        y_te[i,0] = d[k1, 1] * w1 + d[k2, 1] * (1 - w1)
        y_te[i,1] = d[k1, 2] * w1 + d[k2, 2] * (1 - w1)
print('prepared df_xy_pred', int(time.time() - start_time), 'sec')
del df3a_np
gc.collect()
df_xy_pred['x'] = y_te[:,0]
df_xy_pred['y'] = y_te[:,1]
df_xy_pred.columns = ['id0', 'x', 'y'] # use id0 here for easier merge
    


# predict in batches based on DR with offset; use x0/y0 as val DR.
# add adjacent points to form a batch. Add offset to them.

# bring in pred coordinates - need them for offset
df1_tot = df1_tot.merge(df_xy_pred, how='left', on='id0')
df1_tot.columns = ['id0', 'cc', 'id', 'diff', 'x', 'y', 'f', 'x0', 'y0']

# bring in path for each id
df_p = df.groupby('id')['path'].mean().reset_index()
paths = np.array(df_p['path'])
def in_1(x):
    return x in ids2

df1_tot0 = df1_tot.copy() # save
max_offset = 90 # only add points withing this distance of current
outlier = 18
for shift in range(1, 43): # only add up to 42 points from before/after (up to 85 total)
    # next point on the same path
    ids = np.array(df_p['id'].iloc[shift:]) # skip first - it is never next
    ids2 = set(ids[paths[shift:] == paths[:-shift]]) # this is the list of ids that can be reduced by 1 and still be on the same path
    df1_tot_m = df1_tot0.loc[df1_tot0['id0'].map(in_1)].copy()
    df1_tot_m['id0'] -= shift # make it the same as base
    # get offset for it
    df1_tot_m = df1_tot_m.merge(df_xy_pred, how='left', on='id0')
    df1_tot_m.columns = ['id0', 'cc', 'id', 'diff', 'x', 'y', 'f', 'x0', 'y0', 'x0a', 'y0a']
    # add offset
    df1_tot_m['x'] -= df1_tot_m['x0'] - df1_tot_m['x0a']
    df1_tot_m['y'] -= df1_tot_m['y0'] - df1_tot_m['y0a']
    # only keep if offset < max_offset
    idx = ((df1_tot_m['x0'] - df1_tot_m['x0a'])**2 + (df1_tot_m['y0'] - df1_tot_m['y0a'])**2) < max_offset**2
    df1_tot_m = df1_tot_m.loc[idx].reset_index(drop=True)
    # append next point
    df1_tot_m.drop(['x0a', 'y0a'], axis=1, inplace=True)
    df1_tot = df1_tot.append(df1_tot_m).reset_index(drop=True)

    # prev point on the same path
    ids = np.array(df_p['id'].iloc[:-shift]) # skip last - it is never previous
    ids2 = set(ids[paths[shift:] == paths[:-shift]])  # this is the list of ids that can be increased by 1 and still be on the same path
    df1_tot_p = df1_tot0.loc[df1_tot0['id0'].map(in_1)].copy()
    df1_tot_p['id0'] += shift # make it the same as base
    # get offset for it
    df1_tot_p = df1_tot_p.merge(df_xy_pred, how='left', on='id0')
    df1_tot_p.columns = ['id0', 'cc', 'id', 'diff', 'x', 'y', 'f', 'x0', 'y0', 'x0a', 'y0a']
    # add offset
    df1_tot_p['x'] -= df1_tot_p['x0'] - df1_tot_p['x0a']
    df1_tot_p['y'] -= df1_tot_p['y0'] - df1_tot_p['y0a']
    # only keep if offset < max_offset
    idx = ((df1_tot_p['x0'] - df1_tot_p['x0a'])**2 + (df1_tot_p['y0'] - df1_tot_p['y0a'])**2) < max_offset**2
    df1_tot_p = df1_tot_p.loc[idx].reset_index(drop=True)
    # append prev point
    df1_tot_p.drop(['x0a', 'y0a'], axis=1, inplace=True)
    df1_tot = df1_tot.append(df1_tot_p).reset_index(drop=True)


# calc score - raw
# weight of each point
df1_tot['w'] = (np.exp(- df1_tot['diff']/diff_mult) * df1_tot['cc'].map(cc_di)).astype('float32')
df1_tot['x1'] = (df1_tot['w'] * df1_tot['x']).astype('float32')
df1_tot['y1'] = (df1_tot['w'] * df1_tot['y']).astype('float32')
df1_tot['f1'] = (df1_tot['w'] * df1_tot['f']).astype('float32')
df2 = df1_tot.groupby('id0')[['w', 'x1', 'y1', 'f1']].sum().reset_index()
df1_tot.drop(['x1', 'y1', 'f1'], axis=1, inplace=True)
df2['x1'] = df2['x1'] / df2['w']
df2['y1'] = df2['y1'] / df2['w']
df2['f1'] = df2['f1'] / df2['w']

# calc score - drop outliers
df1_tot = df1_tot.merge(df2[['id0', 'x1', 'y1', 'f1']], how='left', on='id0') # adds x1, y1
dist = np.sqrt((df1_tot['x'] - df1_tot['x1'])**2 + (df1_tot['y'] - df1_tot['y1'])**2)
df1_tot['x1'] = (df1_tot['w'] * df1_tot['x']).astype('float32')
df1_tot['y1'] = (df1_tot['w'] * df1_tot['y']).astype('float32')
df1_tot['f1'] = (df1_tot['w'] * df1_tot['f']).astype('float32')
df2 = df1_tot.loc[dist < outlier].groupby('id0')[['w', 'x1', 'y1', 'f1']].sum().reset_index() # drop outliers here
df1_tot.drop(['w', 'x1', 'y1', 'f1'], axis=1, inplace=True)
df2['x1'] = df2['x1'] / df2['w']
df2['y1'] = df2['y1'] / df2['w']
df2['f1'] = df2['f1'] / df2['w']
df2 = f_pred_path(df2) # make predicted floor the same for all points on the same path


    
# put predictions into df_dr
print('put predictions into df_dr - start', int(time.time() - start_time), 'sec')
df_tp = df.groupby('id')[['ts','path']].mean().reset_index()
df2 = df2.merge(df_tp, how='left', left_on='id0', right_on='id')
x_p = np.array(df_dr[1])
y_p = np.array(df_dr[2])
df_dr[3] += 100000
for p in df2['path'].unique():
    d = df2.loc[df2['path'] == p].reset_index(drop=True)
    o1 = (df_dr[3] == p).argmax()
    o2 = (df_dr[3] == p).sum() + o1
    # start
    n1 = (df_dr[0].iloc[o1:o2] < d['ts'].iat[0]).sum()
    x_p[o1:o1+n1] += d['x1'].iat[0] - x_p[o1+n1]
    y_p[o1:o1+n1] += d['y1'].iat[0] - y_p[o1+n1]
    for i in range(1, d.shape[0]): # i is end of the range
        n2 = (df_dr[0].iloc[o1:o2] < d['ts'].iat[i]).sum()
        t = np.array(df_dr[0].iloc[o1+n1:o1+n2])
        t = (t- t[0])/ (t[-1] - t[0]) # 0 to 1
        x_p[o1+n1:o1+n2] += (d['x1'].iat[i-1] - x_p[o1+n1]) + t * ((d['x1'].iat[i] - x_p[o1+n2-1]) - (d['x1'].iat[i-1] - x_p[o1+n1]))
        y_p[o1+n1:o1+n2] += (d['y1'].iat[i-1] - y_p[o1+n1]) + t * ((d['y1'].iat[i] - y_p[o1+n2-1]) - (d['y1'].iat[i-1] - y_p[o1+n1]))
        n1 = n2
    # end
    x_p[o1+n1:o2] += d['x1'].iat[i] - x_p[o1+n1]
    y_p[o1+n1:o2] += d['y1'].iat[i] - y_p[o1+n1]
df_dr[1] = x_p
df_dr[2] = y_p
df_dr.columns = ['ts','x_p','y_p','path']
df2a = df2.groupby('path')['f1'].mean().reset_index()
df_dr = df_dr.merge(df2a[['path','f1']], how='left', on='path')
print('put predictions into df_dr - end', int(time.time() - start_time), 'sec') 


# now only select data for waypoints
df3a = df_dr[['ts','path','x_p','y_p','f1']]
df3a.columns = ['ts', 'path', 'x_p', 'y_p', 'f_p']
path0 = -1
df3a_np = np.array(df3a[['ts', 'x_p', 'y_p', 'f_p','path']], dtype=np.float32)
for i in range(sub.shape[0]):
    path = sub['path'].iat[i]
    ts   = sub['ts'].iat[i]

    if path != path0:
        d = df3a_np[df3a_np[:,4] - 100000 == path,:]
        path0 = path
    sub['floor'].iat[i] = d[0,3]

    if ts <= d[0,0]:
        sub['x'].iat[i] = d[0, 1]
        sub['y'].iat[i] = d[0, 2]
    elif ts >= d[-1,0]:
        sub['x'].iat[i] = d[-1, 1]
        sub['y'].iat[i] = d[-1, 2]
    else:# interpolate between 2 surrounding wifi points
        k2 = ((d[:,0] - ts) > 0).argmax()
        k1 = k2 - 1
        w1 = (d[k2,0] - ts)/ (d[k2,0] - d[k1,0])
        sub['x'].iat[i] = d[k1, 1] * w1 + d[k2, 1] * (1 - w1)
        sub['y'].iat[i] = d[k1, 2] * w1 + d[k2, 2] * (1 - w1)





######################################################################################
# part 4 - post-processing ###########################################################
######################################################################################

# post-processing parameters
threshold = 5   # snap to grid if dist to grid point is < x
step_mult = 0.6 # snap next point on the path if dist to grid is < x * dist to current path point

# save starting prediction
sub['x1'] = sub['x']
sub['y1'] = sub['y']

# drop duplicate waypoints
train_waypoints = train_waypoints.sort_values(by=['site','floor','x','y'])
train_waypoints = train_waypoints.drop_duplicates(subset=['site','floor','x','y'], ignore_index=True)

def add_xy(df): # add x/y
    df['xy'] = [(x, y) for x,y in zip(df['x'], df['y'])]
    return df

train_waypoints = add_xy(train_waypoints)

def closest_point(point, points): # find closest point from a list of points
    return points[cdist([point], points).argmin()]


# snap to grid
sub.drop('path', axis=1, inplace=True)
sub = pd.concat([sub['site_path_timestamp'].str.split('_', expand=True).rename(columns={0:'site',1:'path',2:'timestamp'}), sub], axis=1).copy()
for N in range(20):# loop until converges
    ds = []
    sub = add_xy(sub)
    for (site, myfloor), d in sub.groupby(['site','floor']):
        idx = (train_waypoints['floor'] == myfloor) & (train_waypoints['site'] == site)
        true_floor_locs = train_waypoints.loc[idx].reset_index(drop=True)
        d['matched_point'] = [closest_point(x, list(true_floor_locs['xy'])) for x in d['xy']]
        d['x_'] = d['matched_point'].apply(lambda x: x[0])
        d['y_'] = d['matched_point'].apply(lambda x: x[1])
        ds.append(d)
    sub = pd.concat(ds)
    sub['dist'] = np.sqrt( (sub.x-sub.x_)**2 + (sub.y-sub.y_)**2 )

    # Snap to grid if within a threshold.
    sub['_x_'] = sub['x']
    sub['_y_'] = sub['y']
    idx = sub['dist'] < threshold
    sub.loc[idx, '_x_'] = sub.loc[idx]['x_']
    sub.loc[idx, '_y_'] = sub.loc[idx]['y_']
        
    # shift each path by mean shift, snap again
    dft = sub.groupby('path')[['x','_x_','y','_y_']].mean().reset_index()
    dft['dx'] = dft['_x_'] - dft['x']
    dft['dy'] = dft['_y_'] - dft['y']
    sub = sub.merge(dft[['path','dx','dy']], how='left', on='path')
    sub['x'] = sub['x'] + sub['dx']
    sub['y'] = sub['y'] + sub['dy']
    sub = add_xy(sub)
    sub.drop(['dx','dy'], axis=1, inplace=True)


# proceed 1 step at a time
for N in range(5):# loop until converges
    # pass forward
    sub['x2'] = sub['_x_'] # init to best prediction
    sub['y2'] = sub['_y_']
    sub['t'] = 0
    for i in range(0, sub.shape[0]):
        if i == 0 or sub['path'].iat[i] != sub['path'].iat[i-1]:# process new path
            site = sub['site'].iat[i]
            myfloor = sub['floor'].iat[i]
            idx = (train_waypoints['floor'] == myfloor) & (train_waypoints['site'] == site)
            true_floor_locs = train_waypoints.loc[idx].reset_index(drop=True)
            points = list(true_floor_locs['xy'])
            x = sub['x2'].iat[i]
            y = sub['y2'].iat[i]
            d0 = np.sqrt((sub['x1'].iat[i] - sub['x1'].iat[i+1])**2 + (sub['y1'].iat[i] - sub['y1'].iat[i+1])**2)
        else: # get 1-step predicted current point: last point + dPDR
            x = sub['x2'].iat[i-1] + sub['x1'].iat[i] - sub['x1'].iat[i-1]
            y = sub['y2'].iat[i-1] + sub['y1'].iat[i] - sub['y1'].iat[i-1]
            d0 = np.sqrt((sub['x1'].iat[i] - sub['x1'].iat[i-1])**2 + (sub['y1'].iat[i] - sub['y1'].iat[i-1])**2)
        # find closest grid point to it
        dists = cdist([(x,y)], points)
        ii = dists.argmin()
        p = points[ii]
        dist = dists.min()
        if dist < d0 * step_mult: # if grid point is close, snap to it
            sub['x2'].iat[i] = p[0]
            sub['y2'].iat[i] = p[1]
            sub['t'].iat[i] = 1
    sub['_x_'] = sub['x2'] # put this in final sub
    sub['_y_'] = sub['y2']

    # pass backward
    sub['x3'] = sub['_x_'] # init to best pred
    sub['y3'] = sub['_y_']
    sub['t'] = 0
    for i in range(sub.shape[0] - 1, 0, -1):
        if i == sub.shape[0] - 1 or sub['path'].iat[i] != sub['path'].iat[i+1]:# process new path
            site = sub['site'].iat[i]
            myfloor = sub['floor'].iat[i]
            idx = (train_waypoints['floor'] == myfloor) & (train_waypoints['site'] == site)
            true_floor_locs = train_waypoints.loc[idx].reset_index(drop=True)
            points = list(true_floor_locs['xy'])
            x = sub['x3'].iat[i]
            y = sub['y3'].iat[i]
            d0 = np.sqrt((sub['x1'].iat[i] - sub['x1'].iat[i-1])**2 + (sub['y1'].iat[i] - sub['y1'].iat[i-1])**2)
        else: # get 1-step predicted current point: last point + dPDR
            x = sub['x3'].iat[i+1] + sub['x1'].iat[i] - sub['x1'].iat[i+1]
            y = sub['y3'].iat[i+1] + sub['y1'].iat[i] - sub['y1'].iat[i+1]
            d0 = np.sqrt((sub['x1'].iat[i] - sub['x1'].iat[i+1])**2 + (sub['y1'].iat[i] - sub['y1'].iat[i+1])**2)
        # find closest grid point to it
        dists = cdist([(x,y)], points)
        ii = dists.argmin()
        p = points[ii]
        dist = dists.min()
        if dist < d0 * step_mult: # if grid point is close, snap to it
            sub['x3'].iat[i] = p[0]
            sub['y3'].iat[i] = p[1]
            sub['t'].iat[i] = 1
    sub['_x_'] = sub['x3'] # put this in final sub
    sub['_y_'] = sub['y3']
# blend forward/backward 50/50
sub['_x_'] = (sub['x3'] + sub['x2']) / 2
sub['_y_'] = (sub['y3'] + sub['y2']) / 2



# save submission
sub.drop(['x','y'], axis=1, inplace=True)
sub = sub.rename(columns={'_x_':'x', '_y_':'y'})
sub[['site_path_timestamp','floor','x','y']].to_csv('submission.csv', index=False)
print('Finished', int(time.time() - start_time), 'sec')