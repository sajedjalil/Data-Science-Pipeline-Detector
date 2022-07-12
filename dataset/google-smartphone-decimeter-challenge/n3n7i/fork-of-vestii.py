# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import sys
#import csv

sys.path.append('/kaggle/input/vest-loc');
import vestLoc;

vestLoc.Vest();
print("version 0.2\n");

"""for dirname, _, filenames in os.walk('/kaggle/input/sp-stump-dmprec/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
""";



l1 = pd.read_csv("/kaggle/input/sp-stump-dmprec/baseline_sample.csv")
l2 = pd.read_csv("/kaggle/input/sp-stump-dmprec/sat_sample.csv")

h1 = [x for x in l1];
h2 = [x for x in l2];

l1 = np.array(l1);
l2 = np.array(l2);

l1 = l1[:, 1:];
l2 = l2[:, 1:];

h1 = h1[1:];
h2 = h2[1:];



## correctedPrM = rawPrM(15) + satClkBiasM(13) - isrbM(17) - ionoDelayM(18) - tropoDelayM(19)

## C_ms = 299792458.0;

print("c-PrM:", h2[13:]);

def range_Est(samp):
    d = (samp[13] + samp[15]) - (samp[17] + samp[18] + samp[19]); #/1000
    flight = d / 299792458.0;
    return d; #[d, flight];

"""def range_fields(samp):
    print("(", samp[13], "+", samp[15], ")",  samp[17], samp[18], samp[19]);
    #return [(flight * 299792458.0) / 1e9, flight];
    
def range_vals(samp):
    print("(", samp[13], "+", samp[15], ")",  samp[17], samp[18], samp[19]);
"""    
def PrM_table(data):
    r = np.zeros(data.shape[0]);
    for i in range(r.shape[0]):
        r[i] = range_Est(data[i,:]);
    return r;

def pos_table(data):
    return data[:,7:10];


"""range_fields(h2);

range_vals(l2[0,:]);
range_vals(l2[1,:]);
range_vals(l2[2,:]);
""";


def alt(x,y,z):
    return np.sqrt(np.sum(x**2 + y**2 + z**2));

def WGS84_to_ECEF(lat, lon, alt):
    # convert to radians
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a    = 6378137.0
    # f is the flattening factor
    finv = 298.257223563
    f = 1 / finv   
    # e is the eccentricity
    e2 = 1 - (1 - f) * (1 - f)    
    # N is the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))
    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (N * (1 - e2) + alt)        * np.sin(rad_lat)
    return x, y, z

"""def lla(samp):
    return [np.float(samp[3]), np.float(samp[4]), np.float(samp[5])];

def ecef(samp):
    return np.array([np.float(samp[7]), np.float(samp[8]), np.float(samp[9])]);

def logVec(inds, r):
    return [x in inds for x in range(r)];
""";

def ecef_table(data):
    r = np.zeros([data.shape[0], 3]);
    for i in range(r.shape[0]):
        r[i, :] = WGS84_to_ECEF(data[i,3], data[i,4],data[i,5]); #range_Est(data[i,:]);
    return r;


ps_range = PrM_table(l2);
ps_data = pos_table(l2);
loc_data = ecef_table(l1);
ts_data = l2[:,2];

def collect_Set(a,b,c, ts, i):

    x = pd.unique(ts)[i];
    rx = ts == x;
    return [a[rx,:], b[rx], c[i]];



def xDist(targ, data):
    f1 = np.sum((data - targ)**2, axis=1);
    for i in range(f1.shape[0]):
        f1[i] = np.sqrt(f1[i]);
    return f1;



def xUpdate(p, r, t):
    
    d_stat = xDist(t, p);
    e_stat = r - d_stat; ##! [short = +'ve // long = -'ve]

    return e_stat;

def eMetric(e):
    mE = np.mean(e)
    mae = np.mean(np.abs(e));
    mse = np.mean(e ** 2);
    mss = np.mean((e**2) * np.sign(e) );
    return [mE, mae, mse, mss, "e, mae, mse, mss"];

def xUpdate2(p,r,t, err, alpha, a2=0.5): ##Individual err
    
    r = r.reshape(r.shape[0], 1);
    err = err.reshape(err.shape[0], 1);
    #tp = t-p;
    #rer = (r+err) / r;
    #tpr = tp*rer;
    #step_err = ((t - p) * ((r+err) / r) + p) - t; ##short = +err ##! R is ~ correct !##
    
    step_err = ((t - p) * (r / (r-err)) + p) - t; ##short = +err ## ! Fixed [1.0/0.75 || 1.25/1.0]
    
    #step_alt = xDist(np.zeros(3), step_err);
    #print(np.mean(step_alt), "meanGrad");
    
    x_step = np.mean(step_err, axis=0)*alpha;
    #x_step2 = np.sum(step_err, axis=0)*a2;
    #print(alt(x_step[0], x_step[1], x_step[2]),"g", alt(x_step2[0], x_step2[1], x_step2[2]),"g2");
    #x_step = x_step * (np.mean(step_alt) / alt(x_step[0], x_step[1], x_step[2])) * a2;
    return t + x_step; #np.mean(step_err, axis=0)*alpha;
    
def xUpdate_alt(t, err):
    ta = alt(t[0], t[1], t[2]);
    return t * ((ta+err) / ta);
##-------------

def xSolver(iters, dataset, alpha=1.0):
    
    rI = dataset;
    nxT = dataset[2];
    for i in range(iters):
    #print("step ",i);
        e1 = xUpdate(rI[0], rI[1], nxT); ## mae Min?    
        e2 = (e1**3) / np.sum(e1**2); ##?    msse Min? /Square-weight?   
        e3 = (e1**2 * np.sign(e1)) / np.sum(e1); ## ?? /lin-w?
        nxT = xUpdate2(rI[0], rI[1], nxT, e1, alpha);    
        #if (np.sqrt(i)%1)==0: 
            #print(np.round(eMetric(e1)[0:4],4), "mse iter", i, np.mean(e2/e1));            
    return nxT;

def xSolve_Alt(iters, dataset):
    
    rI = dataset;
    nxT = dataset[2];
    for i in range(iters):
    #print("step ",i);
        e1 = xUpdate(rI[0], rI[1], nxT); ## mae Min?    
        e2 = (e1**3) / np.sum(e1**2); ##?    msse Min?    
        #nxT = xUpdate2(rI[0], rI[1], nxT, e2, 0.5);    
        
        nxT = xUpdate_alt(nxT, -np.mean(e1));  ## mean err Min?
        
        if (i%2)==0: 
            print(np.round(eMetric(e1)[0:4],4), "mse iter", i);            
    return nxT;

loc_main = loc_data;

loc_data = loc_main[20:40, :];

print("Timesteps: ", loc_main.shape[0]);
print("Blocksize", loc_data.shape[0]);

result_Pass = np.zeros([loc_data.shape[0], 3]);
result_Pass2 = np.zeros([loc_data.shape[0], 3]);
#result_Pass3 = np.zeros([loc_data.shape[0], 3]);

err_Pass = np.zeros([loc_data.shape[0], 11]);
err_Pass2 = np.zeros([loc_data.shape[0], 11]);

for i in range(loc_data.shape[0]):
    
    rII = collect_Set(ps_data, ps_range, loc_data, ts_data, i);
    
    result_Pass[i,:] = xSolver(25, rII);
    
    e1x = xUpdate(rII[0], rII[1], result_Pass[i,:]);
    
    err_Pass[i,:] = e1x;
        
    #print(e1x, e1x.shape, "\n");
    
print(":pass Comp:")

## sat Correction:?
s_Fix = np.mean(err_Pass, axis=0);
## rec Correction?
r_Fix = np.mean(err_Pass, axis=1);

print("mean err by axis", s_Fix, r_Fix)

s_Fix = np.std(err_Pass, axis=0);
r_Fix = np.std(err_Pass, axis=1);
print("dev:", s_Fix, r_Fix)

#"""
s_Fix = np.mean(err_Pass, axis=0);
s_corr = s_Fix;
r_Fix = np.mean(err_Pass, axis=1);
r_corr = r_Fix;

for i in range(loc_data.shape[0]):
    
    rII = collect_Set(ps_data, ps_range, loc_data, ts_data, i);
    
    rII[1] = rII[1] - (s_Fix);# + r_Fix[i]);#[i];
    
    result_Pass2[i,:] = xSolver(150*2, rII);
    
    e1x = xUpdate(rII[0], rII[1], result_Pass2[i,:]);
    
    err_Pass2[i,:] = e1x;
        
    #print(e1x, e1x.shape, "\n");
    
print(":passII Comp:")
print(np.mean(err_Pass2, axis=0), np.std(err_Pass2, axis=0))

print("a1: ", np.mean(err_Pass2, axis=1), np.std(err_Pass2, axis=1))

## sat Correction:?
s_Fix = np.mean(err_Pass2, axis=0);
## rec Correction?
r_Fix = np.mean(err_Pass2, axis=1);

"""
print("s_corr", s_corr);
print("mean err by axis", s_Fix, r_Fix)

s_Fix = np.std(err_Pass2, axis=0);
r_Fix = np.std(err_Pass2, axis=1);
print("dev:", s_Fix, r_Fix)

r_Fix = np.mean(err_Pass2, axis=1);
r_corr = r_Fix;
s_Fix = s_corr;

#s_Fix += np.mean(err_Pass2, axis=0);

#""
for i in range(loc_data.shape[0]):
    
    rII = collect_Set(ps_data, ps_range, loc_data, ts_data, i);
    
    rII[1] = rII[1] - (s_Fix + r_Fix[i]);#[i];
    
    result_Pass2[i,:] = xSolver(150*3, rII);
    
    e1x = xUpdate(rII[0], rII[1], result_Pass2[i,:]);
    
    err_Pass2[i,:] = e1x;
        
    print(e1x, e1x.shape, "\n");
    
print(":passIII Comp:")
"
#"

## rec Correction?
r_Fix = np.mean(err_Pass, axis=1);
print(r_Fix, np.std(err_Pass, axis=1));

"for i in range(loc_data.shape[0]):
    
    rII = collect_Set(ps_data, ps_range, result_Pass, ts_data, i);#ts_data, i);    
    result_Pass2[i,:] = xSolve_Alt(10, rII);
    print("\n\n");

print(":passII Comp:")


for i in range(loc_data.shape[0]):
    rII = collect_Set(ps_data, ps_range, result_Pass2, ts_data, i);    
    result_Pass3[i,:] = xSolver(50, rII);
    print("\n\n");    
print(":passIII Comp:")
""";
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session