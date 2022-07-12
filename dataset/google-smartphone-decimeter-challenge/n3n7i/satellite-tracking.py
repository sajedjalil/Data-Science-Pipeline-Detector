# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os

import time

start = time.time()

sys.path.append("../input/sp-datafun-dev/");
import datatools as dt;

sys.path.append("../input/centext/");
import centExt as cE;

sys.path.append("../input/sp-quadr/");
import quadr as qd;

if os.path.exists("/kaggle/temp/")==False:
    os.mkdir("/kaggle/temp/");

class satC2(cE.satClass):
    
    def xThres(self, data, data2, lim):        
        #print("");
        return np.sum((data - data2)**2, axis=1) < lim**2;
    
    def xGroup_init(self, data, field, xs, lim=50):
        
        dx = data; #catchd[:,4:7];
        alt_res = np.zeros(data.shape[0], int) -1; #n

        j = 0;
        self.append([dx[0,:]]);
        while (np.sum(alt_res==-1)>0):
#for i in range(len(k2.collection)):
            c=self.collection[j].ext[field]; #"vel"];
            clim = cE.euclid(c) * xs; #/4.0    
            rVal = self.xThres(dx, c, clim);
            alt_res[rVal & (alt_res==-1)] = j;
            if np.sum(alt_res==-1)>0:
                self.append([dx[alt_res==-1,:][0,:]]);
            j+=1;
            print("iter", j, np.sum(alt_res==-1));
            if(j>lim): break;
                
        return alt_res;

    def xyGroup_init(self, data, field, data2, xs, lim=50):
        
        dx = data; #catchd[:,4:7];
        alt_res = np.zeros(data.shape[0], int) -1; #n

        j = 0;
        self.append([dx[0,:]]);
        while (np.sum(alt_res==-1)>0):
#for i in range(len(k2.collection)):
            c=self.collection[j].ext[field]; #"vel"];
            cl=data2[alt_res==-1,:][0,:]; #self.collection[j].ext[f2];
        
            clim = cE.euclid(cl) * xs; #/4.0    
            rVal = self.xThres(dx, c, clim);
            alt_res[rVal & (alt_res==-1)] = j;
            if np.sum(alt_res==-1)>0:
                self.append([dx[alt_res==-1,:][0,:]]);
            j+=1;
            print("iter", j, np.sum(alt_res==-1));
            if(j>lim): break;
                
        return alt_res;

#dt.putGnss(dt.paths,0);

#traw = pd.read_csv("/kaggle/temp/Gnss/Raw");
tder = dt.getDeriv(dt.paths,0);
tbase = dt.getBaseline(dt.xtrain, dt.paths, 0);

def xprint(strlist, s=5):
    i=0; 
    while(i+s<len(strlist)):
        print(strlist[i:i+s]);
        i+=s;
    print(strlist[i:], "\n");

def logVec(a, b):    
    return np.array([i in a for i in b]);
    
dlab = dt.getLabels(tder);
#rlab = dt.getLabels(traw);
#baselab = dt.getLabels(tbase);
#xprint(rlab);# :2
xprint(dlab);#7:9 | 10:12, :2
xprint(dt.tlab); #3:4, :2
#xprint(baselab);

catchd = np.array(tder)[:,logVec([7,8,9,10,11,12], range(20))];
catchbase = np.array(tbase)[:,logVec([3,4], range(tbase.shape[1]))];

k2 = cE.satClass(["pos", "vel", "alt", "speed"]);

kx = satC2(["vel"]);


def samp(data, xid):
    return [data[xid,:3], data[xid,4:], cE.euclid(data[xid,:3]), cE.euclid(data[xid,4:])];

k2.append([catchd[0,:3], catchd[0,4:], cE.euclid(catchd[0,:3]), cE.euclid(catchd[0,4:])]);

print(k2.inrange("pos", samp(catchd,1)[0], "vel"));

ccount=1;
ncount=0;

n = catchd.shape[0];

res = np.zeros(n, int) -1;

print(f'Elapsed Time: {time.time() - start}')
#start = time.time();
"""
for i in range(1, n):
    r = k2.thres("vel", samp(catchd,i)[1], samp(catchd,i)[3]/(2.0**1)) ##0.5-1.5xVel
    ncount += r==-1;
    if (r==-1) & (ccount<64):
        k2.append(samp(catchd, i));
        r=ccount;
        ccount+=1;
    res[i] = r;
    #print(r);
print("nclas", ncount, "from", ccount, "of 55k");
#""";

print(f'Elapsed Time: {time.time() - start}')

dx = catchd[:,4:7]; #vel

dx2 = catchd[:,1:4]; #pos
kxb = satC2(["pos"]);

"""
alt_res = np.zeros(n, int) -1;

j = 0;
kx.append([dx[0,:]]);
while (np.sum(alt_res==-1)>0):
#for i in range(len(k2.collection)):
    c=kx.collection[j].ext["vel"];
    clim = cE.euclid(c)/4.0;    
    rVal = kx.xThres(dx, c, clim);
    alt_res[rVal & (alt_res==-1)] = j;
    if np.sum(alt_res==-1)>0:
        kx.append([dx[alt_res==-1,:][0,:]]);
    j+=1;
    print("iter", j, np.sum(alt_res==-1));
    if(j>50): break;
    
""";
alt_res = kx.xGroup_init(dx, "vel", 1/4.0); ##+-25%

alt_resb = kxb.xyGroup_init(dx2, "pos", dx, 300);##velx300 #1/8.0); ## 1/16 alt?

print(np.sum((alt_res!=-1) & (alt_resb!=-1)));

print(np.sum(alt_res == res), "of", n, "xThres\n\n");
print(np.sum(alt_res == alt_resb), "of", n, "xyGroup\n\n");
print(f'Elapsed Time: {time.time() - start}')

res = alt_res;

def mm_nano(x):
    return np.max(x) - np.min(x);


for i in range(50):
    for j in range(50):
        res2 = (alt_res==i) & (alt_resb == j);
        c = np.sum(res2);
        if(c>0):
            zdata = catchd[res2, 0];
            print(i, j, c, mm_nano(zdata)/1000);
            print("duplicate or pair: ", c > ((mm_nano(zdata)/1000)*2), "split: ", (2*c) < (mm_nano(zdata)/1000));


def std(x):
    return [np.std(x[:,0]), np.std(x[:,1]), np.std(x[:,2]),  np.std(x[:,3]), np.std(x[:,4]), np.std(x[:,5])]

def ms(x):
    return [np.mean(x[:,3]), np.mean(x[:,4]), np.mean(x[:,5]),  np.std(x[:,3]), np.std(x[:,4]), np.std(x[:,5])]

for i in range(64):
    groupx = res == i;
    catchgroup = catchd[groupx, :];
    if catchgroup.shape[0]>30:
        print("iter:",i, "count", catchgroup.shape);
        print(ms(catchgroup));

    
groupx = res == 4;
catchgroup = catchd[groupx, :];

##---

k3 = cE.satClass(["pos", "vel", "alt", "speed", "nanos"]);

def subsamp(data, xid):
    return [data[xid,1:4], data[xid,4:7], cE.euclid(data[xid,1:4]), cE.euclid(data[xid,4:7]), data[xid,0]];

subcatch = np.array(tder)[groupx, :][:, logVec([2, 7,8,9,10,11,12], range(20))];

k3.append(subsamp(subcatch, 0));#subcatch[0,:]);

subn = subcatch.shape[0];
subres = np.zeros(subn, int) -1;
ncount=0;
ccount=0;

for i in range(1, subn): ##rerun position
    r = k3.thres("pos", subsamp(subcatch,i)[0], subsamp(subcatch,i)[3]*180.0)
    ncount += r==-1;
    if (r==-1) & (ccount<64):
        k3.append(subsamp(subcatch, i));
        r=ccount;
        ccount+=1;
    subres[i] = r;

print("nclas", ncount, "from", ccount, "of", subn);
print(f'Elapsed Time: {time.time() - start}')


"""
for i in range(64):
    groupx = subres == i;
    gc = np.sum(groupx);
    if gc>0:
        #catchgroup = catchd[groupx, :];
        print("iter:",i, "count", gc);
        if gc>30:
            print(ms(subcatch[groupx,:]));
            print(mm_nano(subcatch[groupx, 0]));
    
print("sample \n", subcatch[subres==6, :][:10, :]);
print("sample \n", subcatch[subres==6, :][-10:, :]);

""";
f = qd.wbqf([], []);

mxval = np.array([0,0,0,0]);

sc = subcatch[subres==2, :];

print("\nvelgroup 4 position 2 | 0:60 of ", np.sum(subres==2), "\n");

sum_err = np.array([0,0,0]);
mse = 0;
for i in range(0,60):
    #print(f.surfquad(sc[0, 1:4], sc[29, 1:4],sc[59, 1:4], 1/60), sc[1,1:4]);
    #est= f.quad(sc[0, 1:4], sc[29, 1:4],sc[59, 1:4], i/59.0);
    #scII = f.lin(sc[29, 1:4], sc[30, 1:4], 0.5); 
    
    scII = f.surfquad(sc[28, 1:4], sc[29, 1:4], sc[30, 1:4], 0.75); 
    #estII= f.surfquad(sc[0, 1:4], sc[29, 1:4], sc[59, 1:4], i/59.0);    
    est= f.quad(sc[0, 1:4], scII, sc[59, 1:4], i/59.0);
    estII= f.surfquad(sc[0, 1:4], scII, sc[59, 1:4], i/59.0);    
    eIII = sc[0,1:4] + i*sc[0, 4:7];
    eIV = f.lin(sc[0, 1:4], sc[59, 1:4], i/59.0)
    mxval = np.maximum([cE.euclid(est - sc[i, 1:4]), cE.euclid(estII - sc[i, 1:4]), cE.euclid(eIII - sc[i, 1:4]), cE.euclid(eIV - sc[i, 1:4])], mxval);
    sum_err = sum_err + (estII - sc[i, 1:4]);
    mse += (cE.euclid(estII - sc[i, 1:4])*10)**2;
    print("step:", i);
    print("quad est:", cE.euclid(est - sc[i, 1:4]), ", surf est:", cE.euclid(estII - sc[i, 1:4]));
    print("basic est:", cE.euclid(eIII - sc[i, 1:4]), ", lin est:", cE.euclid(eIV - sc[i, 1:4]));
    if i==29:
        print("time displace:", (estII - sc[i, 1:4]) / sc[i, 4:7], (estII - sc[i, 1:4]), sc[i, 4:7]);
    #print(est, sc[i, 1:4]);

print("linear speed?",cE.euclid(sc[0,1:4] - sc[59, 1:4])/59.0, cE.euclid(sc[0,4:7]));
print("peak err:", mxval);
print("accum err:", sum_err/60, "mse:", mse / 60, "dm^2");
#print(np.std(catchgroup, 0));
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

##----------
"""
bias = sum_err/60.0;

sum_err = np.array([0,0,0]);
mse = 0;
for i in range(0,60):
    #print(f.surfquad(sc[0, 1:4], sc[29, 1:4],sc[59, 1:4], 1/60), sc[1,1:4]);
    #est= f.quad(sc[0, 1:4], sc[29, 1:4],sc[59, 1:4], i/59.0);
    scII = f.lin(sc[29, 1:4] - bias, sc[30, 1:4] - bias, 0.5); 
    
    estII= f.surfquad(sc[0, 1:4]-bias, scII, sc[59, 1:4]-bias, i/59.0);    
    
    sum_err = sum_err + np.abs(estII - sc[i, 1:4]);
    mse += (cE.euclid(estII - sc[i, 1:4])*10)**2;
    #print(cE.euclid(estII - sc[i, 1:4]), estII - sc[i, 1:4]);
    
print("accum err per item:", sum_err/60, "mseII:", mse / 60, "dm^2");

print(f'Elapsed Time: {time.time() - start}'); """;