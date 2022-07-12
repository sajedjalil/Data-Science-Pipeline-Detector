

from timeit import default_timer as timer
from datetime import timedelta

def ptime(xstr, s):
    print(xstr, ": ", timedelta(seconds=timer()-s));
    s = timer();
    pass;

st = timer();


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numba
        
import l5kit.geometry as l5geo
r33yaw = l5geo.rotation33_as_yaw;

import zarr

import sys, os
#from pathlib import Path

sys.path.append("../input/pycent/");
import pyCent as pkit

sys.path.append("../input/pysvg/");
import pySvg as gkit

ptime("import loading: ", st);

timeDEBUG = False;

def recordAccess(data, sc):
    rS = data["scenes"][sc];    
    rF = data["frames"][slice(rS[0][0], rS[0][1])];    
    rA = data["agents"][slice(rF[0][1][0], rF[-1][1][1])];
    return [rS, rF, rA];

@numba.stencil()
def filter1d(data):
    return (data[0] - data[-10]); ## ~1 sec delta

@numba.stencil()
def filter1d4(data):
    return (data[0] - data[-4]);


def translations(rf):
    
    ego_x = rf["ego_translation"][:, 0]
    ego_y = rf["ego_translation"][:, 1]
    
    #print(rf["ego_translation"].shape)
    
    #etx = filter1d(ego_x);
    #ety = filter1d(ego_y);
    
    return [ego_x, ego_y]; #[etx, ety];


#r33yaw = l5geo.rotation33_as_yaw;


def collectYaw(rf):
    
    ego_R = rf["ego_rotation"][:][:,:]
    
    #print(rf["ego_rotation"].shape)
    
    rvec = np.zeros(ego_R.shape[0]);
    for i in range(0, ego_R.shape[0]):
        rvec[i] = r33yaw(ego_R[i]);
    return rvec;


def rotations(r):
    etR = filter1d(r);
    return etR;


def frameAccess(data, sc):
    rS = data["scenes"][sc];    
    #rF = data["frames"][slice(rS[0][0], rS[0][1])]; 
    rF = data.frames[slice(rS[0][0], rS[0][1])]; 
    return [rS, rF];


def frame_Data(scene, rf):
    
    if timeDEBUG: ptime("fDInit: ", st);
    #rf = frameAccess(zdz, scene)[1];
    
    yaws = collectYaw(rf); # heading
    
    if timeDEBUG: ptime("fDYaw: ", st);
    
    t = translations(rf); # vels    
    #r = rotations(yaws); # heading'
    #v1d = np.hypot(t[0], t[1]); # combined vel
    if timeDEBUG: ptime("fDUpper: ", st);
    #accel = filter1d(v1d[10:]); #  vel'
    #rad_cel = filter1d(r[10:]); # heading''
    if timeDEBUG: ptime("fD: ", st);
    
    #vr = np.array([v1d[20:], accel[10:], r[20:], rad_cel[10:]]).T[:200, :];
    
    
    
    if timeDEBUG: ptime("fDEnd: ", st);
    
    #vr = np.array([t, yaws]).T;
    vr = np.vstack([t, yaws]);
    return vr;
    

def collectFrames(xr, zdz):

    tx = frameAccessII(zdz, [xr[0], xr[-1]+1]);
    
    xlist = [];
    splist = [];
    spl =0;
    for i in xr:
        
        if(i%500 == 1): print(" step ",i, end='');
        
        tn = frameAccessIII(tx, i);
        
        ##print(tn.shape)
        
        xlist.append(frame_Data(i,tn));
        
        spl += tn.shape[0];
        splist.append(spl);
        
        
    print('\n');
    
    return [xlist, splist];

def frameAccessII(data, sc):
    rS = data.scenes[slice(sc[0], sc[1])];
    if timeDEBUG: ptime("faII: ", st);
    
    #rF = data["frames"][slice(rS[0][0], rS[0][1])]; 
    rF = data.frames[slice(rS[0][0][0], rS[-1][0][1])];
    if timeDEBUG: ptime("faIIB: ", st);
    
    
    return [rS, rF];

def frameAccessIII(rx, sc):
    
    #print(len(rx))
    rS = rx[0][sc];    
    #rF = data["frames"][slice(rS[0][0], rS[0][1])]; 
    ##eSlice = np.min(rS[0][1] - rS[0][0], 240);
    
    rF = rx[1][slice(rS[0][0], rS[0][1])];     
    if timeDEBUG: ptime("fAIIIB ", st);
    return rF;



ptime("defines", st);

DATA_ROOT = "../input/lyft-motion-prediction-autonomous-vehicles"
zdx = zarr.open(DATA_ROOT+"/scenes/train.zarr", mode='r')

ptime("zdz open ", st);

print("Scenes:", zdx.scenes.shape);

print("Frames:", zdx.frames.shape);

#rf = recordAccess(zdz, 0)[1];

"""rf = recordAccess(zdz, 0)[1];

print(rf.dtype)

print(len(rf))

t = translations(rf); # vels

yaws = collectYaw(rf); # heading

r = rotations(yaws); # heading'

v1d = np.hypot(t[0], t[1]); # combined vel

accel = filter1d(v1d[10:]); #  vel'

rad_cel = filter1d(r[10:]); # heading''


vr = np.array([v1d[20:], accel[10:], r[20:], rad_cel[10:]]).T;

w = 1.0 / np.std(vr, axis=0);

m = np.mean(vr, axis=0);

k = pkit.centKitIV(vr - m, w);

k.pushCent(0);

k.distPass(0);

print(vr.shape) ##"""

try:
    
    setFrames = 1000; ##No of scenes
    
    print("Using ", setFrames);
    
    runonce == True ;
    
except:
    
    vrII, vnS = collectFrames(range(0,setFrames), zdx);
    
    r_unonce = True;
    
    ptime("runonce timer ", st);
    
    
#t = np.concatenate(vrII);

t = np.hstack(vrII);

print(t.shape)

t2 = np.copy(t);

ptime("cat: ", st);

for i in range(0,3):
    
    t2[i,:] = filter1d(t[i,:]);
    
tz = np.hypot(t2[0,:], t2[1,:]);

tz2 = np.vstack([tz, t2[2,:]]);

t3 = np.copy(tz2);

for i in range(0,2):
    
    t3[i,:] = filter1d(tz2[i,:]);

ptime("filters: ", st);

out3 = np.vstack([tz2, t3]);

print(out3.shape)
    
##ptime("filter_: ", st);

#s = 60*10;

s = setFrames * 11;

##err

##break;

#tx = np.split(out3[:, (out3.shape[1]%s):], s, axis=1);

tx2 = np.split(out3, vnS, axis=1)[:-1];

tx3 = [x[:, 20:240] for x in tx2];

tx4 = np.hstack(tx3);

tx = np.split(tx4, s, axis=1);

print(tx4.shape);

t4 = np.dstack(tx);

print(t4.shape, s);

print(np.array(tx4).shape, " #?#");

t5 = np.transpose(t4, (2,1,0));

print(t5.shape, "??")



g = gkit.Svg()

"""


g.drawkit_mat(t3[:,:,0], [600,900])

#g.fileout("./t3_0-velo")

g.drawkit_mat(t3[:,:,1], [600,900])

#g.fileout("./t3_1-accel")

g.drawkit_mat(t3[:,:,2]*100, [600,900])

#g.fileout("./t3_2-rota")

g.drawkit_mat(t3[:,:,3]*100, [600,900])

#g.fileout("./t3_3-rota-1")

ptime("nodraw strings gen: ", st);

""";



t3 = t5[:5000, :,:]; ## Training size ## ---------------------------



k = pkit.centKitIV(t3[:,:,3], np.hstack([np.ones(8), np.zeros(12)])); #np.array(np.ones(12));

ptime("centkit init: ", st);

#k.train(2, 50.);
#print(len(k.centroids))
#k.train(8, 15.5);
#print(len(k.centroids))
k.train(100, 0.15);
print(len(k.centroids))

ptime("k.train: ", st);

#for i in range(0,4):
#    k.train(15, 5.5);
#    print(len(k.centroids))



vcen = np.vstack([i.centre for i in k.centroids]);
g.drawkit_mat(vcen, [600,900])
#g.fileout("./testcent_")

print(vcen.shape)

xweights= np.hstack([np.ones(8), np.zeros(12)]);


k = pkit.centKitIV(t3[:,:,0], xweights); #np.array(np.ones(12));
k.train(30, 0.75);
print(len(k.centroids))

vcen = np.vstack([i.centre for i in k.centroids]);
g.drawkit_mat(vcen, [600,900])
g.fileout("./testcent_Vel")

##

k = pkit.centKitIV(t3[:,:,1], xweights); #np.array(np.ones(12));
k.train(50, 0.5);
print(len(k.centroids))

vcen = np.vstack([i.centre for i in k.centroids]);
g.drawkit_mat(vcen, [600,900])
g.fileout("./testcent_Accel")

k = pkit.centKitIV(t3[:,:,2], xweights); #np.array(np.ones(12));
k.train(50, 0.15);
print(len(k.centroids))

vcen = np.vstack([i.centre for i in k.centroids]);
g.drawkit_mat(vcen, [600,900])
g.fileout("./testcent_Rota")



k = pkit.centKitIV(t3[:,:,3], xweights); #np.array(np.ones(12));
k.train(50, 0.05);
print(len(k.centroids))

vcen = np.vstack([i.centre for i in k.centroids]);
g.drawkit_mat(vcen, [600,900])
g.fileout("./testcent_radCel")


g.drawkit_matII(t3[:150,:,0], t3[:150,:,1], [600,900])
g.fileout("./sample_veloXaccel")

g.drawkit_matII(t3[:150,:,2], t3[:150,:,3], [600,900])
g.fileout("./sample_rotaXradcel")


g.drawkit_matII(t3[:150,:,0], t3[:150,:,3], [600,900])
g.fileout("./sample_veloXradcel")

g.drawkit_matII(t3[:150,:,2], t3[:150,:,1], [600,900])
g.fileout("./sample_rotaXaccel")

ptime("end: ", st);


##""";