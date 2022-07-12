# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from timeit import default_timer as timer
from datetime import timedelta

def ptime(xstr, s):
    print(xstr, ": ", timedelta(seconds=timer()-s));
    s = timer();
    pass;

st = timer();

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import zarr
import sys

sys.path.append("../input/n3l5-a/");
sys.path.append("../input/motion-import/");
sys.path.append("../input/semdata/");
sys.path.append("../input/pysvg/");


import Nelsa
import motion_Import as motion
import semData
import pySvg

import l5kit.data.labels as l5Labels

import pymap3d

HC_Lat = 37.429333
HC_Lng = -122.154361

HC_Ex, HC_Ey, HC_Ez = pymap3d.geodetic2ecef(HC_Lat, 0, 0); ## try lngZero?


def ecefNorm(dLat, dLng):
    
    rx1, ry1, rz1 = pymap3d.geodetic2ecef(dLat+HC_Lat, 0, 0); ## NS Dist // EW Centre
    
    rx2, ry2, rz2 = pymap3d.geodetic2ecef(dLat+HC_Lat, dLng, 0); ## EW Dist @ lat
    
    return rx2-rx1, ry2-ry1, rx1-HC_Ex, rz1-HC_Ez; ##EW-Fall, East-west, NS-Fall, Northsouth

## East-West Horizontal? // NS Direction skew


def toDist(tupl):
    return [np.hypot(tupl[0], tupl[1]) *np.sign(tupl[1]), np.hypot(tupl[2], tupl[3]) *np.sign(tupl[3])]

#nelsa = Nelsa.Nelsa;


def scene_Assign(subTimes, dataTimes):
    
    ids = np.zeros(subTimes.shape[0], dtype=int) -1;
    
    idR = dataTimes.shape[1];
    
    idx = 0;
    
    idl = subTimes.shape[0];
    
    for xiter in range(0,idR):
        
        while dataTimes[0, xiter] <= subTimes[idx] <= dataTimes[1, xiter]:
        
            ids[idx] = xiter;
            
            idx += 1;
            
            if (idx == idl): break;
    
    return ids;        
            

def scene_Targets(frameids, trackids):
    
    xlist = [];
    
    for i in range(0, np.max(frameids)):
        
        mask = np.nonzero(frameids == i)[0];
        
        xlist.append(trackids[mask]);
        
    return xlist;


def target_Agents(data, sc, aglist):
    #def recordAccess(data, sc):
    rS = data["scenes"][sc];    
    rF = data["frames"][slice(rS[0][0], rS[0][1])]; ##?
    
    rA = data["agents"][slice(rF[0][1][0], rF[-1][1][1])];
    
    idR = rA["track_id"]
    
    rN = np.sum([idR==x for x in aglist], axis=0);
    
    rAA = rA[np.nonzero(rN)[0]]
        
    return rAA;

def target_AgentsII(data, sc, aglist):
    #def recordAccess(data, sc):
    rS = data["scenes"][sc];    
    #rF = [ data["frames"][rS[0][0]], data["frames"][rS[0][1]]]; ##?
    rF = data["frames"][slice(rS[0][0], rS[0][1])]; ##?
    
    rA = data["agents"][slice(rF[0][1][0], rF[-1][1][1])];
    
    idR = rA["track_id"]
    
    asLanes = [];
    
    #rN = np.sum([idR==x for x in aglist], axis=0);
    for x in aglist:
        
        asLanes.append(rA[np.nonzero(idR==x)[0]])
    
    #rAA = rA[np.nonzero(rN)[0]]
        
    return asLanes;

    
    




#for dirname, dirs, filenames in os.walk('/kaggle/input/'): ##/input/
#    print(dirname[-30:], dirs, len(filenames), " files")
    #pass;

s_sub = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv";
subH = pd.read_csv(s_sub, header=0, usecols=[0,1]);

print(subH.head());

DATA_ROOT = "../input/lyft-motion-prediction-autonomous-vehicles"
zdz = zarr.open(DATA_ROOT + "/scenes/test.zarr", mode="r")

print(zdz.info);

print(zdz.scenes.shape);

ptime(" ", st);

mapi = semData.mapII("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb", [[1,0],[0,1]])
print(type(mapi), len(mapi.elements), "elements");

mapii = semData.mapData(mapi);

ptime("map init ", st);

x1 = np.array([zdz.scenes["start_time"][:], zdz.scenes["end_time"][:]])

z1 = subH["timestamp"];

z2 = subH["track_id"];

subframes = scene_Assign(z1, x1);

scene_t = scene_Targets(subframes, z2)

agent_t = target_Agents(zdz, 0, scene_t[0]);


rA = Nelsa.Nelsa.recordAccess(zdz, 0);

tids = rA[2]["track_id"];

tmask = tids == subH["track_id"][0]

tid_data = rA[2][np.nonzero(tmask)[0]];

predind = np.nonzero(rA[1]["timestamp"] == subH["timestamp"][0]);

print(predind[0][0])

print(tid_data[predind],"\n", tid_data.dtype);


## ego data

ptime("motion collect init ", st);

st=timer();

setFrames=350;

try:
    
    runonce == true

except: 
    
    vrII, vnS = motion.collectFrames(range(0,setFrames), zdz);
    
    runonce = True;


agent_tf = []

ptime("motion collect", st);

st=timer();
    
for i in range(0, setFrames):
    
    agent_tf.append(target_AgentsII(zdz, i, scene_t[i]));
    

ptime("agent collect", st);

#matx = ;

s = pySvg.Svg();

for sid in range(0, setFrames):

    scene0a = agent_tf[sid];
    
    if len(scene0a)==0: continue;
        
    agent0_path = scene0a[0]["centroid"]

#scene0a = agent_tf[6][1];
    #agent1_path = scene0a[1]["centroid"]
    #agent2_path = scene0a[2]["centroid"]
    #agent3_path = scene0a[3]["centroid"]

    ego0_path = vrII[sid];

    #print(agent0_path.shape, ego0_path.shape);
    
    if(agent0_path.shape[0] == ego0_path.T.shape[0]):

        matx = np.vstack([agent0_path[:,0], ego0_path[0,:]])#;, agent2_path[:,0]])#, ego0_path[1,:]]);
                 
        maty = np.vstack([agent0_path[:,1], ego0_path[1,:]]);#, agent2_path[:,1]])#, ego0_path[0,:]]);
                 
        s.drawkit_matII(matx, maty, [400,600], True);
                 
        s.fileout("./scene"+str(sid));
        
        os.remove("./scene"+str(sid)+".svg");
                 

s.taglayer(2500, 1500);

print(len(s.layered))

s.layerout("./overlays");

Nelsa.Nelsa.xStore(Nelsa.Nelsa.xRaster_f("./overlays.svg"), "./overlays.png");

os.remove("./overlays.svg");

            
degree2kmNS = toDist(ecefNorm(1, 0))[1] / 1000;

degree2kmEW = toDist(ecefNorm(0, 1))[0] / 1000;
                 
def world2gps(wx, wy):
    
    gX = (wx/1000) / degree2kmEW;
    
    gY = (wy/1000) / degree2kmNS;

    return gY, gX




#array([  540.27227783, -2400.25268555,   288.60223389])

for i in range(0,250,50):
    
    print(zdz.frames[i]["ego_translation"])

    et = zdz.frames[i]["ego_translation"]

    #world2gps(et[0], et[1])
#(-0.021625118001533463, 0.006104248289230795)

    gxC = world2gps(et[0], et[1])

    print(toDist(ecefNorm(gxC[0], gxC[1])))
#[540.4345079160805, -2400.074577237883]


