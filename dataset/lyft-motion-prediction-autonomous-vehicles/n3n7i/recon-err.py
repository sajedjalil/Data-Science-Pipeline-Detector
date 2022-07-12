# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os, sys, math

import zarr

import l5kit

print(l5kit.__version__);

import l5kit.evaluation as l5eval;


"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#sys.path.append("../input/pysvg");
sys.path.append("../input/datafunc");
#sys.path.append("../input/pycent");

#import pySvg;
#import pyCent;
import datafunc as motion;

import numba;

class remoteDebug(motion.data.ego):
    
    @numba.stencil(neighborhood = ((0, 0), (-5,5)))
    def filter1d2(data):
      return data[0,5] - data[0,-5];

motion.data.ego = remoteDebug;
    


def yawfromvel(vels):
    return np.arctan2(vels[1], -vels[0]);


def unitvelfromyaw(yaw):
    return -np.cos(yaw), np.sin(yaw); ## unscaled v[0], v[1]


DATA_ROOT = "../input/lyft-motion-prediction-autonomous-vehicles"
zdz = zarr.open(DATA_ROOT + "/scenes/test.zarr", mode="r")

print(zdz.info);

print(zdz.scenes.info);



def feat_ConstructionA(data):
    velx = motion.dataFun.ego.filter1d2(data[:,0,:]);
    vely = motion.dataFun.ego.filter1d2(data[:,1,:]);
    vels = np.hypot(velx, vely) ##(data[:,0,:], data[:,1,:]);
    print(vels.shape);
    accel = [];
    #for i in range(data.shape[0]):
    accel = motion.dataFun.ego.filter1d2(vels);
    yaws = data[:,2,:];
    yaws2 = feat_altYaw(yaws);    
    rota = motion.dataFun.ego.filter1d2(yaws);    
    rota2 = motion.dataFun.ego.filter1d2(yaws2);    
    rota = np.minimum(rota,rota2);
    print("sub1.125 degrees/sec", np.sum(np.abs(rota) < 0.02));
    print("sub4.5 rotas/sec", np.sum(np.abs(rota) < 0.08));
    print("sub45 rotas/sec", np.sum(np.abs(rota) < 0.8));
    print("sub90 rotas", np.sum(np.abs(rota) < 1.6));
    print("+90 rotas", np.sum(np.abs(rota) >= 1.6));
    print("big rotas", np.sum(np.abs(rota) > 3.1));
    print("+270 rotas", np.sum(np.abs(rota) > 4.7));    
    radcel = motion.dataFun.ego.filter1d2(rota);    
    return [vels, accel, rota, radcel];


def feat_altYaw(rad):
    r2 = (2*math.pi + rad) % (2*math.pi) - math.pi;
    return r2;


def unfilter1d(match, ns, x=10, lagfilt=0, xmod = 10.):
    
    v = match[0, :]; ## same
    
    r = match[2, :]; ## position
    
    a = match[1, :]; ## swapped
    
    dr = match[3, :]; ## same
    
    resP = np.zeros([ns, 4], dtype=float);
    
    for i in range(x, ns):
        
        resP[i,0] = resP[i-1,0] + v[i-lagfilt] / xmod; ##?
        
        resP[i,1] = resP[i-1,1] + r[i-lagfilt] / xmod;#10.; ##?
        
        resP[i,2] = resP[i-1,2] + a[i-lagfilt] / xmod;#10.; ##?
        
        resP[i,3] = resP[i-1,3] + dr[i-lagfilt] / xmod;#10.; ##?
        
    return resP;
        
    #pass;

    
def distat2coord(step, direction, offang = 0, stepvar=10):
    
    coord = np.zeros([100, 2], dtype=float); ##inp/out relative?
    
    for i in range(100):
        
        ux, uy = math.cos(direction[i] + offang), math.sin(direction[i] + offang); # rota ???
        
        #[np.cos(agent_yaw_rad), -np.sin(agent_yaw_rad), agent_centroid_m[0]],
        
        coord[i,0] = ux * step[i]/ stepvar;#10; ##velocity 
        
        coord[i,1] = uy * step[i]/ stepvar;#10;
        
    return coord;


def feat_ConstructV(data):
    velx = motion.dataFun.ego.filter1d2(data[:,0,:]);
    vely = motion.dataFun.ego.filter1d2(data[:,1,:]);

    return np.array([velx, vely]);


def feat_ConstructY(data):    
    yaws = data[:,2,:];
    #yaws2 = feat_altYaw(yaws);    
    return yaws;
    


setFrames = 3200#14;#14; ##

targ_Len = 100;



subJ = motion.dataFun.csvTrainer(zdz, range(setFrames), motion.l5ind('car'));

subK = np.array(subJ, dtype=int);

print(subK.shape)

subH = {"timestamp": subK[:,0], "track_id": subK[:,1]};



print("using nScenes", setFrames, " target length", targ_Len);

x1 = np.array([zdz.scenes["start_time"][:], zdz.scenes["end_time"][:]])

z1 = subH["timestamp"];

z2 = subH["track_id"];

subframes = motion.dataFun.scene_Assign(z1, x1);

scene_t = motion.dataFun.scene_Targets(subframes, z2)

agent_t = motion.dataFun.target_AgentsII(zdz, 0, scene_t[0]);

agent_tf = [];

cmax =0;

for i in range(0,setFrames):

    agent_tf.append(motion.dataFun.target_AgentsII(zdz, i, scene_t[i]));
    
    cmax += len(agent_tf[-1]);


    
mlist = [];

vaII = np.zeros([cmax, 3, targ_Len]);

velcheck = np.zeros([cmax, 2, targ_Len]);

ca=0;

for  j in range(setFrames):
    
    xitem = agent_tf[j];
    for i in range(len(xitem)):
        
        if len(xitem[i]) == targ_Len:
            #mlist.append(xitem[i]);
            
            vaII[ca,0:2,:] = agent_tf[j][i]["centroid"].T#
            #vaII[ca,1,:] = #
            vaII[ca,2,:] = agent_tf[j][i]["yaw"].T#
            
            velcheck[ca,:,:] = agent_tf[j][i]["velocity"].T#
            
            ca += 1;

            
vaIII = np.array(feat_ConstructionA(vaII[:ca,:,:]));

vaX = feat_ConstructV(vaII[:ca,:,:]);

vaDir = feat_ConstructY(vaII[:ca,:,:]);

vaDirII = np.arctan2(vaX[0], -vaX[1]); ##?

#def yawfromvel(vels):
#    return np.arctan2(vels[1], -vels[0]);




print(vaIII.shape)

tsum = 0;

samp = ca; #14000;

filter_lag = 0;

predict_at = 10 + filter_lag; # ?velocity/direction encoding?

predict_atB = 40; #45@ 247; #25@ 234 ##15 @400

predict_to = predict_atB + 50;


modsum = 0.;

direrr= 0;

filter_wid = 10;


emat = np.zeros([samp, 100], dtype=float);

for i in range(samp):
    
    resp = unfilter1d(vaIII[:,i,:], 100, predict_at+1, filter_lag, filter_wid);
    
    #repath = distat2coord
    
    zr = vaII[i, 2, predict_at];
    
    #resp[:20, 1] = 0;
    
    #rxy = distat2coord(resp[:,2], resp[:,1], zr);#math.pi); #, (i*2*math.pi)/50) ##? spirograph?
    
    #rxy = distat2coord(resp[:,2], vaIII[3,i,:], zr+0); ##354(+100)
    
    #rxy = distat2coord(vaIII[0,i,:], vaIII[3,i,:], zr+0); ##241(+0)
    
    rxy = distat2coord(vaIII[0,i,:], resp[:,1], zr+0, filter_wid); ## 98.5
    
    #rxy = distat2coord(vaIII[0,i,:], vaIII[2,i,:], zr+0); ## 165
    
    if(np.mean(np.abs(10*rxy[20:40,:] - velcheck[i,:,20:40].T))>1):
    
      print(i, "step sim: ", np.mean(np.abs(rxy[20:40, :] - vaX[:, i, 20:40].T), axis=0));
    
      print(i, "direct sim: ", np.mean(np.abs(vaDir[i,20:40] - vaDirII[i,20:40]), axis=0));
    
      print(i, "vel sim: ", np.mean(np.abs(10*rxy[20:40,:] - velcheck[i,:,20:40].T), axis=0));
    
      print("vel at x:30", velcheck[i,:,30], rxy[30,:]);
    
    direrr += np.mean(np.abs(vaDir[i,20:40] - vaDirII[i,20:40]), axis=0) > (math.pi/2)
    
    
    #rxy[:20, :] = 0;

    cxy = np.cumsum(rxy, axis=0);
    
    dxy = vaII[i,:2,:];
    
    exy = np.zeros([3,100,2], dtype=float);
    
    exy[0,:,:] = cxy;
    
    #print(cxy.shape, dxy.shape)
    
    #print("tot err:", np.sum(np.abs( dxy.T - dxy.T[predict_at,:] - cxy)));
    
    #print("last loc:", np.sum(np.abs( (dxy.T[-1,:] - dxy.T[predict_at,:]) - cxy[-1,:]))); #)); ##
    
    tsum += np.sum(np.abs( dxy.T - dxy.T[0,:] - cxy));
    
    emat[i,:] = np.sum(np.abs((dxy.T - dxy.T[predict_atB,:]) - (cxy - cxy[predict_atB,:])), axis=1);
    
    p2 = cxy - cxy[predict_atB,:];
    
    g2 = dxy.T - dxy.T[predict_atB,:];
    
    #print(l5eval.metrics.neg_multi_log_likelihood(dxy.T[predict_atB:30,:], exy[:,predict_atB:30,:], np.array([1.,0,0]), np.ones(10)));
    
    #print("full:", l5eval.metrics.neg_multi_log_likelihood(dxy.T[:,:], exy[:,:,:], np.array([1.,0,0]), np.ones(100)));
    
    exy[0,:,:] = p2;
    
    #print("mod:", l5eval.metrics.neg_multi_log_likelihood(g2[predict_atB:30,:], exy[:,predict_atB:30,:], np.array([1.,0,0]), np.ones(10)));
    
    modsum += l5eval.metrics.neg_multi_log_likelihood(g2[predict_atB:predict_to,:], exy[:,predict_atB:predict_to,:], np.array([1.,0,0]), np.ones(50));
    
    pass;
            

#print(tsum / samp);

evec = np.mean(emat, axis=0);

print(np.mean(emat, axis=0));

print(direrr, "+pi/2 direction errors?")
#print((dxy.T - dxy.T[predict_atB,:]) - (cxy - cxy[predict_atB,:]));


print(modsum / samp, " mean nll recon50, samples:", samp);


