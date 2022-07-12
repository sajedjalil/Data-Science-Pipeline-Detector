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

sys.path.append("../input/pysvg");
sys.path.append("../input/datafunc");
sys.path.append("../input/pycent");


import pySvg;
import pyCent;
import datafunc as motion;


def modelCollect(pathn):
    
    #np.array(  )[:,1]
    
    cent_s = np.array( pd.read_csv("../input/subii/"+pathn+"_a.csv"), dtype=int)[:,1]
    
    #cent_s = pd.read_csv("../input/subii/"+pathn+"_a.csv")
    
    #w = pd.read_csv("../input/subii/"+pathn+"_b.csv");
    
    w = np.array(  pd.read_csv("../input/subii/"+pathn+"_b.csv"))[:,1]
    
    bool_param = np.array( pd.read_csv("../input/subii/"+pathn+"_c.csv") )[:,1]
    
    cent_n = np.array( pd.read_csv("../input/subii/"+pathn+"_d.csv"), dtype=int)[:,1]
    
    #print(" [", cent_s, cent_n, "] ")
    
    e = pd.read_csv("../input/subii/"+pathn+"_e.csv");
    
    k = np.array(e)[:,1].reshape(cent_n[0], int(cent_s[0]), int(cent_s[1]));
    
    return {'weight': w, 'param': bool_param, 'centr': k}; #pass;


"""    @numba.stencil()
    def filter1d(data):
      return (data[0] - data[-10]);"""



def unfilter1d(match, ns, x=10):
    
    v = match[0, :]; ## same
    
    r = match[2, :]; ## position
    
    a = match[1, :]; ## swapped
    
    dr = match[3, :]; ## same
    
    resP = np.zeros([ns, 4], dtype=float);
    
    for i in range(x, ns):
        
        resP[i,0] = resP[i-1,0] + v[i] / 10.; ##?
        
        resP[i,1] = resP[i-1,1] + r[i] / 10.; ##?
        
        resP[i,2] = resP[i-1,2] + a[i] / 10.; ##?
        
        resP[i,3] = resP[i-1,3] + dr[i] / 10.; ##?
        
    return resP;
        
    #pass;

    
def distat2coord(step, direction, offang = 0):
    
    coord = np.zeros([100, 2], dtype=float); ##inp/out relative?
    
    for i in range(100):
        
        ux, uy = math.cos(direction[i] + offang), math.sin(direction[i] + offang); # rota
        
        #[np.cos(agent_yaw_rad), -np.sin(agent_yaw_rad), agent_centroid_m[0]],
        
        coord[i,0] = ux * step[i]/10; ##velocity 
        
        coord[i,1] = uy * step[i]/10;
        
    return coord;


kx = modelCollect("test1/model");

ks = kx["centr"];

n_Cent = ks.shape[0]; ## 50's

resp = unfilter1d(ks[12], 100, 20)

vel2 = np.cumsum(resp[:,2] / 10); ## position from accel, vel_initial=0 ##

print(np.abs(resp[:,0] - vel2));

print(np.sum(np.abs(resp[:,0] - vel2)));

s = pySvg.Svg();

for i in range(n_Cent): #(50):
    
    resp = unfilter1d(ks[i], 100, 10)
    
    orig = ks[i];
    
    vel2 = np.cumsum(resp[:,2] / 10); ## dist from accel, vel_initial=0
    
    print("dist.diff: ", np.sum(np.abs(resp[:,0] - vel2)));
    
    rota2 = np.cumsum(resp[:,3] / 10); ## direction from radcel, vel_initial=0
    
    print("r2: ", np.sum(np.abs(resp[:,1] - rota2)));
    
    rxy = distat2coord(resp[:,2], resp[:,1], (i*2*math.pi)/50) ##? spirograph?

    cxy = np.cumsum(rxy, axis=0);
    
    s.out += s.draw_vec(cxy.T, "#F004")
    
    print("end location:", cxy[-1,:]);
    
    #print("dist est1:", resp[:,0][-1]);
    
    #print("dist est2:", vel2[-1]);

    
    print("rota2?");
    
    rxy = distat2coord(resp[:,2], rota2, (i*2*math.pi)/n_Cent) ##?

    cxy = np.cumsum(rxy, axis=0);
    
    print("end location:", cxy[-1,:]);
    
    s.out +=  s.draw_vec(cxy.T, "#0F04")
    
    #print("dist est1:", resp[:,0][-1]);
    
    #print("dist est2:", vel2[-1]);

    print("orig?");
    
    rxy = distat2coord(orig[0,:], resp[:,1], (i*2*math.pi)/n_Cent); ##?

    cxy = np.cumsum(rxy, axis=0);
    
    print("end location:", cxy[-1,:]);
    
    s.out +=  s.draw_vec(cxy.T, "#00F4")
    
    s.layered += s.out;
    
    s.out = "";

    
s.taglayer(150,150);

s.layerout("./paths");



s_sub = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles/single_mode_sample_submission.csv";
subH = pd.read_csv(s_sub, header=0, usecols=[0,1]);

subHII = pd.read_csv(s_sub, header=0, nrows=0);

print(subH.head());

retSub = np.zeros([subH.shape[0], subHII.shape[1]]);


DATA_ROOT = "../input/lyft-motion-prediction-autonomous-vehicles"
zdz = zarr.open(DATA_ROOT + "/scenes/test.zarr", mode="r")

print(zdz.info);

print(zdz.scenes.info);



setFrames = 11314;#14; ##

print("using nScenes", setFrames);

x1 = np.array([zdz.scenes["start_time"][:], zdz.scenes["end_time"][:]])

z1 = subH["timestamp"];

z2 = subH["track_id"];

subframes = motion.dataFun.scene_Assign(z1, x1);

scene_t = motion.dataFun.scene_Targets(subframes, z2)

agent_t = motion.dataFun.target_AgentsII(zdz, 0, scene_t[0]);

agent_tf = [];

for i in range(0,setFrames):

    agent_tf.append(motion.dataFun.target_AgentsII(zdz, i, scene_t[i]));
    
    
#track1 = agent_tf[0][0];


    
    #return r2;?
    
def circleDiff(ang1, ang2): ## Always returns -pi < x < pi?
    if np.abs(ang1 - ang2) <= math.pi:
        return ang1 - ang2;
    if (ang1 - ang2) < -math.pi:        
        return (ang1 - ang2) + math.pi*2;    
    if (ang1 - ang2) > math.pi:        
        return (ang1 - ang2) - math.pi*2;
    

vcircleDiff = np.vectorize(circleDiff)


def feat_ConstructionA(data):
    velx = motion.dataFun.ego.filter1d2(data[:,0,:]);
    vely = motion.dataFun.ego.filter1d2(data[:,1,:]);
    vels = np.hypot(velx, vely) ##(data[:,0,:], data[:,1,:]);
    print(vels.shape);
    accel = [];
    #for i in range(data.shape[0]):
    accel = motion.dataFun.ego.filter1d2(vels);
    yaws = data[:,2,:];
    #yaws2 = feat_altYaw(yaws);    
    rota = vcircleDiff(motion.dataFun.ego.filter1d2(yaws), 0);    
    #rota2 = motion.dataFun.ego.filter1d2(yaws2);    
    #rota = np.minimum(rota,rota2);
    print("sub1.125 degrees/sec", np.sum(np.abs(rota) < 0.02));
    print("sub4.5 rotas/sec", np.sum(np.abs(rota) < 0.08));
    print("sub45 rotas/sec", np.sum(np.abs(rota) < 0.8));
    print("sub90 rotas", np.sum(np.abs(rota) < 1.6));
    print("+90 rotas", np.sum(np.abs(rota) >= 1.6));
    print("big rotas", np.sum(np.abs(rota) > 3.1));
    print("+270 rotas", np.sum(np.abs(rota) > 4.7));    
    radcel = motion.dataFun.ego.filter1d2(rota);    
    return [vels, accel, rota, radcel];



class xkit(pyCent.centKitIV):
    
    def matchcollect(self, t):
        
        dst = np.zeros([len(self.centroids)], dtype=float);
        
        for i in range(len(self.centroids)):
            
            dst[i] = self.centroids[i].dist_FloatVI(t, self.weight);
            
            
        #fval2 = self.weighted_err(data, self.centre, weight, 2.0); 
    
        return np.argmin(dst);
    

    def floatcollect(self, i, t):
        
        return self.centroids[i].weighted_err(t, self.centroids[i].centre, self.weight, 2.0); 
        
        
    

    
mlist = [];

targ_Len = 100;

for  j in range(setFrames):
    
    xitem = agent_tf[j];
    for i in range(len(xitem)):
        
        mlist.append(len(xitem[i]) == targ_Len);
        
        #if len(xitem[i]) == targ_Len:
            
#acti = np.sum(np.array(mlist));

vaII = np.zeros([len(mlist), 3, targ_Len], dtype=float);

ca = 0;


#retSub = np.zeros([subH.shape[0], len(mlist)]);

retSub = np.zeros([len(mlist), subHII.shape[1]]);

print("rS:", retSub.shape);

    
for  j in range(setFrames):
    
    xitem = agent_tf[j];
    
    for i in range(len(xitem)): #inds[j]):
        
        #print(agent_tf[j][i].shape);

        vlen = agent_tf[j][i].shape[0];
        
        ###print("ts:", xitem[j]["timestamp"][-1] == z1[ca], xitem[j]["timestamp"][-1], z1[ca])
        
        #mlist.append(len(xitem[i]) == targ_Len);
        
        #if len(xitem[i]) == targ_Len:

        vaII[ca,0:2,-vlen:] = agent_tf[j][i]["centroid"].T#
            #vaII[ca,1,:] = #
        vaII[ca,2,-vlen:] = agent_tf[j][i]["yaw"].T#
            
        ca += 1;
    
#track1 = agent_tf[0][0];

#trackx = agent_tf;

def feat_altYaw(rad):
    r2 = (2*math.pi + rad) % (2*math.pi) - math.pi;
    return r2;


def csvRow(timest, trackid, preds, stat=20): ##stat?
    
    r = [];
    
    r.append(timest);
    
    r.append(trackid);
    
    r.append(1);
    
    r.append(0); ##conf_
    
    r.append(0);
    
        
    for i in range(stat, stat+50): ##coord_x00
        
        r.append(preds[i, 0]);
        
        r.append(preds[i, 1]);
    
    return r;

    
def csvOut(fn, head, data):
    
    xfmt = "%i,"*2 + "%.5f,"*302 + "%.5f"
    
    np.savetxt(fn, data, xfmt, delimiter=',', newline='\n', header=head);
 
    
print(vaII.shape);

vaIII = np.array(feat_ConstructionA(vaII));

print(vaIII.shape);

print(ks.shape);

track0 = vaIII[:,0,-20:]

k_pred = xkit(np.array([]), np.ones(20));

for i in range(ks.shape[0]):
    
    k_pred.centroids = np.append(k_pred.centroids, pyCent.liveCentIV(ks[i,:, 20:40]));


matchres = np.zeros([vaIII.shape[1]], dtype=int) -1;

floatres = np.zeros([vaIII.shape[1], 4], dtype=float);

cxyP = np.zeros([vaII.shape[0], 50, 2], dtype=float); ##50-sequence

print("coordP:", cxyP.shape)

print(vaIII.shape[1]);
    
for i in range(vaIII.shape[1]): ##50?
    
    #print(z1[i], z2[i]);
    
    track0 = vaIII[:,i,-20:]
    
    matchres[i] = k_pred.matchcollect(track0);
    
    floatres[i,:] = k_pred.floatcollect(matchres[i], track0);
    
    zn = vaII[i,0:2,-1];
    
    k_float = ks[ matchres[i] ] + np.broadcast_to(floatres[i,:], ks[0].T.shape).T;
    
    resp = unfilter1d(ks[ matchres[i] ], 100, 10);
    
    #resp = unfilter1d(k_float, 100, 10);
    
    zr = vaII[i, 2, -1];
    
    
    if i%2500==0:
        
        s.out = s.head_auto(50,50);
    
        s.out +=  s.draw_vec(vaII[i, 0:2, :] - np.broadcast_to(zn, (100,2)).T, "#00FF")
    
    ##resp
    
    
        print("iter: ",i, "yaw:",  zr, "float:", floatres[i,:]);
    
    rxy = distat2coord(resp[:,2], resp[:,1], zr+0);#math.pi); #, (i*2*math.pi)/50) ##? spirograph?

    cxy = np.cumsum(rxy, axis=0);
    
    if i%2500==0:        
    
        s.out += s.draw_vec(cxy.T, "#F00F")
    
        s.out += s.tail();
    
        s.fileout("./scene_"+str(i));
        
    retSub[i, 0:105] = np.array(csvRow(z1[i], z2[i], cxy));
    
    cxyP[i,:,:] = cxy[45:95,:] - cxy[44,:];##stat 45-Norm  #cxy[15:65,:]; ##stat 15
                                            ## +R
        
    #cxyP[i,:,:] = np.cumsum(rxy[45:95, :], axis=0); ##stat 45-NormII
    
print(np.sum(matchres == -1));

"""s.layered += s.out;


s.taglayer(250,250);

s.layerout("./pathsII");
"""

#s.out +=  s.draw_vec(cxy.T, "#0F04")

hIII = [x for x in subHII.columns];

hIV = ",".join(hIII)

csvOut("./submissionX.csv", hIV, retSub[:,:]);


if setFrames == 11314:

    l5eval.write_pred_csv("./submission.csv", z1, z2, cxyP);
    

for i in range(np.min(matchres), np.max(matchres)):
    
    print("Class count", np.sum(matchres == i), "for id", i);


# z1 = subH["timestamp"];

# z2 = subH["track_id"];
                     
#f = open("./submission.csv", "r+")

#fstr = f.read();

#f.seek(0);

#f.write(fstr[2:]+'\0');

#f.close();

##def testangle(i):

#    return math.cos(i), math.sin(i); 

"""
    Encode the ground truth into a csv file. Coords can have an additional axis for multi-mode.
    We handle up to MAX_MODES modes.
    For the uni-modal case (i.e. all predictions have just a single mode), coords should not have the additional axis
    and confs should be set to None. In this case, a single mode with confidence 1 will be written.
    Args:
        csv_path (str): path to the csv to write
        timestamps (np.ndarray): (num_example,) frame timestamps
        track_ids (np.ndarray): (num_example,) agent ids
        coords (np.ndarray): (num_example x (modes) x future_len x num_coords) meters displacements
        confs (Optional[np.ndarray]): (num_example x modes) confidence of each modes in each example.
        Rows should sum to 1"""



    
    