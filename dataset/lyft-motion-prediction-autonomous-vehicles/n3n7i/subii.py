# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, sys, zarr
import math

def ptime(x,y):
    pass;

st=0;

"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""


sys.path.append("../input/n3l5-a/");
sys.path.append("../input/datafunc/");
sys.path.append("../input/semdata/");
sys.path.append("../input/pysvg/");
sys.path.append("../input/pycent/");


import Nelsa
import datafunc as motion
import semData
import pySvg
import pyCent


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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



for x in mapii.lanes:
    
    x.Delta[0]["px"] += x.delta_Offset[0];
    x.Delta[0]["py"] += x.delta_Offset[1];
    
    x.Delta[1]["px"] += x.delta_Offset[0];
    x.Delta[1]["py"] += x.delta_Offset[1];
    
    
s=pySvg.Svg();

s.semt(0, 0)
#s.drawkit_semTraffLanes(mapii, range(0, len(mapii.lanes),3), 2000, 2000);#, hX=True); #len(mapii.lanes)
#s.fileout("./l100_mappedII");

#os.remove("./l100_mappedII.svg");    

#s.layered += s.out.split("\n")[1];




setFrames = 2500; #/11314


print("using scenes:", setFrames);

cT = motion.dataFun.csvTrainer(zdz, range(0,setFrames), motion.l5ind("car"), [0,10,30]);

print(cT[:5]);


x1 = np.array([zdz.scenes["start_time"][:], zdz.scenes["end_time"][:]])

z1 = subH["timestamp"];

z2 = subH["track_id"];

subframes = motion.dataFun.scene_Assign(z1, x1);

scene_t = motion.dataFun.scene_Targets(subframes, z2)

agent_t = motion.dataFun.target_AgentsII(zdz, 0, scene_t[0]);

agent_tf = [];

for i in range(0,setFrames):

    agent_tf.append(motion.dataFun.target_AgentsII(zdz, i, scene_t[i]));
    

    
#setFrames=50;    

try:
    
    runonce == true

except: 
    
    vrII, vnS = motion.dataFun.collectFrames(range(0,setFrames), zdz);
    
    runonce = True;

    
#tx = ;

##s = pySvg.Svg();

for sid in range(0, 1): #setFrames):

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
                 
        s.drawkit_matII(matx, maty, [400,600], False);
                 
        #s.fileout("./scene"+str(sid));
        
        s.layered += s.out.split("\n")[1];
        
        #os.remove("./scene"+str(sid)+".svg");    
    

#s.taglayer(2500,1500);
s.taglayer(400,600);

#s.layerout("./multidata")
    
#Nelsa.Nelsa.xStore(Nelsa.Nelsa.xRaster_f("./multidata.svg"), "./multidata.png");

#os.remove("./multidata.svg");



targ_Len = 100

inds = [len(x) for x in agent_tf];

mlist = [];



for  j in range(setFrames):
    
    xitem = agent_tf[j];
    for i in range(inds[j]):
        
        mlist.append(len(xitem[i]) == targ_Len);
        
        #if len(xitem[i]) == targ_Len:
            
acti = np.sum(np.array(mlist));

vaII = np.zeros([acti, 3, targ_Len], dtype=float);

ca = 0;


for  j in range(setFrames):
    
    xitem = agent_tf[j];
    for i in range(inds[j]):
        
        #mlist.append(len(xitem[i]) == targ_Len);
        
        if len(xitem[i]) == targ_Len:

            vaII[ca,0:2,:] = agent_tf[j][i]["centroid"].T#
            #vaII[ca,1,:] = #
            vaII[ca,2,:] = agent_tf[j][i]["yaw"].T#
            
            ca += 1;


vxII = np.vstack([vrII, vaII]);


#def crossZero(rad)

    #r2 = (2*math.pi) - (rad + math.pi)
    #==r2 = math.pi - (rad + math.pi)

def feat_altYaw(rad):
    
    r2 = (2*math.pi + rad) % (2*math.pi) - math.pi;
    
    return r2;


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
            
            

vxIII = np.stack(feat_ConstructionA(vxII), axis=1);


#"""
class remoteDebug(pyCent.liveCentIV):    
    
    def weighted_err(self, data, targ, actives, eX=1):
        
        dt = (data - targ) * actives;        
        
        #print("dt: ", dt.shape)            
        
        e1 = np.max(np.abs(dt), axis=1) + 1; #e-4; ##axis        
        
        #print(e1);
        
        if(eX == 1): 
            de = dt * (np.abs(dt) / np.broadcast_to(e1, dt.T.shape).T); ##(dt/e1) ^1?            
            #print("e1_broad: ", np.broadcast_to(e1, dt.T.shape).T.shape)            
        if(eX != 1): 
            de = dt * (np.abs(dt) / np.broadcast_to(e1, dt.T.shape).T)**eX ;
            
        ds = np.sum(de, axis=1) / np.sum(actives); ##mean weighted error? ##axis        
        return ds;
    
    
    
""" def dist_FloatIII(self, data, weight): ## least linears // nD floating
        
        ##print(data.shape, weight.shape);
                
        fval = np.sum((data - self.centre) * weight, axis=1) / np.sum(weight); ##?
        
        ##print(fval.shape, fval);
        
        vout = np.sum((((data - np.broadcast_to(fval, data.T.shape).T) - self.centre) * weight )**2)
        
        voutII = np.sum(((data - self.centre) * weight )**2);
        
        print("impB:", vout <= voutII);
        
        return np.sum(np.abs(((data - np.broadcast_to(fval, data.T.shape).T) - self.centre) * weight ))
    
    
    def dist_FloatIV(self, data, weight): ## virtual least squares? // nD floating        
        ##print(data.shape, weight.shape);        
        sval = (data - self.centre) * weight;        
        sval2 = sval * np.abs(sval); ## signed square                
        fval = np.sum(sval2, axis=1) / np.sum(weight); ##?
        
        
        fval2 = np.sqrt(fval) * np.sign(fval); ## signed root mean signed square
        
        vout = np.sum(np.abs(((data - np.broadcast_to(fval2, data.T.shape).T) - self.centre) * weight ));
        
        voutII = np.sum(np.abs((data - self.centre) * weight ));
        
        #print("impA:", vout <= voutII);
        
        vout = np.sum((((data - np.broadcast_to(fval2, data.T.shape).T) - self.centre) * weight )**2);
        
        voutII = np.sum(((data - self.centre) * weight )**2);
        
        #print("impB:", vout <= voutII);
        
        ##print(fval.shape, fval);        
        return np.sum(np.abs(((data - np.broadcast_to(fval2, data.T.shape).T) - self.centre) * weight )) """
    
#"""
#pyCent.liveCentIV = remoteDebug;
            

k = pyCent.centKitIV(vxIII[:,:,:], np.hstack([np.zeros(20), np.ones(30), np.zeros(20), np.ones(5), np.zeros(20), np.ones(5)]));#np.zeros(50)])); #np.array(np.ones(12));

#k.least_Linear = True; ### Least linears ;)

#k.least_Squares = True; ### Least signed squares?

#k.least_Weighted = True; ### another squares attempt

k.least_WeightII = True; ### probably cubic?

ptime("centkit init: ", st);

#k.train(2, 50.);
#print(len(k.centroids))
#k.train(8, 15.5);
#print(len(k.centroids))
k.train(125, 60.);  ##k.train(50, 15.);
print(len(k.centroids))

ptime("k.train: ", st);
#"""

def collectMat(cent, r=[0,1]):
    
    mA = np.array([x.centre[r[0],0:] for x in cent.centroids]);
    
    mB = np.array([x.centre[r[1],0:] for x in cent.centroids]);

    return mA, mB;


k2 = collectMat(k);

#svgtest = '<set attributeName="width" to="400" end="me.click" begin="me.dblclick" /><set attributeName="height" to="400" end="me.click" begin="me.dblclick" />'


s = pySvg.Svg();

s.drawkit_matII(k2[0], k2[1])

#s.out = s.out.split('\n')[0] + svgtest + s.out.split('\n')[1] + s.out.split('\n')[2] 

s.fileout("./cent01")

s.drawkit_mat(k2[0])

s.fileout("./cent0")

s.drawkit_mat(k2[1])

s.fileout("./cent1")

k2 = collectMat(k, [2,3]);

s.drawkit_matII(k2[0], k2[1])

s.fileout("./cent23")

s.drawkit_mat(k2[0])

s.fileout("./cent2")

s.drawkit_mat(k2[1])

s.fileout("./cent3")

k2 = collectMat(k, [0,3]);

s.drawkit_matII(k2[0], k2[1])

s.fileout("./cent03")

k2 = collectMat(k, [1,2]);

s.drawkit_matII(k2[0], k2[1])

s.fileout("./cent12")

k2 = collectMat(k, [1,3]);

s.drawkit_matII(k2[0], k2[1])

s.fileout("./cent13")

k2 = collectMat(k, [0,2]);

s.drawkit_matII(k2[0], k2[1])

s.fileout("./cent02")


def Modelstore(k):
    
    a = k.dshape;
    
    b = k.weight;
    
    c = [k.least_Linear, k.least_Squares, k.least_Weighted, k.least_WeightII];
    
    d = [len(k.centroids)];
    
    e = [x.centre for x in k.centroids];
    
    return [a,b,c,d,e];


def toPandas(x, modName, xdir =""):
    
    os.mkdir("/kaggle/working/"+xdir)
    
    pd.DataFrame(x[0]).to_csv("./"+xdir+modName+"_a.csv");
    
    pd.DataFrame(x[1]).to_csv("./"+xdir+modName+"_b.csv");
    
    pd.DataFrame(x[2]).to_csv("./"+xdir+modName+"_c.csv");
    
    pd.DataFrame(x[3]).to_csv("./"+xdir+modName+"_d.csv");
    
    pd.DataFrame(np.array(x[4]).flatten()).to_csv("./"+xdir+modName+"_e.csv");
    
    #return [a,b,c,d,e];


toPandas(Modelstore(k), "model", "test1/")


    