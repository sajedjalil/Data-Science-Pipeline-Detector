# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, sys, zarr
"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""

import l5kit.evaluation as l5eval;

#sys.path.append("../input/pysvg");
sys.path.append("../input/datafunc");
#sys.path.append("../input/pycent");


#import pySvg;
#import pyCent;
import datafunc as motion;



DATA_ROOT = "../input/lyft-motion-prediction-autonomous-vehicles"
zdz = zarr.open(DATA_ROOT + "/scenes/test.zarr", mode="r")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


## interpolate /+ive
## w/ three point stencil

## extrap conf = linearity?

def linear(x1, x2, p):
  return x1 * (1.0 - p) + p * x2;

def quadratic(x1,x2,x3, p):
  return linear(linear(x1,x2, p), linear(x2,x3,p), p);

def aquad(x1,x2,x3, p):
  return quadratic(x1, 2*(x2 - linear(x1,x3, 0.5)) + linear(x1,x3, 0.5), x3, p);

def linearity(x1,x2,x3):
  dists = np.array([np.hypot(x1[0]-x2[0], x1[1]-x2[1]), np.hypot(x2[0]-x3[0], x2[1]-x3[1])]);
  return np.min(dists) / np.max(dists);


setFrames = 513#14;#14; ##
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

vaII = np.zeros([cmax, 3, targ_Len]);

ca=0;

for  j in range(setFrames):
    xitem = agent_tf[j];
    for i in range(len(xitem)):
        if len(xitem[i]) == targ_Len:            
            vaII[ca,0:2,:] = agent_tf[j][i]["centroid"].T;
            vaII[ca,2,:] = agent_tf[j][i]["yaw"].T#            
            ca += 1;

            
print(ca);
rx = 3400;         

st = 0;

sl = 0;
            
for xi in range(rx):
            
  samp = xi#250;

  vsT = vaII[samp, :, 20:61]

  vs = vaII[samp, :, [20,40,60]]

  r = np.array([aquad(vs[0], vs[1], vs[2], i) for i in np.arange(0,1.025,0.025)])

  xin = np.zeros([2, 41], dtype=float);

  xout = np.zeros([3, 2, 41], dtype=float);

  xin[:,:] = vsT[:2, :];

  xout[0,:,:] = r[:,:2].T;


  sc = l5eval.metrics.neg_multi_log_likelihood(xin.T, xout[:,:2,:].transpose(0,2,1), np.array([1.,0,0]), np.ones(41));

  st += sc;

  #print(sc);

  sl += linearity(vs[0,:2], vs[1,:2], vs[2,:2]);

  if(sc>5): 
    print(xi, sc, "lin:", linearity(vs[0,:2], vs[1,:2], vs[2,:2]))

print("mean nll:", st / rx);

print("meanlin:", sl/rx);
    
    
    