
from timeit import default_timer as timer
from datetime import timedelta

def ptime(xstr, s):
    print(xstr, ": ", timedelta(seconds=timer()-s));

start = timer()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys

#import l5kit, l5kit.configs, l5kit.data
import zarr

from pathlib import Path

sys.path.append("../input/pycent/");
import pyCent as pkit

sys.path.append("../input/pysvg/");
import pySvg as gkit

sys.path.append("../input/n3l5-a/");

import Nelsa

sys.path.append("../input/semdata/");

import semData

import numba

ptime("import loading: ", start);

#end = timer()
#print(timedelta(seconds=end-start))    
    
"""
def recordAccess(data, sc):
    rS = data["scenes"][sc];    
    rF = data["frames"][slice(rS[0][0], rS[0][1])];    
    rA = data["agents"][slice(rF[0][1][0], rF[-1][1][1])];
    return [rS, rF, rA];

def pointAccess(rA, col=0):    
    return [x[col] for x in rA];

def pointVec(pA):
    return [ [x[0] for x in pA], [x[1] for x in pA]];

"""
def xf(n):
    return(n[0] != '_');

def xdir(n):
    n2 = [];
    for x in filter(xf, n): n2.append(x);
    return n2;

def xhelp(n):
    [print(x) for x in xdir(dir(n))];
    pass;

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]

#!pip list --format columns

#help("l5kit.data")
#print("\n---\n");
#help("l5kit.data.zarr_utils")
#print("\n---\n");
#help("l5kit.data.zarr_dataset")

# set env variable for data
#os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
# get config
#cfg = l5kit.configs.load_config_data("../input/lyft-config-files/visualisation_config.yaml")

#dm = l5kit.data.LocalDataManager()

#dataset_path = dm.require(cfg["val_data_loader"]["key"])
#zarr_dataset = l5kit.data.ChunkedDataset(dataset_path)
#zdx = zarr_dataset.open("r", True, 255*(1024**2));
#print(zarr_dataset)
#print(type(zarr_dataset))
#zd = zarr_dataset;

start = timer()    

try:
    run_once == True;
    
except:
    
    run_once=True;
    print("----------Ran once------------");
    


DATA_ROOT = Path("../input/lyft-motion-prediction-autonomous-vehicles")
zdz = zarr.open(DATA_ROOT.joinpath("scenes/train.zarr").as_posix(), mode="r")

#mapi = l5kit.data.MapAPI("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb", [[1,0],[0,1]])

#mapi = Nelsa.mapI("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb", [[1,0],[0,1]])

print(timedelta(seconds=timer()- start), " zdz loading ");

mapi = semData.mapII("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb", [[1,0],[0,1]])

mapii = semData.mapData(mapi);



##    message Element {
"""   oneof element {
            RoadNetworkSegment segment = 1;
            RoadNetworkNode node = 2;
            Lane lane = 3;
            TrafficControlElement traffic_control_element = 4;
            Junction junction = 5;
            SegmentSequence segment_sequence = 6;
            AnnotatedShape annotated_shape = 8;
        }
    }"""

##

print(timedelta(seconds=timer()- start), " mapi ii loading ");

print(zdz.info);

print(zdz.scenes.info);

print(zdz.agents.info);

def drawscene(zz, ix):

    start = timer()

    r = recordAccess(zz, ix);
    
    f1 = Nelsa.Nelsa.movementFilter(Nelsa.Nelsa, r[2]);


    p = pointAccess(r[2][f1])
    
    v = pointVec(p);
    
    print("points: ", len(v[0]));

    vm0 = min(v[0]);
    vm1 = min(v[1]);

    for i in range(0, len(v[0])):
        v[0][i] -= vm0;
        v[1][i] -= vm1;        

    end = timer()
    print(timedelta(seconds=end-start), "part1")        
        
    c = pkit.centKit(v);

    c.stepPass(range(0,len(v[0])), 10.0, 500);
    
    start = timer()
    
    print(timedelta(seconds=start-end), "part2")        

    s = gkit.Svg();

    s.drawkit(c, max(v[0]), max(v[1]), True);
    
    end = timer()
    print(timedelta(seconds=end-start), "part3")        
    

    s.fileout("./scene_"+str(ix));
    




##
"""

s = timer();

r = recordAccess(zdz, 0);

ptime("rA: ", s);


f1 = Nelsa.Nelsa.movementFilter(Nelsa.Nelsa, r[2]);

p = pointAccess(r[2][f1]) ##def

p2 = pointAccess(r[2][f1], 3); ##velo's

v = pointVec(p);
v2 = pointVec(p2);

vm0 = min(v[0]);
vm1 = min(v[1]);

start = timer()    

#@numba.jit
for i in range(0, len(v[0])):
    v[0][i] -= vm0;
    v[1][i] -= vm1;        

print(timedelta(seconds=timer()-start), ": norm")        

c = pkit.centKitII(v, v2); ##

end = timer()    

step1 = False;

oldi = 0;

bat = 512;

for iter in range(bat, len(v[0]), bat):
    
    c.stepPassIV(range(oldi, iter), 10., 250);#len(v[0]) ?
    oldi = iter;
    
c.stepPassIV(range(oldi, len(v[0])), 10., 250);

start = timer()    

print(timedelta(seconds=start-end), "part-s")        


s = gkit.Svg();

s.drawkit(c, max(v[1]), max(v[0]));

s.fileout("./c100");

print(len(v[0]))
##"""

#for n in range(0, 1):

 #   drawscene(zdz, n);

start = timer()    
    
lm = [mapi.lane_Mask(i) for i in range(0,len(mapi))]
ptime("mapi lanemask", start)

s = gkit.Svg();

s.drawkit_sem(mapi, np.nonzero(lm)[0][slice(0,800)], 2000, 1500, hA=True);
print(len(s.out) / 1024 , " kb svg");

#s.fileout("sem_t")

ptime("drawkit strings", start)

ximg = Nelsa.Nelsa.xRaster(s.out)

ptime("svg -> pixels", start)

Nelsa.Nelsa.xStore(ximg, "./sem_test.png")
    
ptime("semantic times", start)

    
##

start = timer()    

lx = np.array([x.Centre[0]*10000 for x in mapii.traffics])
ly = np.array([x.Centre[1]*10000 for x in mapii.traffics])

lx = lx - min(lx) + 25
ly = ly - min(ly) + 25

c = pkit.centKit([ly, lx])

c.stepPass(range(0, len(mapii.traffics)), 100, 100);#len(v[0]) ?

s.drawkit(c, max(lx)+50, max(ly)+50);

#s.fileout("./n100");

ptime("semantic group", start)

c3 = pkit.centKitIII(c, pkit.liveCentIII);

lx = np.array([x.Centre[0]*10000 for x in mapii.lanes])
ly = np.array([x.Centre[1]*10000 for x in mapii.lanes])
lx = lx - min(lx) + 25
ly = ly - min(ly) + 25


c3.member = c3.get_Member(np.array([ly, lx])); ##c.data

print("Unassigned?: ", sum(c3.member == -1));

ptime("semantic groupIII", start)

s.drawkit_sem(mapi, np.nonzero(lm)[0][np.nonzero(c3.member != -1)[0]], 2500, 1500, hA=True);

print(len(s.out) / 1024 , " kb svg");

ptime("drawkit strings", start)

ximg = Nelsa.Nelsa.xRaster(s.out)

ptime("svg -> pixels", start)

Nelsa.Nelsa.xStore(ximg, "./sem_test_group.png")



s.drawkit_semTraff(mapii, range(0, 500), 1250, 1250, hA=True); #len(mapii.traffics)

#s.fileout("./t100");

s.drawkit_semTraffLanes(mapii, range(0, 500), 1250, 1250); #len(mapii.lanes)

#s.fileout("./tL100");

ximg = Nelsa.Nelsa.xRaster(s.out)

#ptime("svg -> pixels", start)

Nelsa.Nelsa.xStore(ximg, "./tL100.png")


for x in mapii.traffics:
    
    x.Delta[0]["px"] += x.delta_Offset[0];
    x.Delta[0]["py"] += x.delta_Offset[1];
    

#s.drawkit_semTraff(mapii, range(0, len(mapii.traffics)), 2000, 2000, hA=True); #len(mapii.traffics)
#s.fileout("./t100_mappedII");


for x in mapii.lanes:
    
    x.Delta[0]["px"] += x.delta_Offset[0];
    x.Delta[0]["py"] += x.delta_Offset[1];
    
    x.Delta[1]["px"] += x.delta_Offset[0];
    x.Delta[1]["py"] += x.delta_Offset[1];
    
    

#s.semt(-2000, -2000)
s.drawkit_semTraffLanes(mapii, range(0, len(mapii.lanes)), 2000, 2000);#, hX=True); #len(mapii.lanes)
s.fileout("./l100_mappedII");

s.drawkit_semTraff(mapii, range(0, len(mapii.traffics)), 2000, 2000, hA=True); #len(mapii.traffics)
s.fileout("./t100_mappedII");



print("traffic mapII")

s.taglayer(2500, 1500);

s.layerout("./overlays");

Nelsa.Nelsa.xStore(Nelsa.Nelsa.xRaster_f("./overlays.svg"), "./overlays.png");

#!rm ./overlays.svg
#!rm ./t100_mappedII.svg
#!rm ./l100_mappedII.svg

os.remove("./overlays.svg")
os.remove("./t100_mappedII.svg")
os.remove("./l100_mappedII.svg")



#!dir -s
