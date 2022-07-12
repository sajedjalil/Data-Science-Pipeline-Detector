# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, zarr, sys

import l5kit.data.proto.road_network_pb2 as roadNet;

"""IdGlob = roadNet.GlobalId

class glo_id():
    
    def __init__(self, val):        
        self.id = val.__str__();
    
    def __hash__(self):
        return hash(repr(self));
    
    def __repr__(self):
        return str(self.id);
    
    pass;

#roadNet.GlobalId = glo_id;"""


sys.path.append("../input/n3l5-a/");
sys.path.append("../input/semdata/");
sys.path.append("../input/trcent/");

#sys.path.append("../input/pysvg/");
#sys.path.append("../input/motion-import/");


import Nelsa
#import motion_Import as motion
import semData
#import pySvg
import trCent



def laneDict(xmap):
    
    ld = dict();
    
    for i in range(0, len(xmap.lanes)):
        
        ld[xmap.lanes[i].Id.__str__()] = i;

    return ld;


def laneAhead(xmap, li):
    
    return xmap.lanes[li].obj.element.lane.lanes_ahead;


def laneAdj(xmap, li):

    r = [];
    
    if(len(xmap.lanes[li].obj.element.lane.adjacent_lane_change_left.id)>0): r.append(xmap.lanes[li].obj.element.lane.adjacent_lane_change_left);
        
    if(len(xmap.lanes[li].obj.element.lane.adjacent_lane_change_right.id)>0): r.append(xmap.lanes[li].obj.element.lane.adjacent_lane_change_right);
    
    #xlanes = [xmap.lanes[li].obj.element.lane.adjacent_lane_change_left, xmap.lanes[li].obj.element.lane.adjacent_lane_change_right];
    
    return r;


    
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

mapi = semData.mapII("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb", [[1,0],[0,1]])
print(type(mapi), len(mapi.elements), "elements");

mapii = semData.mapData(mapi);

laneids = laneDict(mapii);

#laneids[laneAhead(mapii,[],8217)[0].id.__str__()]

lane_Visited = np.zeros([len(mapii.lanes)], dtype=bool);

lanes_Ahead = np.zeros([len(mapii.lanes), 8], dtype=int)-1;

lanes_Adjacent = np.zeros([len(mapii.lanes), 2], dtype=int)-1;

lanes_Behind = np.zeros([len(mapii.lanes), 8], dtype=int)-1;

lane_Class = np.zeros([len(mapii.lanes), 5], dtype=bool); ##[start end middle start_or_end active]

laneQ = [0];

#ls = laneAhead(mapii,0);

#for i in range(0,len(ls)):
    
#    lanes_Ahead[0,i] = laneids[ls[i].id.__str__()]
    
#    if not lane_Visited[lanes_Ahead[0,i]]: laneQ.append(lanes_Ahead[0,i]);

#lane_Visited[0] = True;

#nx = laneQ.pop();

#ls = laneAhead(mapii,[],nx);

#print(nx, ls)

stepi = 0;

while len(laneQ) > 0:
    
    stepi += 1;
    
    if(stepi % 400) == 0: print("step ",stepi);
    
    nx = laneQ.pop();  
    
    ls = laneAhead(mapii, nx);

    for i in range(0,len(ls)):
    
        lanes_Ahead[nx,i] = laneids[ls[i].id.__str__()]
    
        if not lane_Visited[lanes_Ahead[nx,i]]: laneQ.append(lanes_Ahead[nx,i]);
            
    lA = laneAdj(mapii, nx);

    for i in range(0,len(lA)):
    
        lanes_Adjacent[nx,i] = laneids[lA[i].id.__str__()]
    

    lane_Visited[nx] = True;

    
    
laneQ.extend(np.nonzero(lane_Visited == False)[0]);

while len(laneQ) > 0:    
    stepi += 1;
    if(stepi % 500) == 0: print("Cstep ",stepi);
    nx = laneQ.pop();  
    ls = laneAhead(mapii, nx);
    for i in range(0,len(ls)):
        lanes_Ahead[nx,i] = laneids[ls[i].id.__str__()]
        #if not lane_Visited[lanes_Ahead[nx,i]]: laneQ.append(lanes_Ahead[nx,i]);
    lA = laneAdj(mapii, nx);
    for i in range(0,len(lA)):
        lanes_Adjacent[nx,i] = laneids[lA[i].id.__str__()]
        
    lane_Visited[nx] = True;




print(np.sum(lane_Visited));

nm1 = np.sum(lanes_Ahead != -1, axis=1);

print(np.mean(nm1), np.std(nm1), np.median(nm1), np.max(nm1));

print(lanes_Ahead[0,:], lanes_Ahead[lanes_Ahead[0,:],:]);


s0 = 10;
s1 = lanes_Ahead[s0,0];
s1m = lanes_Ahead[s0,:];

s2 = lanes_Adjacent[s0,:];

lane_reVisited = np.zeros([len(mapii.lanes)], dtype=int);

escape_loop = False;

for i in range(100):
    print(i, s0, "lane", s1m,"ahead", s2, "adjacents", lane_reVisited[s1], "revisit");
    s2 = lanes_Adjacent[s1,:];
    
    lane_reVisited[s1] += 1;
    
    s0 = s1;
    
    s1m = lanes_Ahead[s1, :];
    
    if lane_reVisited[s1] > 1:        
        escape_loop = True;
    
    s1 = lanes_Ahead[s1, lane_reVisited[s1]-1];
    
    if escape_loop and s1 ==-1:
        
        s1 = lanes_Ahead[s0, lane_reVisited[s0]-2];
        
        escape_loop = False;
        
        if lane_reVisited[s1] > 6: break;
    

lanes_Behind[:,0] = 0;
    
for j in range(lanes_Ahead.shape[0]):
    
    backtr = lanes_Ahead[j,:];
    
    for k in backtr:
        
        if k!=-1:
        
            lanes_Behind[k, lanes_Behind[k,0]+1] = j;
            
            lanes_Behind[k,0] += 1;
          

    
    
    pass;

    

## parallel?

"""s1 = lanes_Ahead[0,:];
s2 = lanes_Adjacent[0,:];

for i in range(4):
    print(i, "ahead:",  s1); #, "adjacent:", s2);
    s1 = lanes_Ahead[s1,0];
    #s2 = lanes_Adjacent[s1,0];
"""    

print(len(mapii.lanes), "Tot lanes", np.sum(lane_Visited == True), "visited");

print(np.sum(lanes_Ahead != -1), "Tot laneAhead");

print(np.sum(lanes_Adjacent != -1), "Tot laneAdjacent");

print(np.sum(lanes_Behind[:,1:] != -1), "Tot laneBehind");


print(np.sum(np.sum(lanes_Behind[:,1:] != -1, axis=1)==0), "laneBehind beginnodes");

print(np.sum(np.sum(lanes_Ahead[:,:] != -1, axis=1)==0), "laneAhead Endnodes");

print(np.sum(np.sum(lanes_Ahead[:,:] != -1, axis=1)==1), "laneAhead pointnodes");

print(np.sum(np.sum(lanes_Behind[:,1:] != -1, axis=1)==1), "laneBehind pointnodes");

print(np.sum(np.sum(lanes_Ahead[:,:] != -1, axis=1)>1), "laneAhead junctnodes");

print(np.sum(np.sum(lanes_Behind[:,1:] != -1, axis=1)>1), "laneBehind junctnodes");

lane_Class[:,0] = np.sum(lanes_Behind[:,1:] != -1, axis=1) != 1;

lane_Class[:,1] = np.sum(lanes_Ahead[:,:] != -1, axis=1) != 1;

print(np.sum(np.sum(lane_Class, axis=1)==2), "lane_Class: standalone");

print(np.sum(np.sum(lane_Class, axis=1)==1), "lane_Class: start/end node", np.sum(lane_Class[:,0]), "start", np.sum(lane_Class[:,1]), "end");

print(np.sum(np.sum(lane_Class, axis=1)==0), "lane_Class: middle node");


lane_Class[:,2] = (lane_Class[:,1] | lane_Class[:,0]) == False; ## neither

lane_Class[:,3] = (lane_Class[:,1] ^ lane_Class[:,0]) == True; ## one or other

lane_Class[:,4] = (lane_Class[:,1] & lane_Class[:,0]) == False; ## not both


for k in range(0,lane_Class.shape[0], 125):
    
    if lane_Class[k,4] and lane_Class[k,0]:
    
        print(k, lane_Class[k, :], "start");
        
    if lane_Class[k,4] and lane_Class[k,1]:
    
        print(k, lane_Class[k, :], "end");
        
    
    if lane_Class[k,4] and lane_Class[k,2]:
    
        print(k, lane_Class[k, :], "middle");
        
    
    


#ptime("map init ", st);





