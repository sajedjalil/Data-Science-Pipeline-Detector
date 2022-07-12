# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys

sys.path.append("../input/n3l5-a/");
import Nelsa


semData = """

import Nelsa
import numpy as np # linear algebra

import pymap3d


class mapII(Nelsa.mapI):
    
    def tce(self, elem):
        return elem.traffic_control_element

    def pdelta(self, elem):
        return {'px': self.cm_Norm(elem.points_x_deltas_cm), 'py': self.cm_Norm(elem.points_y_deltas_cm), 'pz': self.cm_Norm(elem.points_z_deltas_cm)};
    
    def ldelta(self, elem):
        return {'px': self.cm_Norm(elem.vertex_deltas_x_cm), 'py': self.cm_Norm(elem.vertex_deltas_y_cm), 'pz': self.cm_Norm(elem.vertex_deltas_z_cm)};
    
    
    pass;

class gpsData:
  
  def __init__(self):

    self.HC_Lat = 37.429333
    self.HC_Lng = -122.154361

    self.HC_Ex, self.HC_Ey, self.HC_Ez = pymap3d.geodetic2ecef(self.HC_Lat, 0, 0); ## try lngZero?
    
    self.degree2kmNS = self.toDist(self.ecefNorm(1, 0))[1] / 1000;
    self.degree2kmEW = self.toDist(self.ecefNorm(0, 1))[0] / 1000;
    

  def ecefNorm(self, dLat, dLng):
    
    rx1, ry1, rz1 = pymap3d.geodetic2ecef(dLat+self.HC_Lat, 0, 0); ## NS Dist // EW Centre    
    rx2, ry2, rz2 = pymap3d.geodetic2ecef(dLat+self.HC_Lat, dLng, 0); ## EW Dist @ lat    
    return rx2 - rx1, ry2 - ry1, rx1 - self.HC_Ex, rz1 - self.HC_Ez; ##EW-Fall, East-west, NS-Fall, Northsouth

    ## East-West Horizontal? // NS Direction skew

  def toDist(self, tupl):
    return [np.hypot(tupl[0], tupl[1]) *np.sign(tupl[1]), np.hypot(tupl[2], tupl[3]) *np.sign(tupl[3])]

                 
  def world2gps(self, wx, wy):
    
    gX = (wx/1000) / self.degree2kmEW;
    gY = (wy/1000) / self.degree2kmNS;

    return gY, gX

  def gps2world(self, lat, lng):
    return self.toDist(self.ecefNorm(lat - self.HC_Lat, lng - self.HC_Lng));


class mapData:
    
    def scan_Labels(self, xstr):
        return [self.map.label_Mask(i, xstr) for i in range(0, self.maplen)];
    
    def collect_Labels(self, bvec):
        return [self.map[int(x)] for x in np.nonzero(bvec)[0]];
    
    
    def delta_Offsets(self):
        for x in self.traffics:    
            #x.delta_Offset = self.map.gps_to_World(x.Centre);
            #x.delta_Offset = self.gps.gps2world(x.Centre[0], x.Centre[1]);
            x.delta_Offset = self.gps.gps2world(x.Origin[0], x.Origin[1]);
            
        for x in self.lanes:    
            #x.delta_Offset = self.map.gps_to_World(x.Centre);
            #x.delta_Offset = self.gps.gps2world(x.Centre[0], x.Centre[1]);
            x.delta_Offset = self.gps.gps2world(x.Origin[0], x.Origin[1]);

    
        
    def __init__(self, xmap):
        self.lanes = [];
        self.traffics = [];
        self.junctions = [];
        self.map = xmap; ##self.map.Nelsa_?
        self.maplen = len(xmap);
        
        self.setlanes(self.collect_Labels(self.scan_Labels("lane")));
        self.settraffic(self.collect_Labels(self.scan_Labels("traffic_control_element")));
        self.setjunction(self.collect_Labels(self.scan_Labels("junction")));
        
        for x in range(0, len(self.lanes)):
            self.lanes[x] = self.lane(self.lanes[x]);

        for x in range(0, len(self.traffics)):
            self.traffics[x] = self.traffic(self.traffics[x]);
        
        self.gps = gpsData();

        self.delta_Offsets();

    class common:        
        #def __init__(self, xob):
        #    self.obj = xob;

        def __init__(self, xob):
            self.obj = xob;
            self.Id = self.get_Id();
            self.Centre = self.get_Centroid();
        
        def get_Id(self):
            return self.obj.id.id;
        
        def get_Centroid(self):
            #print(type(self));
            #if(type(self) == mapData.traffic): print("trafficlight");
            #if(type(self) == mapData.lane): print("lane");
            bound = self.obj.bounding_box;
            
            e27 = 20000000;
            lat = (bound.south_west.lat_e7 + bound.north_east.lat_e7) / e27; #2e7
            long = (bound.south_west.lng_e7 + bound.north_east.lng_e7) / e27; #2e7
            return [lat, long];##?
        
        def get_Origin(self):
            #print(type(self));
            #origin = self.obj.element;
            if(type(self) == mapData.traffic): 
                origin = self.obj.element.traffic_control_element.geo_frame.origin;
            if(type(self) == mapData.lane): 
                origin = self.obj.element.lane.geo_frame.origin;
            
            e17 = 10000000;
            lat = origin.lat_e7 / e17; #2e7
            long = origin.lng_e7 / e17; #2e7
            return [lat, long];##?
            #pass;
            
        def get_Delta(self):
            #print(type(self), type(self.obj))
            if(type(self) == mapData.traffic):
            
                return [mapII.pdelta(self, self.obj.element.traffic_control_element)];
            
            if(type(self) == mapData.lane):
                
                lane1 = mapII.ldelta(self, self.obj.element.lane.left_boundary);
                lane2 = mapII.ldelta(self, self.obj.element.lane.right_boundary);
                
                #print(self.obj == mII[self.obj.get_Id()])
                #return mII.pdelta(mII[self.obj.get_Id()]); ##
                return [lane1, lane2];
            #pass;
            
        def cm_Norm(self, seq):    ## to metres absolute
            return np.cumsum(np.asarray(seq) / 100)

                

        
        #pass;
        
    class node(common): pass;
#        def __init__(self, xob):
#            self.obj = xob;
            ## road segments
            ## junction
            
    class segment(common): pass;
#        def __init__(self, xob):
#            self.obj = xob;
            ## start node / end node
            ## lanes

    class junction(common): pass;
#        def __init__(self, xob):
#            self.obj = xob;
            ## road net nodes
            ## traffic
            ## lanes

    class lane(common):
        def __init__(self, xob):
            self.obj = xob;
            self.Id = self.get_Id();
            self.Centre = self.get_Centroid();
            self.Origin = self.get_Origin();
            self.Delta = self.get_Delta();
            self.delta_Offset = np.array([0., 0.]);
    
    class traffic(common):
        def __init__(self, xob):
            self.obj = xob;
            self.Id = self.get_Id();
            self.Centre = self.get_Centroid();
            self.Origin = self.get_Origin();
            self.Delta = self.get_Delta();
            self.delta_Offset = np.array([0., 0.]);

    
    def setlanes(self, xl):
        self.lanes.extend(xl);
        
    def settraffic(self, xt):
        self.traffics.extend(xt);
    
    def setjunction(self, xt):
        self.junctions.extend(xt);
    
#"""

if(semData != ""):
    f = open("./semData.py", "w")
    f.write(semData);
    f.close();


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#mapi = mapII("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb", [[1,0],[0,1]])

#mapii = mapData(mapi);

