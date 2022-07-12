# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, sys

#sys.path.append("../kaggle/input/kaggle-l5kit");

import pymap3d

"""for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""        
##gps_Static = [37.429333, -122.154361];

HC_Lat = 37.429333
HC_Lng = -122.154361

HC_x, HC_y, HC_z = pymap3d.geodetic2ecef(HC_Lat, HC_Lng, 0);

HC_Ex, HC_Ey, HC_Ez = pymap3d.geodetic2ecef(HC_Lat, 0, 0);

def gpsRel(dLat, dLng):
    
    rx, ry, rz = pymap3d.geodetic2ecef(dLat+HC_Lat, HC_Lng +dLng, 0);
    
    return rx-HC_x, ry-HC_y, rz-HC_z;


def ecefRel(dLat, dLng):
    
    rx, ry, rz = pymap3d.geodetic2ecef(dLat+HC_Lat, dLng, 0);
    
    return rx-HC_Ex, ry-HC_Ey, rz-HC_Ez;


def ecefNorm(dLat, dLng):
    
    rx1, ry1, rz1 = pymap3d.geodetic2ecef(dLat+HC_Lat, 0, 0); ## NS Dist // EW Centre
    
    rx2, ry2, rz2 = pymap3d.geodetic2ecef(dLat+HC_Lat, dLng, 0); ## EW Dist @ lat
    
    return rx2-rx1, ry2-ry1, rx1-HC_Ex, rz1-HC_Ez; ##EW-Fall, East-west, NS-Fall, Northsouth

## East-West Horizontal? // NS Direction skew


def toDist(tupl):
    return [np.hypot(tupl[0], tupl[1]) *np.sign(tupl[1]), np.hypot(tupl[2], tupl[3]) *np.sign(tupl[3])]



def ecefNorm_Alt(dLat, dLng): ## East-west from central
    
    rx1, ry1, rz1 = pymap3d.geodetic2ecef(HC_Lat, dLng, 0); ## NS norm // EW dist?
    
    rx2, ry2, rz2 = pymap3d.geodetic2ecef(dLat+HC_Lat, dLng, 0); ## NS Dist @ long? (invar?)
    
    #return [rx1, rx2, ry1, ry2, rz1, rz2]

    return rx1-HC_Ex, ry1-HC_Ey, rx2-rx1, rz2-rz1; ##EW-Fall, East-west, NS-Fall, Northsouth

#rx2-rx1, ry2-ry1, rx1-HC_Ex, rz1-HC_Ez; ##EW-Fall, East-west, NS-Fall, Northsouth
## East-west distance skew

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session