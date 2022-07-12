import pandas as pd
import numpy as np

from shapely.wkt import loads as wkt_loads
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch

import matplotlib.pyplot as plt
import tifffile as tiff

import pylab
# turn interactive mode on so that plots immediately
# See: http://stackoverflow.com/questions/2130913/no-plot-window-in-matplotlib
# pylab.ion()

inDir = '../input'

# Give short names, sensible colors and zorders to object types
CLASSES = {
        1 : 'Bldg',
        2 : 'Struct',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Fast H20',
        8 : 'Slow H20',
        9 : 'Truck',
        10 : 'Car',
        }
COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }
ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }

# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv')
print(df.head())

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
print(gs.head())


allImageIds = gs.ImageId.unique()
trainImageIds = df.ImageId.unique()
print(len(allImageIds))
print(len(trainImageIds))








