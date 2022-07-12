# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 20:04:43 2017

@author: REdouane
"""

import pandas as pd
import numpy as np
from shapely import wkt
from shapely import affinity
from matplotlib.patches import Polygon, Patch
# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch
import shapely
import matplotlib.pyplot as plt
#import gdal
import sys
from shapely.geometry import MultiPolygon

polygons_raw = pd.read_csv('../input/train_wkt_v4.csv')
#print(polygons_raw.head())
grid_sizes = pd.read_csv('../input/grid_sizes.csv')
#print(grid_sizes.head())

# classes list
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
cols = grid_sizes.columns.tolist()

if cols[0]!='ImageId':
    cols[0]='ImageId'

grid_sizes.columns = cols

# imageIds in a DataFrame
allImageIds = grid_sizes.ImageId.unique()
trainImageIds = polygons_raw.ImageId.unique()

#display all imageIds
#np.set_printoptions(threshold=np.nan)
#print(trainImageIds)
#print(allImageIds)

def plot_polygons(ax, polygon, cat):
    '''
    Plot descrates.PolygonPatch from list of polygons objs for each CLASS
    '''
    if polygon !='MULTIPOLYGON EMPTY':
        print ('type = ', type(polygon))
        for pp in polygon:
            #sys.exit()
            mpl_poly = PolygonPatch(pp, color=COLORS[cat], lw=0, alpha=0.7, zorder=ZORDER[cat])
            ax.add_patch(mpl_poly)
    # ax.relim()
    ax.autoscale_view()
    ax.set_title('Objects')
    ax.set_xticks([])
    ax.set_yticks([])
    return 1
    

#process all trainImages
imgCount = 0
for imgId in trainImageIds:
    i_grid_size = grid_sizes[grid_sizes.ImageId == imgId]
    x_max = i_grid_size.Xmax.values[0]
    y_min = i_grid_size.Ymin.values[0]
    
    #load all polygones og each category
    classes=[]
    polygonsList={}
    for i in CLASSES.keys():
        #Get just a single class of training polygonsList for this image
        classes.append(polygons_raw[(polygons_raw.ImageId == imgId) & (polygons_raw.ClassType==i)])
        #WKT to shapely object
        #polygonsList.append(classes[i-1].MultipolygonWKT.values[0])
        polygonsList[i-1] = classes[i-1].MultipolygonWKT.values[0]
        
        
    print('ID=%s ============================= CLASSES =========================== ' %imgId)       
    print(classes)
    #print(imgId,' ===================== POLYGONES ===================== ')
    #print polygonsList
    j=0
    for cat in polygonsList:
        po=polygonsList[cat]
        if po !='MULTIPOLYGON EMPTY':
            #sys.exit()
            print('\n********************** POLY N:%d OF ImgID %s *********************'%(j,imgId))
            print('------------- Original Extent -----------')
            poly = wkt.loads(po)
            print(poly.bounds)     
            #Load the image and get its width and height
            #image = gdal.Open('three_band/6120_2_2.tif')
            #W = image.RasterXSize
            #H = image.RasterYSize
            #gdal is not loaded in kaggle yet
            W = 3403.
            H = 3348.
            #Transform the polygonsList 
            new_W = W * (W/(W+1))
            new_H = H * (H/(H+1))
            print ('new_W=%f new_H=%f' %(new_W,new_H))
            x_scaler = new_W / x_max
            y_scaler = new_H / y_min
            print ('x_scaler=%f y_scaler=%f' %(x_scaler,y_scaler))
            new_poly = shapely.affinity.scale(poly, xfact = x_scaler, yfact= y_scaler, origin=(0,0,0))
            print('------------ New Extent  ------------')
            print(new_poly.bounds)           
            j=j+1
            #sys.exit()            
            #plot polygonsList
            fig, axArr = plt.subplots(figsize=(10, 10))
            ax = axArr
            legend_patches = plot_polygons(ax, poly, cat+1)
            ax.set_xlim(0, x_max)
            ax.set_ylim(y_min, 0)
            ax.set_xlabel(x_max)
            ax.set_ylabel(y_min)
            plt.savefig('Objects--' + imgId + '.png')
            plt.clf()
    
    imgCount=imgCount+1
    
print('\n\n %d Images treated' %imgCount)