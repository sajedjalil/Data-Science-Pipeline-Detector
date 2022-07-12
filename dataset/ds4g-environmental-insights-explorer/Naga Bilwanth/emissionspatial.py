# Convert geo location to shape file format

import pandas as pd
import geopandas as gp
import json
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import cv2
#import gdal as gd
import rasterio as ras
import os
import numpy as np

class geoLocation():
    """
        provide geoLocation support objects
    """
    def __init__(self, generatorObj, requiredColumns):
        self.generatorObj = generatorObj
        self.csvfileName = generatorObj["data_path"]
        self.requiredColumns = requiredColumns
        self.projection = generatorObj["projection"]
        self.shapeOutputLocation = generatorObj["outputShapeFilePath"]
        self.fuelconsumedPerMWHinKG = generatorObj["fuelconsumedPerMWHinKG"]
        self.skipFuels = generatorObj["skipFuels"]

    def getGeoLocationAsDataframe(self):
        """
            Return Power plant data as Geopandas Dataframe
        """
        try:
            # required column collection
            cols_required = self.requiredColumns

            # read csv file
            df = pd.read_csv(self.csvfileName, usecols=cols_required, sep=",")

            # rename geometry column
            df.rename(columns={".geo":"geometry"}, inplace=True)

            # convert geometry to shapely supported
            for ind,row in df.iterrows():
                g = json.loads(row["geometry"])    
                df.loc[ind,"geometry"] = Point(g['coordinates'])
                
            # co-ordinate reference
            crs = {'init': 'epsg:'+ self.projection}

            df = df[~df[self.generatorObj["required_columns"]["fuel_type"]].isin(self.skipFuels)]
            df = df.reset_index()
            gpDF = gp.GeoDataFrame(df, crs=crs, geometry="geometry")

            dfFuel = pd.DataFrame().append(self.fuelconsumedPerMWHinKG, ignore_index=True).T
            #dfFuel = dfFuel.rename({"0":"fuelcons"})

            df_dummy = gpDF.merge(dfFuel, left_on=self.generatorObj["required_columns"]["fuel_type"], right_index=True)

            gpDF["fuelcons"] = df_dummy[0]

            dirOut = os.path.dirname(self.shapeOutputLocation)

            if(not os.path.exists(dirOut)):
                os.makedirs(dirOut)
            
            # save dataframe to shape file
            gpDF.to_file(self.shapeOutputLocation)

            return gpDF

        except Exception as e:
            print(e)
            return None

class emissionSpatialLayer():
    """
        provide objects that perform spatial operations to filter power plants within sub region
        provide objects that convert spatial objects to numpy objects
    """
    def __init__(self, appGlobal, layerName):
        self.appGlobal = appGlobal
        self.objAppConfig = appGlobal.objAppConfig
        self.config = self.objAppConfig.getConfig()        
        self.layername = layerName
        self.layerInfo, self.rasterRefernce, self.projection = self.objAppConfig.getlayerInfo(layerName)
        self.raster =   self.rasterRefernce + os.listdir(self.rasterRefernce)[0]
        self.layerDisplayName = self.layerInfo["displayField"]["fieldName"]
        
    def getSpatialLayerNPArr(self):
        """
            convert sub region polygon to numpy array. If provided sub region is power plant (point object), 
            it perform buffer from power plant object for a specified distance in configuration file under "layers" section
            
            Example:
            {
                "name":"powerplant_50Miles",                
                "buffer": 50
            }
        """
        try:
            if not self.layerInfo == None:
                # read the sub region polygon as geopandas dataframe
                layerDF = gp.read_file(self.layerInfo["inpath"])

                # read the raster as configured in "layers" section to fetch GeoTransform information
                # to convert geo co-ordinates to pixel co-ordinates
                #ds_r = gd.Open(self.raster)
                rs_r = ras.open(self.raster)
                #geo_trans = ds_r.GetGeoTransform()
                geo_trans = []            
                geo_trans.append(rs_r.transform[2]) # top left x
                geo_trans.append(rs_r.transform[0]) # Pixel resolution
                geo_trans.append(rs_r.transform[1]) # rotaion
                geo_trans.append(rs_r.transform[5]) # top left y
                geo_trans.append(rs_r.transform[3]) # rotation
                geo_trans.append(rs_r.transform[4]) # top left x

                npSpatialArr = np.zeros([rs_r.shape[0], rs_r.shape[1]])

                npSpatialDict = {}
                
                if not layerDF.empty:
                    # If sub region geometry is point type, it perform buffer from point for analysis purpose
                    # the buffered distance will be in miles
                    if layerDF["geometry"][0].geom_type == 'Point':
                        bufferDist = self.layerInfo["buffer"]
                        
                        # co-ordinate reference
                        layerDF.crs = {'init': 'epsg:'+ self.projection}

                        '''3959 # radius of the great circle in miles...some algorithms use 3956
                        6371 # radius in kilometers...some algorithms use 6367
                        3959 * 5280 # radius in feet
                        6371 * 1000 # radius in meters'''

                        # perform buffer using geopandas library
                        layerDF["geometry"] = layerDF["geometry"].buffer(bufferDist/self.appGlobal.ConversionToMiles)
                        layerDF.to_file(self.layerInfo["outpath"])

                    # convert layer to numpy array
                    for index, row in layerDF.iterrows():
                        if row["geometry"].geom_type == "MultiPolygon":
                            for poly in row["geometry"]:
                                #print(row[self.layerDisplayName])
                                self.draw_polygon(poly, geo_trans, npSpatialArr)

                        if row["geometry"].geom_type == "Polygon":
                            #print(row[self.layerDisplayName])
                            self.draw_polygon(row["geometry"], geo_trans, npSpatialArr)

                        npSpatialArr = np.clip(npSpatialArr, 0 ,1)
                        npSpatialDict[row[self.layerDisplayName]] = npSpatialArr

            return npSpatialDict
        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - getSpatialLayerNPArr")
            raise

    def explode(self, indf):
        """
            explode multipolygon geometry to polygon geometry
        """    
        try:    
            outdf = gp.GeoDataFrame(columns=indf.columns)

            for idx, row in indf.iterrows():
                if type(row.geometry) == Polygon:
                    outdf = outdf.append(row,ignore_index=True)
                if type(row.geometry) == MultiPolygon:
                    multdf = gp.GeoDataFrame(columns=indf.columns)
                    recs = len(row.geometry)
                    multdf = multdf.append([row]*recs,ignore_index=True)
                    for geom in range(recs):
                        multdf.loc[geom,'geometry'] = row.geometry[geom]
                    outdf = outdf.append(multdf,ignore_index=True)
            return outdf
        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - explode")
            raise

    def world_2_pixel(self, geo_trans, i, j):
        """
            convert world co-ordinates to pixel co-ordinates
        """
        try:
            ul_x = geo_trans[0]
            ul_y = geo_trans[3]
            x_dist = geo_trans[1]
            y_dist = geo_trans[5]
            x_pix = (i - ul_x) / x_dist
            y_pix = (j - ul_y) / y_dist
            return[round(x_pix), round(y_pix)]
        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - world_2_pixel")
            raise

    def convert_points(self, points, geo_trans):
        """
            convert points from world co-ordinate system to pixel co-ordinates
        """
        try:
            converted_points = []
        
            for p in points:
                cp = Point(self.world_2_pixel(geo_trans, p[0], p[1]))
                converted_points.append([cp.x, cp.y])
        
            return converted_points
        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - convert_points")
            raise

    def draw_polygon(self, polygon, geo_trans, geometryArr):
        """
            convert polygon co-ordinates to numpy array
        """
        try:
            points = polygon.exterior.coords[:]
            converted_points = self.convert_points(points, geo_trans)

            cv2.drawContours(geometryArr,
                            [np.array(converted_points, dtype=np.int32)],
                            -1, 1, thickness=-1)
                
            if polygon.interiors:
                for inner_polygon in polygon.interiors:
                    points = inner_polygon.coords[:]
                    converted_points = self.convert_points(points, geo_trans)
                    cv2.drawContours(geometryArr,
                                    [np.array(converted_points, dtype=np.int32)],
                                    -1, 0, thickness=-1)
        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - draw_polygon")
            raise
                
    def draw_point(self, pointArr, geo_trans, geometryArr):     
        """
            convert point array to numpy array
        """
        try:   
            converted_points = self.convert_points(pointArr, geo_trans)

            cv2.drawContours(geometryArr,
                            [np.array(converted_points, dtype=np.int32)],
                            -1, 1, thickness=-1)
        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - draw_point")
            raise
             
    def getGeneratorXYLocation(self, subRegion):
        """
            Get power plant GeoDataFrame identified within sub region polygon geometry
        """
        try:
            # get powerplant data as geopandas dataframe
            from emissiondataset import generatorDataset
            et = generatorDataset(self.appGlobal, "Generator")        
            geodf, columnmap = et.getEGeneratorAsGeoDataFrame()

            # read sub region polygon
            polygonlayerDF = gp.read_file(self.layerInfo["outpath"])

            # create geo dataframe for  sub region polygon
            subRegionGeneratorLoc_DF = gp.GeoDataFrame(columns=geodf.columns)
            listDF = []

            # filter polygons matching with input sub region name
            queryDf = polygonlayerDF[polygonlayerDF[self.layerDisplayName] == subRegion]

            for index, row in  queryDf.iterrows():
                if row["geometry"].geom_type == "MultiPolygon":                            
                    for poly in row["geometry"]:
                        # Query power plant within input sub region polygon
                        listDF.append(geodf[geodf.within(poly)])

                if row["geometry"].geom_type == "Polygon":
                    # Query power plant within input sub region polygon
                    listDF.append(geodf[geodf.within(row["geometry"])])      

            if len(listDF) > 0:
                subRegionGeneratorLoc_DF = pd.concat(listDF)

                if not subRegionGeneratorLoc_DF.empty:
                    subRegionGeneratorLoc_DF = gp.GeoDataFrame(subRegionGeneratorLoc_DF)

            return subRegionGeneratorLoc_DF

        except Exception as e:
            print(e)
            print("Failed at emissionspatial.py - getGeneratorXYLocation")
            raise