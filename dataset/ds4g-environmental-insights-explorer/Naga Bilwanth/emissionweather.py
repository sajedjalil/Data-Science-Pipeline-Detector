import sys, os
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import glob
import numpy.ma as ma
from emissiondataset import emissionTimeseries
from emissiondataset import generatorDataset

class weatherAnalyzer():
    def __init__(self, appGlobal, rasterType):
        self.appGlobal = appGlobal
        self.objAppConfig = appGlobal.objAppConfig
        self.config = self.objAppConfig.getConfig()
        self.bandcount = self.objAppConfig.getBandCount(rasterType)        
        self.rasterType = rasterType
        self.getBandAlias = self.objAppConfig.getBandAlias(rasterType)

    def getDatasetTimeseres(self):
        """
            Get raster dataset timeseries by reading the names of raster files in dataset directory
        """
        try:
            dTimeseries = emissionTimeseries(self.objAppConfig, self.rasterType)
            timeseries_df = dTimeseries.getTimeseriesDataFrame()
            return timeseries_df
        
        except Exception as e:
            print(e)
            print("Failed at emissionweather.py - getDatasetTimeseres")
            raise

    def updateAllBandsOfSubRegion(self,npSubRegionArr, analyzeArea):
        try:
            for band in range(self.bandcount):
                npSubRegionArr[:,:,band] = analyzeArea

            return npSubRegionArr
        except Exception as e:
            print(e)
            print("Failed at emissionweather.py - updateAllBandsOfSubRegion")
            raise

    def getWeather_completeCountry(self):
        """
            Get weather information for complete analyze region by reading raster
        """
        try:
            # 3 added for date, month and year
            npArr = np.zeros([148, 475, self.bandcount + 3], dtype=object)

            self.df_Timeseries = self.getDatasetTimeseres()

            # group by date                
            dfTime_group = self.df_Timeseries.groupby('date')

            self.getBandAlias.append('Date')
            self.getBandAlias.append('Month')
            self.getBandAlias.append('Year')
            df_concat = pd.DataFrame(columns=self.getBandAlias)

            for daTime, row_timegroup in dfTime_group:
                #print('Preparing raster data for date ' + str(daTime))

                # group by dataset to average time series if there is more than one tiff file found for one date
                dfDataset_group = row_timegroup.groupby('dataset')
                indexPos = 0
                concatType = None

                month = row_timegroup["month"]._values[0]
                year = row_timegroup["year"]._values[0]
                
                # loop through each weather raster dataset
                for fileListByDS in dfDataset_group:                
                    npDatasetArr, pos, concatType = self.getNPArrForDataset(fileListByDS, indexPos)
                    
                    if len(npDatasetArr) > 0:
                        npArr[:,:, indexPos: pos] = npDatasetArr[:,:]      
                        indexPos = pos 
                    else:
                        print("Dataset is empty for " + fileListByDS[0])

                npArr[:,:, indexPos] = str(daTime)
                npyfilePath = self.appGlobal.outputPath + str(daTime)
                if not os.path.exists(npyfilePath):
                    os.makedirs(npyfilePath)

                if concatType == "sum":
                    npconcatArr = np.sum(npArr[:,:,:-3], axis=1)  
                    npconcatArr = np.sum(npconcatArr[:,:], axis = 0)                
                
                if concatType == "average":
                    npconcatArr = np.mean(npArr[:,:,:-3], axis=1)
                    npconcatArr = np.mean(npconcatArr[:,:], axis = 0)                

                npconcatArr = np.append(npconcatArr, str(daTime))
                npconcatArr = np.append(npconcatArr, month)
                npconcatArr = np.append(npconcatArr, year)
                df_concat = df_concat.append(dict(zip(df_concat.columns, npconcatArr)), ignore_index=True)

                np.save(npyfilePath +"/" + self.rasterType, npArr)

            # data prepared for whole region fo a region
            df_concat.to_csv(self.appGlobal.outputPath + "/" + self.rasterType+".csv", index=False)   
        except Exception as e:
            print(e)
            print("Failed at emissionweather.py - getWeather_completeCountry")
            raise     


    def getWeather_subRegion(self, analyzeArea, subRegionName, subRegiontype, date=""):
        """
            Get weather information for request sub region by reading raster
        """
        try:
            # 4 added for date, month, year and region. It's used to print plot & filter to analyze area
            npArr = np.zeros([148, 475, self.bandcount + 4], dtype=object)

            npSubRegionArr = np.zeros([148, 475, self.bandcount + 4], dtype=object)

            npSubRegionArr = self.updateAllBandsOfSubRegion(npSubRegionArr, analyzeArea)
            npSubRegionArr[:,:,self.bandcount] = analyzeArea

            self.df_Timeseries = self.getDatasetTimeseres()

            if not date == "":
                self.df_Timeseries = self.df_Timeseries[self.df_Timeseries["date"] == date]
                date = "_"+date

            # group by date                
            dfTime_group = self.df_Timeseries.groupby('date')

            self.getBandAlias.append(subRegiontype)
            self.getBandAlias.append('Date')
            self.getBandAlias.append('Month')
            self.getBandAlias.append('Year')
            df_concat = pd.DataFrame(columns=self.getBandAlias)

            outputPath = self.appGlobal.outputPath + "/" + subRegiontype + date + "/" #+ subRegionName + "/"

            if not os.path.exists(outputPath):
                os.makedirs(outputPath)
            
            # Loop through each day
            for daTime, row_timegroup in dfTime_group:
                #print('Preparing raster data for date ' + str(daTime))

                # group by dataset to average time series if there is more than one tiff file found for one date
                dfDataset_group = row_timegroup.groupby('dataset')
                indexPos = 0
                concatType = None

                month = row_timegroup["month"]._values[0]
                year = row_timegroup["year"]._values[0]
                
                # loop through each weather raster dataset
                for fileListByDS in dfDataset_group:                
                    npDatasetArr, pos, concatType = self.getNPArrForDataset(fileListByDS, indexPos)
                    
                    if len(npDatasetArr) > 0:
                        npArr[:,:, indexPos: pos] = npDatasetArr[:,:]      
                        indexPos = pos 
                    else:
                        print("Dataset is empty for " + fileListByDS[0])

                npArr[:,:, indexPos] = analyzeArea

                # cliping analyze area with sub region
                npArr = npArr * npSubRegionArr

                if concatType == "sum":
                    npconcatArr = np.sum(npArr[:,:,:-4], axis=1)  
                    npconcatArr = np.sum(npconcatArr[:,:], axis = 0)                
                
                if concatType == "average":
                    npconcatArr = np.mean(npArr[:,:,:-4], axis=1)
                    npconcatArr = np.mean(npconcatArr[:,:], axis = 0)                

                npconcatArr = np.append(npconcatArr, subRegionName)
                npconcatArr = np.append(npconcatArr, str(daTime))
                npconcatArr = np.append(npconcatArr, month)
                npconcatArr = np.append(npconcatArr, year)
                df_concat = df_concat.append(dict(zip(df_concat.columns, npconcatArr)), ignore_index=True)

                #don't delete. If any analysis to be performed using numpy files un comment it
                '''npyfilePath = outputPath + str(daTime)

                if not os.path.exists(npyfilePath):
                    os.makedirs(npyfilePath)
                np.save(npyfilePath +"/" + self.rasterType, npArr)'''

            # data prepared for whole region fo a region
            df_concat.to_csv(outputPath + "/" + self.rasterType+"_"+subRegionName+".csv", index=False)
        except Exception as e:
            print(e)
            print("Failed at emissionweather.py - getWeather_subRegion")
            raise

    def getNPArrForDataset(self, fileListByDS, indexPos):
        """
            Get raster band information as numpy array list
        """
        try:
            npArr = None
            npReturnMeanArr = None
            npArrList = []
            loopIndexPos = indexPos
            index = 0
            concatType = None        

            for row in fileListByDS[1].iterrows():
                rFile = row[1]['fileName']
                loopIndexPos = indexPos

                for cElements in self.config:                    
                    if cElements == row[1]["dataset"] and self.config[cElements]["type"] == self.rasterType:
                        rasterPath = self.config[cElements]["data_Path"] 
                        concatType = self.config[cElements]["concatType"]
                        image = skimage.io.imread(rasterPath + rFile)                            
                        npArr = np.zeros([image.shape[0], image.shape[1],
                                len(self.config[cElements]["bandsConsidered"])],
                                dtype=np.float32)

                        npReturnMeanArr = npArr
                        index = 0

                        for bandElements in self.config[cElements]["bandsConsidered"]:
                            band = bandElements["band"]

                            '''if not colDesc.__contains__(bandElements['description']):
                                colDesc.append(bandElements['description'])'''

                            # assigning at band level is required since analysis is done based bands 
                            # configured in config.file
                            npArr[:,:, index] = image[:,:, band]

                            loopIndexPos += 1
                            index += 1
                        
                        # to replace nan with default values by columns
                        #npArr = np.where(np.isnan(npArr), ma.array(npArr, mask=np.isnan(npArr)).mean(axis=1), npArr)

                        # to replace nan with default values by rows
                        npArr = np.where(np.isnan(npArr), ma.array(npArr, mask=np.isnan(npArr)).mean(axis=0), npArr)
                        npArrList.append(npArr)
            
            if(len(npArrList) > 1):
                # Taking mean if more than one tiff exists for a day
                npReturnMeanArr = np.mean(npArrList, axis=0)

            elif len(npArrList) == 1:
                npReturnMeanArr = npArrList[0]

            return npReturnMeanArr, loopIndexPos, concatType

        except Exception as e:
            print(e)
            print("Failed at emissionweather.py - getNPArrForDataset")
            raise
