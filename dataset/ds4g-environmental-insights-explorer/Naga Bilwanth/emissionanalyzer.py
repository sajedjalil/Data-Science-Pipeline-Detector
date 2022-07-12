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

class emissionFactorAnalyzer():
    def __init__(self, appGlobal, rasterType):
        self.appGlobal = appGlobal
        self.objAppConfig = appGlobal.objAppConfig
        self.config = self.objAppConfig.getConfig()
        self.bandcount = self.objAppConfig.getBandCount(rasterType)        
        self.rasterType = rasterType
        self.getBandAlias = self.objAppConfig.getBandAlias(rasterType)
        self.generatorObj = self.objAppConfig.getGenerationConfigObj("Generator")
        
    def getDatasetTimeseres(self):
        """
            This method read input rasters names, analyze it and return the given timeseries
            as dataframe
        """
        try:
            dTimeseries = emissionTimeseries(self.objAppConfig, self.rasterType)
            timeseries_df = dTimeseries.getTimeseriesDataFrame()
            return timeseries_df
        except Exception as e:
            print(e)
            print("Failed at emissionanalyzer.py - getDatasetTimeseres")
            raise

    def updateAllBandsOfSubRegion(self,npSubRegionArr, analyzeArea):
        """
            Dummy update of band information, to filter polygon numpy array to sub region polygon provided.
            For the area performing analysis, it update 1 else 0
        """
        try:
            for band in range(self.bandcount):
                npSubRegionArr[:,:,band] = analyzeArea

            return npSubRegionArr
        except Exception as e:
            print(e)
            print("Failed at emissionanalyzer.py - updateAllBandsOfSubRegion")
            raise

    def getEmissionGeneraterDataset(self):
        """
            Get Power plant dataset
        """
        try:
            ge = generatorDataset(self.appGlobal, "Generator")
            gpDf, columnmap = ge.getEGeneratorAsGeoDataFrame()
            return gpDf, columnmap
        except Exception as e:
            print(e)
            print("Failed at emissionanalyzer.py - getEmissionGeneraterDataset")
            raise

    def getEGeneratorReqiredColumns(self):
        """
            Get power plant required columns
        """
        try:
            ge = generatorDataset(self.appGlobal, "Generator")
            return ge.getRequiredColumns()
        except Exception as e:
            print(e)
            print("Failed at emissionanalyzer.py - getEGeneratorReqiredColumns")
            raise

    def generateEF_CompleteRegion(self):
        """
            Generate Emission Factor for whole GeoSpatial Area
        """
        try:
        
            # 3 added for date, month & year. It's used to print plot
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
                    # -1 is Date column
                    npconcatArr = np.sum(npArr[:,:,:-3], axis=1)
                    npconcatArr = np.sum(npconcatArr[:,:], axis = 0)
                
                if concatType == "average":
                    # -1 is Date column
                    npconcatArr = np.mean(npArr[:,:,:-3], axis=1)
                    npconcatArr = np.mean(npconcatArr[:,:], axis = 0)

                npconcatArr = np.append(npconcatArr, str(daTime))
                npconcatArr = np.append(npconcatArr, month)
                npconcatArr = np.append(npconcatArr, year)
                df_concat = df_concat.append(dict(zip(df_concat.columns, npconcatArr)), ignore_index=True)

                np.save(npyfilePath +"/" + self.rasterType, npArr)

            generatorDF, columnmap = self.getEmissionGeneraterDataset()

            # converted estimated capacity per annum to each day
            activityRate = generatorDF[columnmap["estimated_capacity_in_gwh"]].sum() / 365

            aliasNames = self.objAppConfig.getEmissionAliasNames(self.rasterType)

            cnt = 0
            # sum fields configured as emission = true to calculate emission Factor
            for alias in aliasNames:
                if cnt == 0:
                    df_concat["emissionSum"] = df_concat[alias]
                    cnt = 1
                else:
                    df_concat["emissionSum"] = df_concat["emissionSum"] + df_concat[alias]

            df_concat["emissionFactor"] = df_concat["emissionSum"] / activityRate

            # Drop dummy column
            df_concat.drop(columns=["emissionSum"], inplace=True)

            # data prepared for whole region
            df_concat.to_csv(self.appGlobal.outputPath + "/" + self.rasterType+".csv", index=False)

        except Exception as e:
            print(e)
            print("Failed at emissionanalyzer.py - generateEF_CompleteRegion")
            raise

    def generateEF_SubRegion(self, analyzeArea, generatorDF, subRegionName, subRegiontype,date=""):
        """
            Generate Emission factor for Sub Region for provided polygon in numpy array format.
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
                # group by dataset to average time series if there is more than one tiff file found for one date
                dfDataset_group = row_timegroup.groupby('dataset')
                indexPos = 0
                concatType = None
                month = row_timegroup["month"]._values[0]
                year = row_timegroup["year"]._values[0]

                # Loop through each dataset group
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
                    # -1 is Date column
                    npconcatArr = np.sum(npArr[:,:,:-4], axis=1)
                    npconcatArr = np.sum(npconcatArr[:,:], axis = 0)
                
                if concatType == "average":
                    # -1 is Date column
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

            columnmap = self.getEGeneratorReqiredColumns()
            
            # converted estimated capacity per annum to each day
            activityRate = generatorDF[columnmap["estimated_capacity_in_gwh"]].sum() / 365

            aliasNames = self.objAppConfig.getEmissionAliasNames(self.rasterType)

            cnt = 0
            # sum fields configured as emission = true to calculate emission Factor
            for alias in aliasNames:
                if cnt == 0:
                    df_concat["emissionSum"] = df_concat[alias]
                    cnt = 1
                else:
                    df_concat["emissionSum"] = df_concat["emissionSum"] + df_concat[alias]

            #fuelcons = generatorDF["fuelcons"].sum()
            df_concat["emissionFactor"] = df_concat["emissionSum"] / activityRate

            # Drop dummy column
            df_concat.drop(columns=["emissionSum"], inplace=True)

            # data prepared for whole region
            df_concat.to_csv(outputPath + self.rasterType+"_"+subRegionName+".csv", index=False)

        except Exception as e:
            print(e)
            print("Failed at emissionanalyzer.py - generateEF_SubRegion")
            raise

    def getNPArrForDataset(self, fileListByDS, indexPos):
        """
            Update the Raster Band at a particular index pos in Numpy Array
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
            print("Failed at emissionanalyzer.py - getNPArrForDataset")
            raise